
import os
import math
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import torch
from overrides import overrides
from torch.nn import BCEWithLogitsLoss
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModel, AutoTokenizer
from transformers import BartConfig, BartForConditionalGeneration, AutoTokenizer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from .utils import Trie, DummyTrie, TripletUtils
from .constrained_generation import get_information_extraction_prefix_allowed_tokens_fn_hf

log = getLogger(__name__)

import faulthandler

faulthandler.enable()

class GenieHF(BartForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, model_name_or_path, return_dict=True, other_parameters=None):
        config = BartConfig.from_pretrained(model_name_or_path, return_dict=return_dict)
        if model_name_or_path!= "martinjosifoski/genie-rw":
            model = BartForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
        else:
            model = cls(config)

        if other_parameters is not None:
            for key, value in other_parameters.items():
                setattr(config, key, value)

        return model, config


def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    target_attention_mask: torch.Tensor,
    epsilon: float,
    ignore_index: int = None,
    reduce: bool = True,
):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target.clamp_min_(0)

        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    num_tokens = target_attention_mask.sum()
    loss, nll_loss = loss / num_tokens, nll_loss / num_tokens

    return loss, nll_loss


def convert_output_to_triplets(output_obj):
    if isinstance(output_obj[0], str):
        output = []
        for text in output_obj:
            triplets = TripletUtils.convert_text_sequence_to_text_triples(text)
            output.append(triplets)
        return output
    for sample in output_obj:
        sample["textual_triplets"] = TripletUtils.convert_text_sequence_to_text_triples(sample["text"])
    return output_obj



@register('genie_bart_generation')
class GenIEBARTGeneration(TorchModel):

    def __init__(self, model_name,
                 optimizer,
                 optimizer_parameters,
                 lr_scheduler,
                 lr_scheduler_parameters,
                 free_generation: bool,
                 bos_as_first_token_generated: bool,
                 load_path: str,
                 relation_trie_path: str,
                 entity_trie_path: Optional[str] = None,
                 generation_params: Optional[Dict] = None,
                 thresh: int = -0.25,
                 device: str = "cuda",
                 **kwargs) -> None:

        self.free_generation = free_generation
        self.generation_params = generation_params
        self.bos_as_first_token_generated = bos_as_first_token_generated
        self.model_name = model_name
        self.convert_to_triplets = True
        self.thresh = thresh
        
        self.relation_trie_path = os.path.expanduser(relation_trie_path)
        self.entity_trie_path = os.path.expanduser(entity_trie_path)
        self.device=device

        if generation_params is None:
            self.generation_params = {
                "num_beams": 3,
                "num_return_sequences": 3,
                "return_dict_in_generate": True,
                "output_scores": True,
            }

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         lr_scheduler=lr_scheduler,
                         lr_scheduler_parameters=lr_scheduler_parameters,
                         **kwargs)
    
    def train_on_batch(self, batch_x, batch_y):
        self.model.train()
        batch_x = {key: value.to(self.device) for key, value in batch_x.items()}
        batch_y = {key: value.to(self.device) for key, value in batch_y.items()}

        model_output = self.model(
            input_ids=batch_x["input_ids"],
            attention_mask=batch_x["attention_mask"],
            labels=batch_y["input_ids"],
            decoder_attention_mask=batch_y["attention_mask"],
            use_cache=False,
        )
        
        logits = model_output.logits
        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch_y["input_ids"],
            batch_y["attention_mask"],
            epsilon=1e-10,
            ignore_index=self.tokenizer.pad_token_id,
        )

        return {'loss': loss.item(), "nll_loss": nll_loss.item()}


    def sample(self, input_data, prefix_allowed_tokens_fn=None, **kwargs):

        self.model.eval()
        with torch.no_grad():
            input_ids = input_data["input_ids"]
            attention_mask = input_data["attention_mask"]
            raw_input = None

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            no_repeat_ngram_size=self.generation_params.pop("no_repeat_ngram_size", 0),
            max_length=self.generation_params.pop("max_length", 256),
            early_stopping=self.generation_params.pop("early_stopping", False),
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            **self.generation_params,
            )

        k = self.generation_params.get("num_return_sequences", 1)
        if self.generation_params.get("return_dict_in_generate", False):
            output["sequences"] = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            output["sequences_scores"] = output["sequences_scores"].tolist()
            assert len(output["sequences"]) == len(output["sequences_scores"])

            batch = [
                (output["sequences"][i : i + k], output["sequences_scores"][i : i + k])
                for i in range(0, len(output["sequences"]), k)
            ]
            output = []

            for seqs, scores in batch:
                output_obj = [{"text": seq, "log_prob": score} for seq, score in zip(seqs, scores)]
                if self.convert_to_triplets:
                    output_obj = convert_output_to_triplets(output_obj)
                output_obj = sorted(output_obj, key=lambda x: x["log_prob"], reverse=True)
                output.append(output_obj)
            return output

        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        if self.convert_to_triplets:
            output = convert_output_to_triplets(output)
        output = [output[i: i + k] for i in range(0, len(output), k)]
        return output
        
    def __call__(self, batch_x: Dict[str, torch.tensor]):

        batch_x = {key: value.to(self.device) for key, value in batch_x.items()}
        if self.free_generation:
            outputs = self.sample(batch_x)
        else:
            raw_input = None
            prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                self,
                raw_input,
                bos_as_first_token_generated=self.bos_as_first_token_generated,
                entities_trie=self.tries.get("entity_trie", None),
                relations_trie=self.tries.get("relation_trie", None),
            )
            outputs = self.sample(
                batch_x,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        if self.generation_params.get("return_dict_in_generate"):
            preds_batch = []
            for preds in outputs:
                top_preds = []
                for lpred in preds:
                    score = lpred.get("log_prob", -100)
                    pred = lpred.get("textual_triplets", [])
                    if score > self.thresh and not (pred[0] == "unk" and pred[2] == "unk"):
                        top_preds.append(pred)
                preds_batch.append(top_preds)
            return preds_batch
        else:
            return outputs

    @overrides
    def load(self, fname: str = None):
        if fname is not None:
            self.load_path = fname

        self.model, self.config = GenieHF.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        checkpoint = torch.load(f"{self.load_path}.pth.tar", map_location=self.device)
        model_state = checkpoint["model_state_dict"]
        self.model.load_state_dict(model_state, strict=True)
        self.model.to(self.device)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        relation_trie = Trie.load(self.relation_trie_path)
        entity_trie = Trie.load(os.path.expanduser(self.entity_trie_path))
        self.tries = {'relation_trie': relation_trie, 'entity_trie': entity_trie}
        log.info("Everythin loaded succesfully!")
