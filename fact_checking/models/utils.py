import os
import re
import pickle
import json
import jsonlines
from collections import defaultdict
from logging import getLogger

log = getLogger(__name__)

class Trie(object):
    def __init__(self, sequences):
        next_sets = defaultdict(list)
        for seq in sequences:
            if len(seq) > 0:
                next_sets[seq[0]].append(seq[1:])

        self._leaves = {k: Trie(v) for k, v in next_sets.items()}

    def get(self, indices):
        if len(indices) == 0:
            return list(self._leaves.keys())
        elif indices[0] not in self._leaves:
            return []
        else:
            return self._leaves[indices[0]].get(indices[1:])

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            trie = pickle.load(f)
        return trie


class DummyTrie(object):
    def __init__(self, return_values):
        self._return_values = return_values

    def get(self, indices=None):
        return self._return_values


class TripletUtils(object):
    @staticmethod
    def convert_text_sequence_to_text_triples(text, verbose=False, return_set=True):
        text_parts = [element.strip() for element in re.split(r"<sub>|<rel>|<obj>|<et>", text) if element.strip()]
        if verbose and len(text_parts) % 3 != 0:
            log.warning(f"Textual sequence: ```{text}``` does not follow the <sub>, <rel>, <obj>, <et> format!")

        text_triples = [tuple(text_parts[i : i + 3]) for i in range(0, len(text_parts) - 2, 3)]

        if not return_set:
            return text_triples

        unique_text_triples = list(set(text_triples))

        if verbose and len(unique_text_triples) != len(text_triples):
            log.warning(f"Textual sequence: ```{text}``` has duplicated triplets!")

        return unique_text_triples

    @staticmethod
    def triples_to_output_format(triples):
        output_triples = []

        for t in triples:
            sub, rel, obj = t

            formatted_triple = "{} {}{} {}{} {}{}".format(
                " <sub>", sub.strip(), " <rel>", rel.strip(), " <obj>", obj.strip(), " <et>"
            )
            output_triples.append(formatted_triple)

        output = "".join(output_triples)
        return output


class WikidataID2SurfaceForm(object):
    def __init__(self, path):
        self.path = path
        self.id2surface_form = {}
        self.surface_form2id = {}

    def load(self):
        logger.info(f"Reading mapping from: {self.path}")
        id2surface_form = {}
        with jsonlines.open(self.path) as f:
            for e in f:
                wikidata_id = e["wikidata_id"]
                info = e["information"]

                assert wikidata_id not in id2surface_form, "Duplicate Wikidata IDs"
                id2surface_form[wikidata_id] = info

        self.id2surface_form = id2surface_form
        self.construct_surface_form2id(verbose=False)

    def construct_surface_form2id(self, verbose=True):
        surface_form2id = {}

        for _id, info_obj in self.id2surface_form.items():
            surface_form, provenance = self._get_surface_form_from_info_obj(info_obj)
            surface_form2id[surface_form] = _id

        self.surface_form2id = surface_form2id

    @staticmethod
    def _get_surface_form_from_info_obj(info_obj):
        if "en_title" in info_obj:
            surface_form = info_obj["en_title"]
            provenance = "en_title"
        elif "en_label" in info_obj:
            surface_form = info_obj["en_label"]
            provenance = "en_label"
        else:
            raise Exception("Unexpected keys in info object:", info_obj)

        return surface_form, provenance