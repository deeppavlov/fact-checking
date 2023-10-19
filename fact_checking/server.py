import os
import random
import pickle
import logging

from sklearn.metrics import f1_score
from flask import Flask, jsonify, request
from deeppavlov import build_model
from models.utils import Trie


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))

try:
    model = build_model("fact_checker.json", download=True)
    model(["Barack Obama was born in Kazakhstan."])
    logger.info("The model is loaded.")
except Exception as e:
    logger.exception(e)
    raise e


def get_result(request):
    claims_batch = request.json.get("claims", [])
    logger.info(f"claims: {claims_batch}")
    triplets_batch, preds_batch, correct_sents_batch = model(claims_batch)
    res_batch = []
    for claim, triplets_topk, pred, corrected_claim in zip(
        claims_batch, triplets_batch, preds_batch, correct_sents_batch
    ):
        res_batch.append(
            {
                "claim": claim,
                "triplets_topk": triplets_topk,
                "pred": pred,
                "corrected_claim": corrected_claim,
            }
        )
    logger.info(f"res_batch {res_batch[:2]}")
    return res_batch


@app.route("/respond", methods=["POST"])
def respond():
    result = get_result(request)
    return jsonify(result)


@app.route("/get_metrics", methods=["POST"])
def get_metrics():
    num_samples = request.json.get("num_samples", 100)
    claim_type = request.json.get("claim_type", "all")
    filename = "/root/.deeppavlov/downloads/genie/dataset/factkg_test.pickle"
    with open(filename, "rb") as fl:
        dataset = pickle.load(fl)

    claims = []
    for claim, info_dict in dataset.items():
        label, ents, types = (
            info_dict["Label"],
            info_dict["Entity_set"],
            info_dict["types"],
        )
        if claim_type in types or claim_type == "all":
            claims.append((claim, label, types))

    random.shuffle(claims)
    claims = claims[:num_samples]
    batch_size = 20
    num_batches = len(claims) // batch_size + int(len(claims) % batch_size > 0)

    accuracy = 0
    y_pred = []
    for i in range(num_batches):
        curr_batch = claims[i * batch_size : (i + 1) * batch_size]
        input_text = [claim.strip() for claim, label, types in curr_batch]
        gold_labels = [label for claim, label, types in curr_batch]
        pred_triplets_topk, pred_labels, correct_sents = model(input_text)
        for sent, pred_triplets, lbl, pred_lbl in zip(
            input_text, pred_triplets_topk, gold_labels, pred_labels
        ):
            accuracy += int(lbl[0] == pred_lbl)
            y_pred.append(int(pred_lbl))
    y_true = [int(label[0]) for claim, label, types in claims]
    f1 = round(f1_score(y_true, y_pred) * 100, 2)
    accuracy = round(accuracy / len(claims) * 100, 2)
    return jsonify({"accuracy": accuracy, "f1": f1})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
