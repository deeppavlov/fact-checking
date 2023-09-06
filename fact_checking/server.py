import os
import logging

from flask import Flask, jsonify, request
from deeppavlov import build_model
from models.utils import Trie


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

PORT = int(os.getenv("PORT"))

logger.info("STARTING...")
try:
    Trie(["a1", "b2"])
    model = build_model("fact_checker.json", download=False)
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
    for claim, triplets_topk, pred, corrected_claim in \
            zip(claims_batch, triplets_batch, preds_batch, correct_sents_batch):
        res_batch.append({"claim": claim, "triplets_topk": triplets_topk, "pred": pred,
                                    "corrected_claim": corrected_claim})
    logger.info(f"res_batch {res_batch[:2]}")
    return res_batch


@app.route("/respond", methods=["POST"])
def respond():
    result = get_result(request)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)