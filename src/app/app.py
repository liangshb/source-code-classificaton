import argparse
import json
import logging
from typing import Dict

from allennlp.interpret import Attacker, SaliencyInterpreter
from allennlp.interpret.attackers import Hotflip, InputReduction
from allennlp.interpret.saliency_interpreters import (
    IntegratedGradient,
    SimpleGradient,
    SmoothGradient,
)
from allennlp.predictors import Predictor
from flask import Flask, Response, jsonify, render_template, request

logger = logging.getLogger(__name__)


def make_app(
    predictor: Predictor,
    saliency_interpreters: Dict[str, SaliencyInterpreter],
    attackers: Dict[str, Attacker],
):
    app = Flask(__name__)

    @app.route("/")
    def index() -> Response:
        return render_template("index.html")

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict() -> Response:
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()
        prediction = predictor.predict_json(data)
        log_blob = {"inputs": data, "outputs": prediction}
        logger.info("prediction: %s", json.dumps(log_blob))
        return jsonify(prediction)

    @app.route("/saliency", methods=["POST", "OPTIONS"])
    def saliency() -> Response:
        if request.method == "OPTIONS":
            return Response(response="", status=200)
        data = request.get_json()
        interpreter = saliency_interpreters.get(data.pop("gradient"))
        res = interpreter.saliency_interpret_from_json(data)
        log_blob = {"inputs": data, "outputs": res}
        logger.info("prediction: %s", json.dumps(log_blob))
        return jsonify(res)

    @app.route("/attack", methods=["POST", "OPTIONS"])
    def attack() -> Response:
        if request.method == "OPTIONS":
            return Response(response="", status=200)
        data = request.get_json()
        attacker = attackers.get(data.pop("attacker"))
        res = attacker.attack_from_json(data)
        log_blob = {"inputs": data, "outputs": res}
        logger.info("prediction: %s", json.dumps(log_blob))
        return jsonify(res)

    @app.route("/predict_batch", methods=["POST", "OPTIONS"])
    def predict_batch() -> Response:
        """make a prediction using the specified model and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        data = request.get_json()
        prediction = predictor.predict_batch_json(data)
        return jsonify(prediction)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-p", "--predictor", type=str, default="predictor")
    parser.add_argument("-f", "--field-name", type=str, default="tokens")
    parser.add_argument("-t", "--title", type=str, default="Demo")
    args = parser.parse_args()
    predictor = Predictor.from_path(archive_path=args.path, predictor_name=args.predictor)
    saliency_interpreters = {
        "simple": SimpleGradient(predictor),
        "integrated": IntegratedGradient(predictor),
        "smooth": SmoothGradient(predictor),
    }
    attackers = {"hotflip": Hotflip(predictor), "input_reduction": InputReduction(predictor)}
    app = make_app(predictor, saliency_interpreters, attackers)
    app.run()


if __name__ == "__main__":
    main()
