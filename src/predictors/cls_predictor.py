from typing import Dict, List

import numpy
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.fields import LabelField
from allennlp.predictors import Predictor


@Predictor.register("predictor")
class CLSPredictor(Predictor):
    def predict(self, tokens: str) -> JsonDict:
        return self.predict_json({"tokens": tokens})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict["tokens"].split()
        return self._dataset_reader.text_to_instance(tokens)

    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
