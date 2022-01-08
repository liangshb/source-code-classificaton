from typing import Dict, Union

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure
from torch import nn
from torchmetrics import MatthewsCorrcoef, Specificity


@Model.register("simple_bcn")
class SimpleBiattentiveClassificationNetwork(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        embedding_dropout: float,
        pre_encode_feedforward: FeedForward,
        encoder: Seq2SeqEncoder,
        integrator: Seq2SeqEncoder,
        integrator_dropout: float,
        output_layer: Union[FeedForward, Maxout],
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._integrator = integrator
        self._integrator_dropout = nn.Dropout(integrator_dropout)

        self._combined_integrator_output_dim = self._integrator.get_output_dim()
        self._self_attentive_pooling_projection = nn.Linear(
            self._combined_integrator_output_dim, 1
        )
        self._output_layer = output_layer

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            self._pre_encode_feedforward.get_input_dim(),
            "text field embedder output dim",
            "Pre-encoder feedforward input dim",
        )

        check_dimensions_match(
            self._pre_encode_feedforward.get_output_dim(),
            self._encoder.get_input_dim(),
            "Pre-encoder feedforward output dim",
            "Encoder input dim",
        )
        check_dimensions_match(
            self._encoder.get_output_dim() * 3,
            self._integrator.get_input_dim(),
            "Encoder output dim * 3",
            "Integrator input dim",
        )
        check_dimensions_match(
            self._integrator.get_output_dim() * 4,
            self._output_layer.get_input_dim(),
            "Integrator output dim * 4",
            "Output layer input dim",
        )

        check_dimensions_match(
            self._output_layer.get_output_dim(),
            self._num_classes,
            "Output layer output dim",
            "Number of classes.",
        )

        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(positive_label=1)
        self.auc = Auc(positive_label=1)
        self.mcc = MatthewsCorrcoef(num_classes=2)
        self.spec = Specificity(num_classes=2, average=None)
        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(
        self,
        tokens: TextFieldTensors,
        label: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        text_mask = util.get_text_field_mask(tokens)
        embedded_text = self._text_field_embedder(tokens)
        dropped_embedded_text = self._embedding_dropout(embedded_text)
        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)
        encoded_tokens = self._encoder(pre_encoded_text, text_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.masked_softmax(attention_logits, text_mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat(
            [encoded_tokens, encoded_tokens - encoded_text, encoded_tokens * encoded_text], 2
        )
        integrated_encodings = self._integrator(integrator_input, text_mask)

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
            integrated_encodings,
            text_mask.unsqueeze(2),
            util.min_value_of_dtype(integrated_encodings.dtype),
        )
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
            integrated_encodings,
            text_mask.unsqueeze(2),
            util.max_value_of_dtype(integrated_encodings.dtype),
        )
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
            integrated_encodings
        ).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)

        logits = self._output_layer(pooled_representations_dropped)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if label is not None:
            self.accuracy(logits, label)
            self.f1(logits, label)
            self.auc(torch.argmax(logits, dim=-1), label)
            self.mcc.update(torch.argmax(logits, dim=-1), label)
            self.spec.update(torch.argmax(logits, dim=-1), label)
            output_dict["loss"] = self.loss(logits, label)
        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict["label"] = labels
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1 = self.f1.get_metric(reset)
        mcc = self.mcc.compute().item()
        spec = self.spec.compute()[1].item()
        if reset:
            self.mcc.reset()
            self.spec.reset()
        return {
            "accuracy": self.accuracy.get_metric(reset),
            "precision": f1.get("precision"),
            "recall": f1.get("recall"),
            "f1": f1.get("f1"),
            "auc": self.auc.get_metric(reset),
            "mcc": mcc,
            "spec": spec,
        }
