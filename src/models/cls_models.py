from typing import Dict

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import util
from allennlp.training.metrics import Auc, CategoricalAccuracy, F1Measure
from torchmetrics import MatthewsCorrcoef, Specificity


@Model.register("classifier")
class Classifier(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        dropout: float = 0.0,
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(positive_label=1)
        self.auc = Auc(positive_label=1)
        self.mcc = MatthewsCorrcoef(num_classes=2)
        self.spec = Specificity(num_classes=2, average=None)

    def forward(
        self, tokens: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(self.dropout(encoded_text))
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            self.f1(logits, label)
            self.auc(torch.argmax(logits, dim=-1), label)
            self.mcc.update(torch.argmax(logits, dim=-1), label)
            self.spec.update(torch.argmax(logits, dim=-1), label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1 = self.f1.get_metric(reset)
        mcc = self.mcc.compute().item()
        spec = self.spec.compute()[0].item()
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


@Model.register("classifier_tag")
class ClassifierTag(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(positive_label=1)
        self.auc = Auc(positive_label=1)
        self.mcc = MatthewsCorrcoef(num_classes=2)
        self.spec = Specificity(num_classes=2, average=None)

    def forward(
        self, tokens: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(tokens)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(tokens)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            self.f1(logits, label)
            self.auc(torch.argmax(logits, dim=-1), label)
            self.mcc.update(torch.argmax(logits, dim=-1), label)
            self.spec.update(torch.argmax(logits, dim=-1), label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1 = self.f1.get_metric(reset)
        mcc = self.mcc.compute().item()
        spec = self.spec.compute()[0].item()
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
