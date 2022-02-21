from typing import Dict, Optional

import allennlp.nn.util as util
import torch
from allennlp.data import TextFieldTensors
from allennlp.models.heads import Head
from allennlp.modules.backbones.backbone import Backbone
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from torchmetrics import R2Score


@Backbone.register("seq2seq_backbone")
class Seq2SeqBackbone(Backbone):
    def __init__(
        self,
        embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        embedding_dropout: float = 0.0,
    ):
        self.embedder = embedder
        self.encoder = encoder
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout, inplace=True)

    def forward(self, tokens: TextFieldTensors) -> Dict[str, torch.Tensor]:
        embedded_text = self.embedder(tokens)
        embedded_text = self.embedding_dropout(embedded_text)
        mask = util.get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text, mask)
        outputs = {"encoded_text": encoded_text, "encoded_text_mask": mask}
        return outputs


@Head.register("regression_head")
class RegressionHead(Head):
    def __init__(
        self,
        encoder: Seq2VecEncoder,
        feedforward: FeedForward = None,
        dropout: float = 0.0,
    ):
        super.__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.feedforward = feedforward
        if not self.feedforward:
            self.predict = torch.nn.Linear(encoder.get_output_dim(), 1)
        else:
            self.predict = torch.nn.Linear(feedforward.get_output_dim(), 1)

        self.loss = torch.nn.MSELoss()
        self.r2 = R2Score()

    def forward(
        self,
        encoded_text: torch.FloatTensor,
        encoded_text_mask: torch.BoolTensor,
        score: torch.FloatTensor,
    ) -> Dict[str, torch.Tensor]:
        encoding = self.encoder(encoded_text, encoded_text_mask)
        encoding = self.dropout(encoding)
        if self.feedforward:
            encoding = self.feedforward(encoding)
        preds = self.predict(encoding)
        outputs = {"preds": preds}
        if score:
            loss = self.loss(preds, score)
            outputs["loss"] = loss
            self.r2.update(preds, score)
        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        r2 = self.r2.compute().item()
        if reset:
            self.r2.reset()
        return {"r2": r2}


@Head.register("tagger_head")
class Seq2SeqHead(Head):
    def __init__(
        self,
        encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        feedforward: FeedForward = None,
        label_smoothing: float = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = torch.nn.Dropout(dropout)
        self.feedforward = feedforward
        self.label_smoothing = label_smoothing

        self.loss = sequence_cross_entropy_with_logits
        self.accuracy = CategoricalAccuracy()
        self.accuracy3 = CategoricalAccuracy(top_k=3)

    def forward(
        self,
        encoded_text: torch.FloatTensor,
        encoded_text_mask: torch.BoolTensor,
        tags: torch.FloatTensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.encoder(encoded_text, encoded_text_mask)
        if self.feedforward:
            logits = self.dropout(logits)
            logits = self.feedforward(logits)
        outputs = {"logits": logits}
        if tags:
            loss = self.loss(logits, tags, encoded_text_mask, label_smoothing=self.label_smoothing)
            outputs["loss"] = loss
            self.accuracy(logits, tags, encoded_text_mask)
            self.accuracy3(logits, tags, encoded_text_mask)
        return outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset),
            "accuracy3": self.accuracy3.get_metric(reset),
        }
