import torch
import torch.nn.functional as F
from allennlp.modules.attention.attention import Attention
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import weighted_sum


@Seq2VecEncoder.register("rnn-attn-induction")
class RnnSelfAttnInductionEncoder(Seq2VecEncoder):
    def __init__(self, rnn_encoder: Seq2SeqEncoder, attn_dim: int):
        super(RnnSelfAttnInductionEncoder, self).__init__()
        self.rnn_encoder = rnn_encoder
        self.attn1 = torch.nn.Linear(self.rnn_encoder.get_output_dim(), attn_dim)
        self.attn2 = torch.nn.Linear(attn_dim, attn_dim)

    def attention(self, encoded, mask):
        weights = torch.tanh(self.attn1(encoded))
        weights = self.attn2(weights)
        weights = F.softmax(weights, dim=1)
        weights = weights.transpose(1, 2)
        encoded = weights @ encoded
        encoded = torch.mean(encoded, dim=1)
        return encoded

    def get_output_dim(self) -> int:
        return self.rnn_encoder.get_output_dim()

    def get_input_dim(self) -> int:
        return self.rnn_encoder.get_input_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        encoded = self.rnn_encoder(tokens, mask)
        encoded = self.attention(encoded, mask)
        return encoded


@Seq2VecEncoder.register("rnn-self-attn")
class RnnSelfAttnEncoder(Seq2VecEncoder):
    def __init__(
        self,
        rnn_encoder: Seq2SeqEncoder,
        attn_encoder: Seq2SeqEncoder,
        boe_encoder: Seq2VecEncoder,
    ):
        super(RnnSelfAttnEncoder, self).__init__()
        self.rnn_encoder = rnn_encoder
        self.attn_encoder = attn_encoder
        self.boe_encoder = boe_encoder

    def get_input_dim(self) -> int:
        return self.rnn_encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self.boe_encoder.get_output_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        encoded = self.rnn_encoder(tokens, mask)
        encoded = self.attn_encoder(encoded, mask)
        encoded = self.boe_encoder(encoded, mask)
        return encoded


@Seq2VecEncoder.register("rnn-attn")
class RnnAttnEncoder(Seq2VecEncoder):
    def __init__(self, seq2seq_encoder: Seq2SeqEncoder, attention: Attention):
        super(RnnAttnEncoder, self).__init__()
        self.encoder = seq2seq_encoder
        self.hidden_size = self.encoder.get_output_dim() // 2
        self.attention = attention

    def get_input_dim(self) -> int:
        return self.encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self.encoder.get_output_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        encoded = self.encoder(tokens, mask)
        forward_last = encoded[:, -1, : self.hidden_size]
        backward_last = encoded[:, 0, -self.hidden_size :]
        hidden = torch.cat((forward_last, backward_last), dim=1)
        weights = self.attention(hidden, encoded, mask)
        encoded = weighted_sum(encoded, weights)
        return encoded
