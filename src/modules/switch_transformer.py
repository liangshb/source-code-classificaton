from abc import ABC

import torch
import torch.nn as nn
from allennlp.common import FromParams
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.transformer.positional_encoding import SinusoidalPositionalEncoding
from allennlp.nn import Activation
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.utils import clone_module_list


class FeedForwardWrapper(FeedForward, FromParams, ABC):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            dropout,
            Activation.by_name(activation)(),
            is_gated,
            bias1,
            bias2,
            bias_gate,
        )


class SwitchFeedForward(nn.Module, FromParams, ABC):
    def __init__(
        self,
        *,
        capacity_factor: float,
        drop_tokens: bool,
        is_scale_prob: bool,
        n_experts: int,
        expert: FeedForwardWrapper,
        d_model: int
    ):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = clone_module_list(expert, n_experts)
        # Routing layer and softmax
        self.switch = nn.Linear(d_model, n_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """

        # Capture the shape to change shapes later
        seq_len, batch_size, d_model = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, d_model)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        route_prob = self.softmax(self.switch(x))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [
            torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)
        ]

        # Initialize an empty tensor to store outputs
        final_output = x.new_zeros(x.shape)

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[seq_len, batch_size, d_model]`
        final_output = final_output.view(seq_len, batch_size, d_model)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logging
        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max


class MultiHeadAttentionWrapper(MultiHeadAttention, FromParams, ABC):
    def __init__(
        self,
        num_heads: int,
        input_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(num_heads, input_dim, dropout, bias)


class SwitchTransformerLayer(nn.Module, FromParams, ABC):
    def __init__(
        self,
        input_dim: int,
        attn: MultiHeadAttentionWrapper,
        feed_forward: SwitchFeedForward,
        dropout1: float,
        dropout2: float,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.norm_self_attn = nn.LayerNorm([input_dim])
        self.norm_ff = nn.LayerNorm([input_dim])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout1(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the switching feed-forward network
        ff, counts, route_prob, n_dropped, route_prob_max = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout2(ff)

        return x, counts, route_prob, n_dropped, route_prob_max

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.input_dim


@Seq2SeqEncoder.register("switch-transformer")
class SwitchTransformer(nn.Module, FromParams, ABC):
    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        self.input_dim = layer.get_input_dim()
        self.output_dim = layer.get_output_dim()
        self.post_encoding = SinusoidalPositionalEncoding()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.input_dim])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mask = torch.permute(mask, (1, 0))
        mask = torch.unsqueeze(mask, 0)
        x = self.post_encoding(x)
        x = torch.permute(x, (1, 0, 2))
        # Run through each transformer layer
        counts, route_prob, n_dropped, route_prob_max = [], [], [], []
        for layer in self.layers:
            x, f, p, n_d, p_max = layer(x=x, mask=mask)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
            route_prob_max.append(p_max)
        # Finally, normalize the vectors
        x = self.norm(x)
        #
        # return (
        #     x,
        #     torch.stack(counts),
        #     torch.stack(route_prob),
        #     n_dropped,
        #     torch.stack(route_prob_max),
        # )
        return torch.permute(x, (1, 0, 2))

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim
