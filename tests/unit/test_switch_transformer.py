from allennlp.nn.util import get_mask_from_sequence_lengths

from src.modules.switch_transformer import *


def test_switch_transformer():
    feed = FeedForwardWrapper(64, 64)
    switch_feed = SwitchFeedForward(
        capacity_factor=1.0,
        drop_tokens=True,
        is_scale_prob=True,
        n_experts=10,
        expert=feed,
        d_model=64,
    )
    attn = MultiHeadAttentionWrapper(4, 64)
    layer = SwitchTransformerLayer(64, attn, switch_feed, dropout1=0.1, dropout2=0.1)
    model = SwitchTransformer(layer, 1)

    x = torch.randn(128, 25, 64)
    lengths = torch.randint(1, 26, (128,))
    mask = get_mask_from_sequence_lengths(lengths, max(lengths))

    out = model(x, mask)
    pass
