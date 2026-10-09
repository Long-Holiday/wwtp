"""MiT attention that does not materialize unused attention weights."""

from __future__ import annotations

from mmseg.models.backbones import MixVisionTransformer
from mmseg.registry import MODELS
from torch import nn


class WeightFreeMultiheadAttention(nn.MultiheadAttention):
    """Keep PyTorch's attention API, defaulting to its SDPA execution path."""

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        return super().forward(
            query, key, value, key_padding_mask=key_padding_mask,
            need_weights=need_weights, attn_mask=attn_mask,
            average_attn_weights=average_attn_weights, is_causal=is_causal)


@MODELS.register_module()
class MixVisionTransformerSDPA(MixVisionTransformer):
    """Preserve upstream MiT/state-dict structure and let PyTorch choose SDPA.

    MMSeg's efficient attention discards returned weights, but its ordinary
    forward uses PyTorch's ``need_weights=True`` default. Replacing only that
    leaf retains spatial reduction, layouts, dropout and pretrained key names.
    Explicit requests for attention weights still work.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for stage in self.layers:
            for block in stage[1]:
                original = block.attn.attn
                reference = next(original.parameters())
                replacement = WeightFreeMultiheadAttention(
                    original.embed_dim, original.num_heads,
                    dropout=original.dropout,
                    bias=original.in_proj_bias is not None,
                    add_bias_kv=original.bias_k is not None,
                    add_zero_attn=original.add_zero_attn,
                    kdim=original.kdim, vdim=original.vdim,
                    batch_first=original.batch_first,
                    device=reference.device, dtype=reference.dtype)
                replacement.load_state_dict(original.state_dict(), strict=True)
                replacement.train(original.training)
                block.attn.attn = replacement
