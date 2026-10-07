"""Mask2Former slide predictions must use the crop's spatial resolution."""
import torch.nn.functional as F
from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS


@MODELS.register_module()
class SlidingMask2FormerHead(Mask2FormerHead):
    def predict(self, x, batch_img_metas, test_cfg):
        if test_cfg.get('mode', 'whole') == 'slide':
            size = tuple(test_cfg['crop_size'])
            # The upstream head prioritizes pad_shape. Full-frame metadata
            # would expand all 100 query masks to 4K inside every crop.
            batch_img_metas = [dict(meta, pad_shape=size, img_shape=size)
                               for meta in batch_img_metas]
        logits = super().predict(x, batch_img_metas, test_cfg)
        if test_cfg.get('mode', 'whole') == 'slide':
            if logits.shape[-2:] != size:
                logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        return logits
