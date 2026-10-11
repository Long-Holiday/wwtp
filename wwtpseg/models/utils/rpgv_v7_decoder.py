"""V7-only structure injection at F16/F8; historical decoder APIs stay unchanged."""
from torch.nn import functional as F

from .rpgv_v6_decoder import SpatialGeometryDecoderV6


class SpatialGeometryDecoderV7(SpatialGeometryDecoderV6):
    def forward(self, features, fusion_callback=None, structure_callback=None,
                return_pyramid=False):
        if len(features) != 4:
            raise ValueError('decoder expects four RGB feature maps')
        p4, p8, p16, p32 = [projection(feature)
                            for projection, feature in zip(self.projections, features)]
        if self.use_context:
            context = []
            for grid, projection in zip((1, 2, 4), self.context_projections):
                pooled = F.adaptive_avg_pool2d(features[-1], (
                    min(grid, features[-1].shape[-2]), min(grid, features[-1].shape[-1])))
                context.append(self._up(projection(pooled), p32.shape[-2:]))
            p32 = (p32 + sum(context)) / (1 + len(context))
        # Keep memory layout identical with/without callbacks, including Q=0 fallback.
        f16 = (p16 + self._up(p32, p16.shape[-2:])).contiguous()
        if structure_callback is not None:
            f16 = structure_callback(0, f16)
        f8 = (p8 + self._up(f16, p8.shape[-2:])).contiguous()
        if structure_callback is not None:
            f8 = structure_callback(1, f8)
        if fusion_callback is not None:
            f8 = fusion_callback(1, f8, p8)
        if self.spatial_refinement:
            f8 = self._spatial(self.spatial8, f8)
        f4 = (p4 + self._up(f8, p4.shape[-2:])).contiguous()
        if fusion_callback is not None:
            f4 = fusion_callback(0, f4, p4)
        if self.spatial_refinement:
            f4 = self._spatial(self.spatial4, f4)
        decoded = f4
        for block in self.refine:
            decoded = F.gelu(block.block(decoded) + block.shortcut(decoded), approximate='tanh')
        if return_pyramid:
            return decoded, (f4, f8, f16, p32)
        return decoded
