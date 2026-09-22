"""Multi-class CBR-Net baseline for remote sensing benchmarks (e.g. ISPRS Potsdam).

Adapted from HaonanGuo/CBRNet with multi-class semantic segmentation heads,
class-agnostic boundary guidance, and 8-direction boundary correction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_BN_Weights, vgg16_bn


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(
            deep, size=skip.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.conv(torch.cat((skip, deep), dim=1))


def _seg_head(in_channels: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, num_classes, 1),
    )


def _edge_head(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 1, 1),
    )


class CBRNetMultiClass(nn.Module):
    """Multi-class CBR-Net architecture with 8-direction boundary correction."""

    DIRECTIONS = (
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
    )

    def __init__(
        self, num_classes: int = 6, pretrained: bool = True
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg16_bn(weights=weights).features
        self.encoder = nn.ModuleList(
            features[start:end]
            for start, end in ((0, 5), (5, 12), (12, 22), (22, 32), (32, 42))
        )
        self.coarse_head = _seg_head(512, num_classes)
        self.ups = nn.ModuleList(
            (
                _Up(1024, 256),
                _Up(512, 128),
                _Up(256, 64),
                _Up(128, 64),
            )
        )
        self.seg_heads = nn.ModuleList(
            _seg_head(channels + num_classes, num_classes)
            for channels in (256, 128, 64, 64)
        )
        self.edge_heads = nn.ModuleList(
            _edge_head(channels) for channels in (256, 128, 64, 64)
        )
        self.direction_head = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 8, 1),
        )
        shifts = torch.zeros(8, 1, 3, 3)
        for index, (dy, dx) in enumerate(self.DIRECTIONS):
            shifts[index, 0, 1 + dy, 1 + dx] = 1.0
        self.register_buffer("shift_kernel", shifts, persistent=False)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list]:
        features = []
        for block in self.encoder:
            image = block(image)
            features.append(image)

        previous = self.coarse_head(features[-1])
        segmentations = [previous]
        edges = []
        for up, seg_head, edge_head, skip in zip(
            self.ups, self.seg_heads, self.edge_heads, reversed(features[:-1])
        ):
            image = up(image, skip)
            edges.append(edge_head(image))
            guidance = F.interpolate(
                previous,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            previous = seg_head(torch.cat((image, guidance), dim=1))
            segmentations.append(previous)

        direction = self.direction_head(image)
        b, c, h, w = previous.shape
        shifted = F.conv2d(
            previous.view(b * c, 1, h, w),
            self.shift_kernel.to(previous.dtype),
            padding=1,
        ).view(b, c, 8, h, w)

        direction_weights = direction.softmax(dim=1).unsqueeze(1)
        corrected = (shifted * direction_weights).sum(dim=2)

        boundary_gate = (edges[-1].sigmoid().detach() > 0.5).to(previous.dtype)
        refined = boundary_gate * corrected + (1.0 - boundary_gate) * previous

        return dict(
            final=refined,
            segments=segmentations,
            edges=edges,
            direction=direction,
        )


def _compute_multi_class_boundary_and_direction(
    labels: torch.Tensor, valid: torch.Tensor, radius: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute semantic boundary and inward directions from multi-class ground truth."""
    labels_np = labels.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    if labels_np.ndim == 4:
        labels_np = labels_np[:, 0]
    if valid_np.ndim == 4:
        valid_np = valid_np[:, 0]

    boundaries = []
    directions = []
    offsets = np.asarray(CBRNetMultiClass.DIRECTIONS, dtype=np.float32)
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)

    for mask, good in zip(labels_np, valid_np):
        if not np.any(good):
            boundaries.append(np.zeros_like(mask, dtype=np.float32))
            directions.append(np.full_like(mask, -1, dtype=np.int64))
            continue

        # Fast 4-neighbor boundary detection
        diff_y = mask[:-1, :] != mask[1:, :]
        good_y = good[:-1, :] & good[1:, :]
        diff_x = mask[:, :-1] != mask[:, 1:]
        good_x = good[:, :-1] & good[:, 1:]

        b_map = np.zeros_like(mask, dtype=bool)
        b_map[:-1, :] |= diff_y & good_y
        b_map[1:, :] |= diff_y & good_y
        b_map[:, :-1] |= diff_x & good_x
        b_map[:, 1:] |= diff_x & good_x
        b_map &= good

        if not np.any(b_map) or not np.any((~b_map) & good):
            boundaries.append(b_map.astype(np.float32))
            directions.append(np.full_like(mask, -1, dtype=np.int64))
            continue

        # Distance to boundary: 0 at boundary, increases into interior
        dist = distance_transform_edt(~b_map)
        edge = (dist < radius) & good

        # Gradient of distance points AWAY from boundary into object interior
        dy, dx = np.gradient(dist.astype(np.float32))
        angle_scores = (
            dy[..., None] * offsets[:, 0] + dx[..., None] * offsets[:, 1]
        )
        classes = angle_scores.argmax(axis=-1).astype(np.int64)
        classes[~edge] = -1

        boundaries.append(edge.astype(np.float32))
        directions.append(classes)

    boundary = torch.from_numpy(np.stack(boundaries))[:, None].to(labels.device)
    direction = torch.from_numpy(np.stack(directions)).to(labels.device)
    return boundary, direction


@MODELS.register_module()
class CBRNetRemoteSensing(BaseSegmentor):
    """MMSeg adapter for multi-class CBRNet on remote sensing benchmarks."""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        edge_radius: int = 5,
        ignore_index: int = 255,
        data_preprocessor: dict | nn.Module | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes = num_classes
        self.edge_radius = edge_radius
        self.ignore_index = ignore_index
        self.network = CBRNetMultiClass(
            num_classes=num_classes, pretrained=pretrained
        )
        self.test_cfg = test_cfg or dict(mode="whole")
        self.align_corners = False

    def extract_feat(self, inputs: torch.Tensor) -> dict:
        return self.network(inputs)

    def _forward(
        self, inputs: torch.Tensor, data_samples: Sequence | None = None
    ) -> torch.Tensor:
        return self.encode_decode(inputs)

    def encode_decode(
        self, inputs: torch.Tensor, batch_img_metas: Sequence | None = None
    ) -> torch.Tensor:
        del batch_img_metas
        logits = self.network(inputs)["final"]
        if logits.shape[-2:] != inputs.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return logits

    def predict(
        self, inputs: torch.Tensor, data_samples: Sequence | None = None
    ) -> list:
        if self.test_cfg.get("mode", "whole") == "whole":
            logits = self.encode_decode(inputs)
        elif self.test_cfg["mode"] == "slide":
            crop_h, crop_w = self.test_cfg["crop_size"]
            stride_h, stride_w = self.test_cfg["stride"]
            batch, _, height, width = inputs.shape
            h_count = max(height - crop_h + stride_h - 1, 0) // stride_h + 1
            w_count = max(width - crop_w + stride_w - 1, 0) // stride_w + 1
            logits = inputs.new_zeros((batch, self.num_classes, height, width))
            coverage = inputs.new_zeros((batch, 1, height, width))
            for row in range(h_count):
                for col in range(w_count):
                    y2 = min(row * stride_h + crop_h, height)
                    x2 = min(col * stride_w + crop_w, width)
                    y1 = max(y2 - crop_h, 0)
                    x1 = max(x2 - crop_w, 0)
                    crop = inputs[:, :, y1:y2, x1:x2]
                    logits[:, :, y1:y2, x1:x2] += self.encode_decode(crop)
                    coverage[:, :, y1:y2, x1:x2] += 1
            logits /= coverage.clamp_min(1.0)
        else:
            raise ValueError(
                f'unsupported inference mode: {self.test_cfg["mode"]}'
            )
        return self.postprocess_result(logits, data_samples)

    def loss(self, inputs: torch.Tensor, data_samples: Sequence) -> dict:
        outputs = self.network(inputs)
        target = torch.stack(
            [sample.gt_sem_seg.data for sample in data_samples], dim=0
        ).to(inputs.device)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        target = target.long()

        valid = target != self.ignore_index
        if not valid.any():
            return dict(
                loss_refined=outputs["final"].sum() * 0.0,
                loss_seg=outputs["segments"][-1].sum() * 0.0,
                loss_pseudo=outputs["segments"][-1].sum() * 0.0,
                loss_edge=outputs["edges"][-1].sum() * 0.0,
                loss_direction=outputs["direction"].sum() * 0.0,
            )

        edge, direction = _compute_multi_class_boundary_and_direction(
            target, valid, radius=self.edge_radius
        )

        segments = outputs["segments"]
        edges = outputs["edges"]
        final_pred = outputs["final"]
        last_seg = segments[-1]

        # Upsample if needed
        if final_pred.shape[-2:] != target.shape[-2:]:
            final_pred = F.interpolate(
                final_pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        if last_seg.shape[-2:] != target.shape[-2:]:
            last_seg = F.interpolate(
                last_seg,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # 1. Main cross-entropy losses
        loss_refined = F.cross_entropy(
            final_pred, target, ignore_index=self.ignore_index
        )
        loss_seg = F.cross_entropy(
            last_seg, target, ignore_index=self.ignore_index
        )

        # 2. Pseudo-label consistency loss (refined guides segmentor)
        p_target = final_pred.detach().softmax(dim=1)
        val_mask = valid.unsqueeze(1).float()
        loss_pseudo = (
            -(p_target * last_seg.log_softmax(dim=1)) * val_mask
        ).sum() / val_mask.sum().clamp_min(1.0)

        # 3. Edge loss (BCE)
        last_edge = edges[-1]
        if last_edge.shape[-2:] != edge.shape[-2:]:
            last_edge = F.interpolate(
                last_edge,
                size=edge.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        val_edge = valid.unsqueeze(1).float()
        loss_edge = (
            F.binary_cross_entropy_with_logits(
                last_edge, edge, reduction="none"
            )
            * val_edge
        ).sum() / val_edge.sum().clamp_min(1.0)

        losses = dict(
            loss_refined=loss_refined,
            loss_seg=loss_seg,
            loss_pseudo=loss_pseudo,
            loss_edge=loss_edge,
        )

        # Auxiliary segmentation losses
        for idx, pred in enumerate(segments[:-1]):
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(
                    pred,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            losses[f"loss_seg_aux{idx + 1}"] = 0.25 * F.cross_entropy(
                pred, target, ignore_index=self.ignore_index
            )

        # Auxiliary edge losses
        for idx, pred in enumerate(edges[:-1]):
            if pred.shape[-2:] != edge.shape[-2:]:
                pred = F.interpolate(
                    pred,
                    size=edge.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            losses[f"loss_edge_aux{idx + 1}"] = 0.25 * (
                (
                    F.binary_cross_entropy_with_logits(
                        pred, edge, reduction="none"
                    )
                    * val_edge
                ).sum()
                / val_edge.sum().clamp_min(1.0)
            )

        # 4. Direction loss
        direction_pred = outputs["direction"]
        if direction_pred.shape[-2:] != direction.shape[-2:]:
            direction_pred = F.interpolate(
                direction_pred,
                size=direction.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        dir_valid = direction != -1
        if dir_valid.any():
            dir_loss = F.cross_entropy(
                direction_pred.float(),
                direction,
                ignore_index=-1,
                reduction="none",
            )
            losses["loss_direction"] = (dir_loss * dir_valid).sum() / dir_valid.sum()
        else:
            losses["loss_direction"] = direction_pred.sum() * 0.0

        return losses
