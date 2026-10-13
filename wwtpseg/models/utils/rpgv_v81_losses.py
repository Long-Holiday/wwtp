"""Label-aware boundary and soft dense supervision for RPGV v8.1."""
import torch
from torch.nn import functional as F

from ..segmentors.rpgv_net_v2 import masked_mean, region_and_boundary_loss
from .rpgv_v8_losses import boundary_band_loss, class_proportions


def final_pair_losses(logits, target, valid, binary):
    if binary:
        return region_and_boundary_loss(logits, target, valid)
    probability = logits.float().softmax(dim=1)
    one_hot = F.one_hot(target[:, 0].long(), logits.shape[1]).permute(0, 3, 1, 2).float()
    region, boundary = probability.sum() * 0, probability.sum() * 0
    for dimension in (-2, -1):
        left, right = [slice(None)] * 4, [slice(None)] * 4
        left[dimension], right[dimension] = slice(None, -1), slice(1, None)
        a, b = tuple(left), tuple(right)
        pairs = valid[a] & valid[b]
        same = pairs & (target[a] == target[b])
        different = pairs & (target[a] != target[b])
        delta = probability[b] - probability[a]
        truth = one_hot[b] - one_hot[a]
        region = region + masked_mean(delta.abs().mean(1, keepdim=True), same)
        boundary = boundary + masked_mean((delta - truth).abs().mean(1, keepdim=True), different)
    return region / 2, boundary / 2


def balanced_boundary_band_loss(logits, target, valid, binary, radius):
    if not binary:
        return boundary_band_loss(logits, target, valid, False, radius)
    label = target.float() >= 0.5
    known = valid.bool()
    edge = torch.zeros_like(known)
    horizontal = (label[..., :, 1:] != label[..., :, :-1]) & (known[..., :, 1:] & known[..., :, :-1])
    vertical = (label[..., 1:, :] != label[..., :-1, :]) & (known[..., 1:, :] & known[..., :-1, :])
    edge[..., :, 1:] |= horizontal
    edge[..., :, :-1] |= horizontal
    edge[..., 1:, :] |= vertical
    edge[..., :-1, :] |= vertical
    kernel = 2 * radius + 1
    band = F.max_pool2d(edge.float(), kernel, 1, radius) > 0
    near_ignore = F.max_pool2d((~known).float(), kernel, 1, radius) > 0
    band = band & known & ~near_ignore
    positive, negative = band & label, band & ~label
    error = F.binary_cross_entropy_with_logits(logits.float(), label.float(), reduction='none')
    has_both = positive.any() & negative.any()
    return 0.5 * (masked_mean(error, positive) + masked_mean(error, negative)) * has_both.float()


def gaussian_dense_dice_loss(logits, target, valid):
    distribution, known = class_proportions(target, valid, logits.shape[-2:], 2)
    foreground = distribution[:, 1:2]
    probability = logits.float().sigmoid()
    dimensions = (1, 2, 3)
    intersection = (probability * foreground * known).sum(dimensions)
    cardinality = ((probability + foreground) * known).sum(dimensions)
    error = 1 - (2 * intersection + 1) / (cardinality + 1)
    present = (known.sum(dimensions) > 0).float()
    return (error * present).sum() / present.sum().clamp_min(1)
