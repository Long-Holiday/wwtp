"""Label-only supervision for end-to-end Gaussian geometry."""
import torch
from torch.nn import functional as F

from ..segmentors.rpgv_net import _binary_segmentation_loss
from ..segmentors.rpgv_net_v2 import masked_mean
from wwtpseg.rpgv_v6_targets import contour_target_v6


def semantic_loss(logits, target, valid, binary):
    if binary:
        return _binary_segmentation_loss(logits, target, valid)
    error = F.cross_entropy(logits.float(), target[:, 0].long(), reduction='none')
    return masked_mean(error, valid[:, 0])


def class_proportions(target, valid, size, classes):
    """Pool valid class counts, never the integer labels or ignore sentinel."""
    with torch.no_grad():
        one_hot = F.one_hot(target[:, 0].long(), classes).permute(0, 3, 1, 2).float()
        counts = F.adaptive_avg_pool2d(one_hot * valid, size)
        known = F.adaptive_avg_pool2d(valid.float(), size)
        return counts / known.clamp_min(1e-8), known


def soft_semantic_loss(logits, distribution, weight, binary):
    logits = logits.float()
    if binary:
        error = F.binary_cross_entropy_with_logits(logits, distribution[:, 1:2], reduction='none')
    else:
        error = -(distribution * logits.log_softmax(dim=1)).sum(dim=1, keepdim=True)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)


def gaussian_semantic_losses(outputs, target, valid, binary, dense_weight, node_weight):
    classes = 2 if binary else outputs['final_logits'].shape[1]
    raster = outputs['gaussian']['raster']
    distribution, known = class_proportions(target, valid, raster['size'], classes)
    losses = {}
    if dense_weight:
        losses['loss_gaussian_dense'] = dense_weight * soft_semantic_loss(
            outputs['gaussian_dense_logits'], distribution, known, binary)
    if node_weight:
        with torch.no_grad():
            support = raster['support'].detach()
            weights = support * known.flatten(2)
            mass = weights.sum(dim=-1, keepdim=True)
            node_target = torch.bmm(weights, distribution.flatten(2).transpose(1, 2))
            node_target = (node_target / mass.clamp_min(1e-8)).transpose(1, 2)
            # All supervised supports participate, without a hard mass threshold.
            node_weighting = mass.transpose(1, 2)
        losses['loss_gaussian_node'] = node_weight * soft_semantic_loss(
            outputs['gaussian_node_logits'].transpose(1, 2), node_target, node_weighting, binary)
    return losses


def boundary_band_loss(logits, target, valid, binary, radius):
    edge = torch.zeros_like(valid)
    horizontal = (target[..., :, 1:] != target[..., :, :-1]) & (valid[..., :, 1:] & valid[..., :, :-1])
    vertical = (target[..., 1:, :] != target[..., :-1, :]) & (valid[..., 1:, :] & valid[..., :-1, :])
    edge[..., :, 1:] |= horizontal
    edge[..., :, :-1] |= horizontal
    edge[..., 1:, :] |= vertical
    edge[..., :-1, :] |= vertical
    kernel = 2 * radius + 1
    band = F.max_pool2d(edge.float(), kernel, 1, radius) > 0
    near_ignore = F.max_pool2d((~valid).float(), kernel, 1, radius) > 0
    band = band & valid & ~near_ignore
    if binary:
        error = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
        return masked_mean(error, band)
    error = F.cross_entropy(logits.float(), target[:, 0].long(), reduction='none')
    terms, present = [], []
    for index in range(logits.shape[1]):
        selected = band[:, 0] & (target[:, 0] == index)
        terms.append(masked_mean(error, selected))
        present.append(selected.any())
    return torch.stack(terms).sum() / torch.stack(present).sum().clamp_min(1)


def sdf_loss(logits, target, valid, binary, truncation):
    terms = []
    for index in range(logits.shape[1]):
        mask = target.float() if binary else (target == index).float()
        sdf, known = contour_target_v6(mask * valid, valid, logits.shape[-2:], truncation)
        error = F.smooth_l1_loss(logits[:, index:index + 1].float().tanh(), sdf, reduction='none')
        terms.append(masked_mean(error * (1 + 4 * (1 - sdf.abs())), known))
    return torch.stack(terms).mean()
