"""Overlap metrics with class-1 road IoU for validation checkpoint selection."""
from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class RoadIoUMetric(IoUMetric):
    default_prefix = None

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        # IoUMetric collects (intersection, union, prediction area, target area).
        intersection = float(sum(result[0] for result in results)[1])
        union = float(sum(result[1] for result in results)[1])
        prediction = float(sum(result[2] for result in results)[1])
        target = float(sum(result[3] for result in results)[1])
        metrics['roadIoU'] = round(100 * intersection / union, 2) if union else 0.0
        denominator = prediction + target
        metrics['roadF1'] = round(200 * intersection / denominator, 2) if denominator else 0.0
        return metrics
