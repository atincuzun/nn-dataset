import math
from typing import Optional
import torch
import torch.nn.functional as F
from .base.base import BaseMetric


class Perplexity(BaseMetric):
    """
    Per-token perplexity for [B,V] or [B,S,V] outputs
    """
    def __init__(self, ignore_index: Optional[int] = None):
        super().__init__()
        self._total_tokens = None
        self._total_loss = None
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self._total_loss: float = 0.0
        self._total_tokens: int = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        # [B,S,V] -> [B*S,V] and [B,S] -> [B*S]
        if outputs.dim() == 3:
            B, S, V = outputs.shape
            outputs = outputs.reshape(B * S, V)
            targets = targets.reshape(-1)

        targets = targets.long()

        if self.ignore_index is not None:
            loss = F.cross_entropy(outputs, targets, reduction="sum", ignore_index=self.ignore_index,)
            valid = (targets != self.ignore_index).sum().item()
        else:
            loss = F.cross_entropy(outputs, targets, reduction="sum")
            valid = targets.numel()

        self._total_loss += loss.item()
        self._total_tokens += valid

    def result(self) -> float:
        if self._total_tokens == 0:
            return float("nan")
        return math.exp(self._total_loss / self._total_tokens)

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        self.update(outputs, targets)
        return self.result(), self._total_tokens


def compute(outputs: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None):
    metric = Perplexity(ignore_index)
    metric.update(outputs, targets)
    return metric.result(), 1


def create_metric(out_shape=None, **kwargs):
    return Perplexity(**kwargs)
