"""Reversible Instance Normalization (RevIN) for non-stationary time series."""

import torch
from torch import nn


class RevIN(nn.Module):
    """Instance normalization with optional affine and denormalization."""

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, series: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            return self._normalize(series)
        if mode == "denorm":
            return self._denormalize(series)
        msg = f"mode must be 'norm' or 'denorm', got {mode}"
        raise ValueError(msg)

    def _normalize(self, series: torch.Tensor) -> torch.Tensor:
        # series: (batch, seq_len, n_features)
        mean = series.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(series.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        self._mean = mean
        self._std = std
        normalized = (series - mean) / std
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized

    def _denormalize(self, series: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            msg = "Call norm before denorm."
            raise RuntimeError(msg)
        output = series
        if self.affine:
            output = (output - self.affine_bias) / (self.affine_weight + self.eps)
        return output * self._std + self._mean
