"""Abstract encoder interfaces for modular multimodal architecture."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Base interface for latent feature extractors."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the latent representation."""

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent representation, not task predictions."""


class TimeSeriesEncoder(BaseEncoder):
    """Interface for long-term time-series encoders (e.g. iTransformer)."""


class ContextEncoder(BaseEncoder):
    """Interface for NLP / context encoders (e.g. FinBERT)."""
