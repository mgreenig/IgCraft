"""
BFN accuracy schedules.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from torch import nn


class BFNAccuracySchedule(ABC):
    """An interface for BFN accuracy schedules."""

    @abstractmethod
    def compute_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Computes beta (cumulative accuracy) as a function of time."""
        pass

    @abstractmethod
    def compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Computes alpha (accuracy) as a function of time."""
        pass


class DiscreteBFNAccuracySchedule(BFNAccuracySchedule):
    """
    The discrete BFN accuracy schedule from the original BFN paper, i.e. :math:`\\beta(t) = t^2 \\beta(1)`.
    """

    def __init__(self, variables_shape: Sequence[int], beta_1: float):
        """
        :param variables_shape: The shape of the variables that the schedule will be applied to.
        :param beta_1: A hyperparameter specifying the value of beta at :code:`t=1` (end
            of the generative process). This entirely specifies the accuracy schedule.
        """
        super().__init__()

        self._variables_shape = variables_shape
        self._beta_1 = beta_1

    def compute_beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of beta (the cumulative accuracy) at time :code:`t`.

        :param t: The time on the interval [0, 1].
        :return: The value of beta at time :code:`t`.
        """
        beta = t**2 * self._beta_1

        for _ in range(len(self._variables_shape)):
            beta = beta[..., None]

        return beta.expand(*((-1,) * len(t.shape)), *self._variables_shape)

    def compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of alpha (the accuracy, derivative of beta(t)) at time :code:`t`.

        :param t: The time on the interval [0, 1].
        :return: The value of alpha at time :code:`t`.
        """
        alpha = 2 * t * self._beta_1

        for _ in range(len(self._variables_shape)):
            alpha = alpha[..., None]

        return alpha.expand(-1, *self._variables_shape)
