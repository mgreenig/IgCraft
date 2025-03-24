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


class LearnableAccuracySchedule(BFNAccuracySchedule, nn.Module):
    """
    A learnable BFN accuracy schedule that maintains a schedule of learnable positive accuracy
    values that are always enforced to integrate to a constant :code:`beta_1`.
    """

    def __init__(self, variables_shape: Sequence[int], num_t: int, beta_1: float):
        """
        :param variables_shape: The shape of the variables that the schedule will be applied to.
        :param num_t: The number of time steps to discretize the generative process into.
        :param beta_1: The value of beta at the end of the generative process (t=1).
        """
        super().__init__()

        # start with uniform accuracy at each time step
        self._alpha_logits = nn.Parameter(
            torch.zeros(tuple(variables_shape) + (num_t,))
        )
        self._alpha_logits.register_hook(self._rescale_alpha)

        self._num_t = num_t
        self._beta_1 = beta_1

    @property
    def num_t(self) -> int:
        """
        The number of time steps to discretize the generative process into.
        """
        return self._alpha_logits.shape[-1]

    def _rescale_alpha(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Rescales alpha by subtracting its maximum value. This does not affect the gradient (since
        the softmax is invariant to this operation). It only serves to stabilise the parameters.
        """
        with torch.no_grad():
            self._alpha_logits.sub_(self._alpha_logits.data.max(-1).values[..., None])

        return grad

    def t_select(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        For a value :code:`t` on the interval [0, 1], select the corresponding discretized value from the
        input tensor :code:`values`, treating :code:`t` as an index for the last dimension of :code:`values`.

        :param values: The tensor of values to select from, of shape :code:`(batch_size, *variables_shape, num_t)`.
        :param t: The time on the interval [0, 1] as a tensor of shape :code:`(batch_size,)` which is used to
            index the :code:`values` tensor along the final dimension.
        :return: The selected values from the tensor.
        """
        batch_idx = torch.arange(values.shape[0], device=values.device)
        t_idx = torch.round(t * (self._num_t - 1)).long()
        return values[batch_idx, ..., t_idx]

    def compute_beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of beta (the cumulative accuracy) at time :code:`t`.

        :param t: Rank-1 tensor of times on the interval [0, 1], one for each sample in the batch.
        :return: The value of beta at time :code:`t`.
        """
        alphas = torch.nn.functional.softmax(self._alpha_logits, dim=-1) * self._beta_1

        # integrate the alphas and expand to the batch size
        betas = torch.cumsum(alphas, dim=-1)
        betas = betas[None].expand((t.shape[0],) + (-1,) * len(betas.shape))

        beta_t = self.t_select(betas, t)
        return beta_t

    def compute_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of alpha, the derivative of beta(t) at time :code:`t`.

        :param t: Rank-1 tensor of times on the interval [0, 1], one for each sample in the batch.
        :return: The value of alpha at time :code:`t`.
        """
        alphas = torch.nn.functional.softmax(self._alpha_logits, dim=-1) * self._beta_1

        # expand to the batch size
        alphas = alphas[None].expand((t.shape[0],) + (-1,) * len(alphas.shape))

        alpha_t = self.t_select(alphas, t)
        return alpha_t
