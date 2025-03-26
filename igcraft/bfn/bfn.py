"""
Implements a discrete-variable BFN.
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .schedule import BFNAccuracySchedule


class DiscreteBFN(nn.Module):
    """Discrete-variable Bayesian Flow Network."""

    def __init__(
        self,
        schedule: BFNAccuracySchedule,
        variables_shape: Sequence[int],
        num_classes: int,
    ):
        """
        :param schedule: The accuracy schedule for the BFN.
        :param variables_shape: The shape of the discrete variables, excluding the final (category)
            dimension.
        :param num_classes: The number of categories for the discrete variables (the final dimension
            of the parameters produced by the BFN).
        """
        super().__init__()
        self.schedule = schedule
        self.variables_shape = tuple(variables_shape)
        self.num_classes = num_classes

    def get_prior_input_distribution(
        self, t_start: float, batch_size: int | None = None
    ) -> torch.Tensor:
        """Initialises the parameters of an uninformed input distribution.
        If :code:`t_start > 0`, this samples from :math:`N(0, K \\beta(t) I)`.

        :param t_start: The time at which the generative process starts. For
            :code:`t_start > 0`, the input distribution is sampled from a normal distribution.
        :param batch_size: The batch size of the sampled parameters. If not :code:`None`, the output
            shape will be :code`(batch_size, *variables_shape, num_classes)`.
        :return logits: Uniform prior distribution parameters.
        """
        logits = torch.zeros((batch_size,) + self.variables_shape + (self.num_classes,))
        if t_start > 0:
            K = self.num_classes
            t_start = torch.as_tensor(t_start, device=logits.device, dtype=logits.dtype)
            beta = self.schedule.compute_beta(t_start)
            logits += torch.sqrt(K * beta)[..., None] * torch.randn_like(logits)

        return logits

    def sample_sender(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a noise sample for the ground-truth (x) from the sender distribution using
        eq 157 in the BFN paper.

        :param x: A tensor of ground-truth categorical data (integers) with shape specified by
            the attribute :code:`variables_shape`.
        :param alpha: The accuracy parameter for the sender distribution.
        :return: The sample from the sender distribution.
        """
        K = self.num_classes

        mu = alpha[..., None] * (K * F.one_hot(x, num_classes=K) - 1)
        sigma = torch.sqrt(alpha[..., None] * K * torch.ones(K, device=alpha.device))

        # Sample with re-parameterisation trick to allow differentiation w.r.t alpha
        z = torch.randn_like(mu)
        y = mu + sigma * z

        return y

    def sample_flow(
        self,
        x: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the Bayesian flow distribution using the ground-truth discrete data :code:`x`
        and the accuracy schedule :code:`beta`.

        This is sampled as:

        .. math::
            y \\sim Normal(y|\\beta(t)(K e_x − 1),\\beta(t)KI) \\newline
            \\theta = softmax(y)

        :param x: The ground-truth discrete data with shape specified by the attribute :code:`variables_shape`.
        :param beta: The beta value at time t.
        :return: The sample from the Bayesian flow distribution.
        """
        y = self.sample_sender(x, beta)
        return y

    def sample_from_logits(self, pred: torch.Tensor) -> torch.Tensor:
        """Sample from a tensor of input logits.

        :param pred: Input logits of shape :code:`(..., num_classes)`.
        :return: A sample drawn from the predicted distribution.
        """
        probs = F.softmax(pred, dim=-1)
        shape = probs.shape
        if len(shape) > 2:
            probs = probs.reshape((-1, self.num_classes))

        samples = torch.multinomial(probs, 1).squeeze(-1)
        return samples.reshape(shape[:-1])

    def logL(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculate the log probability of the data :code:`x` given the current logits.

        :param logits: The input logits of shape :code:`(..., *variables_shape, num_classes)`.
        :param x: Discrete data of shape :code:`(..., *variables_shape)`.
        :param mask: The mask specifying which variables in x are used for calculating the logL.
            All variables are included if this is :code:`None`. Should be the same shape as
            :code:`x` if provided.
        :return: The log probabilities summed over the variables.
        """
        if x.shape != logits.shape[:-1]:
            raise ValueError(
                f"Input data shape {x.shape} does not match the logits shape {logits.shape[:-1]}"
            )

        dims = tuple(range(len(x.shape)))
        reduce_dims = dims[-len(self.variables_shape) :]

        idx = F.one_hot(x, num_classes=logits.shape[-1])

        # log probability per variable (summed over classes)
        log_probs = F.log_softmax(logits, dim=-1)
        variable_log_probs = torch.sum(log_probs * idx, dim=-1)

        if mask is not None:
            mask = mask.to(variable_log_probs.dtype)
            data_log_prob = torch.sum(mask * variable_log_probs, dim=reduce_dims)
        else:
            data_log_prob = torch.sum(variable_log_probs, dim=reduce_dims)

        return data_log_prob


class DiscreteBFNSDESolver:
    """
    Implements the BFN SDE solver 2 from https://arxiv.org/pdf/2404.15766, which uses
    information from previous predictions to reduce discretization error in the SDE.
    """

    def __init__(self):
        """
        Initialises the BFN SDE solver.
        """
        self.previous_probs = None

    def __call__(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        beta: torch.Tensor,
        beta_next: torch.Tensor,
        t_next: torch.Tensor,
        conditional_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Takes a step in logit space to update the current distribution parameters.

        :param logits: Current logits of the categorical distribution of shape :code:`(*leading_shape, num_classes)`.
        :param probs: Predicted probabilities from the output network.
        :param beta: The beta value for the current time step. Should be the shape specified by
            the attribute :code:`variables_shape`.
        :param beta_next: The beta value for the next time step. Should be the shape specified by
            the attribute :code:`variables_shape`.
        :param t_next: The next time step.
        :return: The updated logits.
        """
        alpha = beta_next - beta
        K = probs.shape[-1]
        drift = alpha[..., None] * (K * probs - 1)

        if self.previous_probs is not None:
            dp = probs - self.previous_probs
            drift += K * dp * 0.5 * alpha[..., None]

        diffusion = (
            torch.sqrt(alpha[..., None] * K)
            * torch.randn_like(drift)
        )
        updated_logits = logits + drift + diffusion

        self.previous_probs = probs.clone()

        # If at t=1, reset the previous_probs
        if torch.all(t_next == 1.0).item():
            self.previous_probs = None

        return updated_logits
