"""Generative models for data sequences."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import replace
from time import time
from typing import Any, Callable, Generic, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from h5py.h5t import NORM_NONE
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR
from tqdm import tqdm

from ..bfn.bfn import DiscreteBFN, DiscreteBFNSDESolver
from ..data.constants import AA1_INDEX
from ..data.structure import PairedStructureData, StructureData, UnpairedStructureData
from .config import PairedSequenceModelConfig, SequenceModelConfig
from .network import PairedSequenceTransformer, SequenceTransformer


class EMAUpdate(torch.nn.Module):
    """A module that updates the exponential moving average of a model's parameters."""

    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay

    @torch.no_grad()
    def forward(
        self, ema_param: torch.Tensor, current_param: torch.Tensor, num_averaged: int
    ) -> torch.Tensor:
        """
        Updates the exponential moving average of a model's parameters.

        :param ema_param: The exponential moving average of the model's parameters.
        :param current_param: The current parameters of the model.
        :param num_averaged: Not used, maintained for compatibility with :code:`avg_fn`
            in :class:`torch.optim.swa_utils.AveragedModel`.
        :return: The updated EMA parameters.
        """
        return self.decay * ema_param + (1 - self.decay) * current_param


# Fundamental type for the generative model's data
DataType = TypeVar("DataType")

# Type of the underlying network architecture
NetworkType = TypeVar("NetworkType", bound=nn.Module)


class SequenceGenerativeModel(ABC, LightningModule, Generic[DataType, NetworkType]):
    """
    A generative model for unpaired data sequences, whose forward pass takes in
    inputs for a single chain (either VH or VL) and a time parameter and returns
    the sequence predictions for the chain.
    """

    def __init__(
        self,
        cfg: SequenceModelConfig,
        network: NetworkType,
    ):
        """
        :param cfg: Configuration object for the model. See :class:`PairedSequenceModelConfig`
            for the required parameters.
        :param network: The prediction network containing all of the model's learnable weights.
        """
        super().__init__()

        self._max_lr = cfg.max_lr
        self._min_lr = cfg.min_lr
        self._use_lr_schedule = cfg.use_lr_schedule
        self._warmup_steps = cfg.warmup_steps
        self._transition_steps = cfg.transition_steps
        self._num_eval_time_bins = cfg.num_eval_time_bins
        self._mask_fwr_rate = cfg.mask_fwr_rate

        self.network = network

        ema_update = EMAUpdate(cfg.ema_decay)
        self.ema_network = torch.optim.swa_utils.AveragedModel(
            network, avg_fn=ema_update
        )

        # Placeholder for tracking step/load times
        self._time = None

        # Lists for validations and test outputs
        self._val_outputs = []
        self._test_outputs = []

        # Objects that will be used to store reference sets for evaluation
        self._train_reference_set = []
        self._test_reference_set = []

        # Time bins for evaluation
        self.register_buffer(
            "t_bins", torch.linspace(0, 1, self._num_eval_time_bins + 1)
        )

    @abstractmethod
    def forward(
        self,
        batch: tuple[DataType, PairedStructureData | UnpairedStructureData | None],
        t: torch.Tensor,
    ) -> DataType:
        """
        Forward pass of the model.

        :param batch: The inputs for the forward pass, which consists of the model's data type
            and optional structural conditioning data.
        :param t: The current time in the generative process.
        """
        pass

    def compile_network(self):
        """
        Compile the underlying network.
        """
        self.network.compile()
        self.ema_network.compile()

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        """Performs the optimizer step, and if using EMA, updates the EMA model."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema_network.update_parameters(self.network)

    def on_train_epoch_start(self) -> None:
        """Initialises the time for measuring batch time."""
        self._time = time()

    def on_train_batch_start(
        self,
        batch: tuple[DataType, PairedStructureData | UnpairedStructureData | None],
        batch_idx: int,
    ):
        """Logs the time the model waited for the batch."""
        # the time it took to load the data
        load_time = time() - self._time
        self.log("load_time", load_time, sync_dist=True)

        # reset the time to measure step time
        self._time = time()

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(
        self,
        outputs: dict[str, Any],
        batch: tuple[DataType, PairedStructureData | UnpairedStructureData | None],
        batch_idx: int,
    ) -> None:
        """Logs the time taken to perform a training step."""
        # time it took to take a training step
        step_time = time() - self._time
        self.log("step_time", step_time, sync_dist=True)

        # reset the time to measure wait time
        self._time = time()

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def configure_optimizers(self):
        """
        Configures an AdamW optimise with the maximum learning rate set in the configuration,
        and a linear warmup/decay learning rate scheduler if specified.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self._max_lr)

        if self._use_lr_schedule:
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-8, total_iters=self._warmup_steps
            )
            transition_scheduler = LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self._min_lr / self._max_lr,
                total_iters=self._transition_steps,
            )
            min_lr = LambdaLR(optimizer, lambda step: self._min_lr / self._max_lr)
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, transition_scheduler, min_lr],
                [self._warmup_steps, self._warmup_steps + self._transition_steps],
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            }
            return [optimizer], [lr_scheduler]

        return optimizer


# Generic type for the type of model used by a sampler
SequenceGenerativeModelType = TypeVar(
    "SequenceGenerativeModelType",
    bound=SequenceGenerativeModel,
)


class SequenceSampler(ABC, Generic[SequenceGenerativeModelType]):
    """Interface for a sampler that generates data sequences."""

    @staticmethod
    def _get_particle_logits(
        probs: torch.Tensor,
        cond_seq_one_hot: torch.Tensor,
        cond_mask: torch.Tensor | None,
        downweight_pads: bool = False,
    ) -> torch.Tensor:
        """
        Calculates sampling logits for particles in SMC-based conditioning using
        the squared error between the predictions and the conditioning data.

        :param probs: Predicted probabilities from the output network of
            shape :code:`(batch_size, num_particles, seq_len, num_classes)`.
        :param cond_seq_one_hot: The conditioning sequences, represented as a one hot tensor. Should be shape
            :code:`(batch_size, 1, seq_len, num_classes)`.
        :param cond_mask: Optional mask for the sequences in :code:`cond_x`, specifying which of the elements
            to condition on (with :code:`mask=True`). If not provided, all elements of :code:`cond_x` are
            conditioned on. Should be shape :code:`(batch_size, 1, seq_len)`.
        :param downweight_pads: Whether to downweight particle probabilities based on the likelihood of
            pad tokens in positions where :code:`cond_mask=False`. This can be useful to ensure the
            model conditionally samples sequences of a fixed length.
        :return: Tensor of shape :code:`(batch_size, num_particles)` containing the sampling probabilities
            for each particle.
        """
        # Softmax logits to get probabilities per-class
        probs = F.softmax(probs, dim=-1)

        # Calculate the squared error between the predictions and the conditioning data
        d_kl = (probs - cond_seq_one_hot) ** 2

        # Zero out elements for elements which are not being used for conditioning
        if cond_mask is not None:
            d_kl = d_kl * cond_mask[..., None].to(d_kl.dtype)

        particle_logits = -torch.sum(d_kl, dim=(-2, -1))

        if downweight_pads and cond_mask is not None:

            # logL of pad tokens for non-conditioned positions
            pad_logL = torch.sum(
                torch.log(probs[..., AA1_INDEX["-"]]) * (~cond_mask).to(d_kl.dtype),
                dim=-1,
            )
            particle_logits -= pad_logL

        return particle_logits

    @abstractmethod
    def __call__(
        self,
        model: SequenceGenerativeModelType[DataType, NetworkType],
        num_samples: int = 1,
        cond_x: (
            tuple[DataType, PairedStructureData | UnpairedStructureData | None] | None
        ) = None,
        cond_mask: DataType = None,
        fix_length: bool = False,
        progress_bar: bool = True,
    ) -> Any:
        """Generate samples with a sequence generative model.

        :param model: The model to use to generate samples.
        :param num_samples: The number of samples to generate. Only used if :code:`cond_x` is :code:`None`.
            Otherwise, if :code:`cond_x` is provided, one sample is generated per batch element.
        :param cond_x: A batch of data to perform conditioning on.
        :param cond_mask: A mask for the conditional data of the same type as the data, specifying which of the
            elements to condition on (with :code:`mask=True`).
        :param fix_length: Whether to fix the length of generated sequences by ensuring the model
            only generates non-padding tokens at positions where :code:`cond_mask=False`.
        :param progress_bar: Whether to display a progress bar of the time in the generative process.
        :return: The generated samples.
        """
        pass


class UnpairedSequenceBFN(SequenceGenerativeModel[torch.Tensor, SequenceTransformer]):
    """
    A BFN model that generates unpaired data sequences (either heavy or light chain individually).
    """

    def __init__(
        self,
        cfg: SequenceModelConfig,
        network: SequenceTransformer,
        bfn: DiscreteBFN,
    ):
        """
        :param cfg: A configuration object containing the model configuration options.
        :param network: The prediction network containing all of the model's learnable weights.
        :param bfn: The Bayesian flow network used to generate samples.
        """
        super().__init__(cfg, network)

        self.bfn = bfn

    def dummy_forward(
        self, batch: tuple[torch.Tensor, UnpairedStructureData | None]
    ) -> None:
        """
        Perform a dummy forward pass to initialise the weights of the model,
        using a batch directly loaded by the dataloader.

        :param batch: A batch of data to perform a forward pass on.
        """
        seq, cond_data = batch
        one_hot = F.one_hot(seq, num_classes=len(AA1_INDEX)).float()
        t = torch.rand(seq.shape[0], device=seq.device)
        self.network(one_hot, t, cond_data)
        self.ema_network(one_hot, t, cond_data)

    def transfer_batch_to_device(
        self,
        batch: tuple[torch.Tensor, dict[str, Any]],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[torch.Tensor, UnpairedStructureData | None]:
        """
        Transfers a batch to the specified device and converts

        :param batch: The batch to transfer.
        :param device: The device to transfer the batch to.
        :param dataloader_idx: The index of the dataloader the batch came from.
        :return: The batch transferred to the specified device.
        """
        seq, cond_data = batch
        seq = seq.to(device)

        if cond_data:
            cond_data = UnpairedStructureData(
                ab=StructureData(**cond_data["ab"]),
                epitope=StructureData(**cond_data["epitope"]),
            )
            cond_data = cond_data.to(device)
        else:
            cond_data = None

        return seq, cond_data

    def forward(
        self,
        batch: tuple[torch.Tensor, UnpairedStructureData | None],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model, taking in input data and returning a tensor
        containing the predicted probabilities for all amino acids.

        :param batch: A tensor containing the current amino acid logits for the unpaired chain, which should be a
            rank-3 tensor of shape :code:`(batch_size, vh_seq_len, num_classes)`, and  a dictionary of
            conditioning data.
        :param t: The current time in the generative process as a rank-1 tensor of shape :code:`(batch_size,)`.
        :return: A tensor containing the predicted amino acid probabilities for the VH and VL chains.
        """
        if self.training:
            network = self.network
        else:
            network = self.ema_network

        logits, cond_data = batch
        probs = F.softmax(logits, dim=-1)
        preds = network(probs, t, cond_data)

        return preds

    @staticmethod
    def continuous_time_loss(
        true: torch.Tensor,
        pred: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the continuous time loss function from the BFN paper (eq. 41).

        :param true: The true unpaired Fv sequences represented as integers (not one-hot encoded).
        :param pred: The predicted amino acid logits, encoded as rank-3 tensors.
        :param alpha: The alpha values for the sequences in the batch.
        :return: The continuous time loss for the sequences as rank-2 tensor of shape :code:`(batch_size, seq_len)`.
        """

        probs = F.softmax(pred, dim=-1)

        one_hot = F.one_hot(true, num_classes=len(AA1_INDEX)).to(dtype=probs.dtype)
        mse = F.mse_loss(probs, one_hot, reduction="none")

        # Sum over classes/tokens but take the mean over sequences in the batch
        loss = torch.sum(alpha[..., None] * mse, dim=-1)

        return loss

    def vlb(
        self,
        batch: tuple[torch.Tensor, UnpairedStructureData | None],
        num_time_steps: int = 10,
        num_flow_samples: int = 10,
    ) -> torch.Tensor:
        """
        Estimates a variational lower bound on the log likelihood of the input data :code:`batch`
        under the model, using the continuous time loss function and the reconstruction loss
        function. Specifically this estimates the integral of eq. 20 from the
        BFN paper (https://arxiv.org/pdf/2308.07037), where :math:`L^n` is replaced with :math:`L^{\infty}`,
        over t in [0, 1].

        :param batch: A batch of unpaired data sequences represented as integers (not one-hot encoded)
            and a dictionary of conditioning data.
        :param num_time_steps: The number of time steps to use to discretize the interval [0, 1].
        :param num_flow_samples: The number of samples to draw from the flow distribution for each time step.
        :return: A tensor containing the estimated VLB for each batch element.
        """
        x, cond_data = batch

        # Compute the continuous time loss per batch element
        ct_loss = torch.zeros(
            (num_time_steps, num_flow_samples, x.shape[0]), device=x.device
        )

        time_steps = torch.linspace(1e-6, 1, num_time_steps, device=x.device)
        for i, t in enumerate(time_steps):

            t = t[None].expand(x.shape[0])

            beta = self.bfn.schedule.compute_beta(t)
            alpha = self.bfn.schedule.compute_alpha(t)

            # Sample from the flow distribution
            for j in range(num_flow_samples):
                theta = self.bfn.sample_flow(x, beta)

                with torch.no_grad():
                    logits = self.forward((theta, cond_data), t)

                mse = self.continuous_time_loss(x, logits, alpha).sum(-1)
                ct_loss[i, j] = mse

        # Compute the expectation over flow samples and integrate over time
        ct_loss = torch.trapezoid(ct_loss.mean(dim=1), time_steps, dim=0)

        # Compute reconstruction loss (t=)
        t_1 = torch.ones(x.shape[0], device=x.device)
        beta_1 = self.bfn.schedule.compute_beta(t_1)

        recon_loss = torch.zeros(x.shape[0], device=x.device)
        for _ in range(num_flow_samples):
            theta_1 = self.bfn.sample_flow(x, beta_1)

            with torch.no_grad():
                logits_1 = self.forward((theta_1, cond_data), t_1)

            recon_loss += -self.bfn.logL(logits_1, x, None)

        recon_loss /= num_flow_samples

        # Scale by the number of tokens
        return -(ct_loss + recon_loss) / x.shape[1]

    def training_step(
        self, batch: tuple[torch.Tensor, UnpairedStructureData | None]
    ) -> torch.Tensor:
        """
        A single training step for the model. This uses the continuous time loss function
        from the BFN paper.

        :param batch: Batch containing chain sequence represented as integers from 0-20 in a
            rank-2 tensor of shape :code:`(batch_size, seq_len)`, and optional structural conditioning data.
        :return: A scalar tensor containing the loss.
        """
        seq, cond_data = batch

        # sample a time for each batch element
        t = torch.rand((seq.shape[0],), device=seq.device)

        beta = self.bfn.schedule.compute_beta(t)
        alpha = self.bfn.schedule.compute_alpha(t)

        theta = self.bfn.sample_flow(seq, beta)
        pred = self.forward((theta, cond_data), t)

        # Sum over classes/tokens but take the mean over sequences in the batch
        loss = self.continuous_time_loss(seq, pred, alpha).mean()
        self.log("train_loss", loss.item(), sync_dist=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, UnpairedStructureData | None]):
        """
        A single validation step for the model. This uses the continuous time loss function
        from the BFN paper.

        :param batch: Batch containing chain sequence represented as integers from 0-20 in a
            rank-2 tensor of shape :code:`(batch_size, seq_len)`, and a dictionary of conditioning data
            (can be empty).
        :return: A scalar tensor containing the loss.
        """
        seq, cond_data = batch

        # sample a time for each batch element
        t = torch.rand((seq.shape[0],), device=seq.device)

        beta = self.bfn.schedule.compute_beta(t)
        alpha = self.bfn.schedule.compute_alpha(t)

        theta = self.bfn.sample_flow(seq, beta)
        pred = self.forward((theta, cond_data), t)

        # sum over classes
        loss = self.continuous_time_loss(seq, pred, alpha)

        # when storing in the metrics dict, sum over tokens as well
        val_loss = loss.sum(-1)
        metrics = {
            "val_loss": val_loss,
            "t": t,
        }

        # sum over the tokens for each individual region to track performance on different IMGT regions
        for region, idx in self.trainer.datamodule.region_indices.items():
            metrics[f"val_{region}_loss"] = torch.sum(loss[..., idx], dim=-1)

        self._val_outputs.append(metrics)

    def on_validation_epoch_end(self) -> None:
        """
        Logs the mean of the validation metrics over the entire validation set,
        as well as the mean of the metrics for each time bin.
        """

        # concatenate the metrics over all validation batches together
        metric_names = set(
            [key for outputs in self._val_outputs for key in outputs.keys()]
        )
        all_metrics = {
            k: torch.cat([metrics[k] for metrics in self._val_outputs if k in metrics])
            for k in metric_names
        }

        t_binned = torch.bucketize(all_metrics["t"], self.t_bins) - 1

        for metric, values in all_metrics.items():
            if metric == "t":
                continue

            self.log(metric, values.mean().item(), sync_dist=True)

            # if the metric is not the same length as the time, skip binning
            if len(values) != len(t_binned):
                continue

            for t in range(self._num_eval_time_bins):
                bin_start = self.t_bins[t].item()
                bin_end = self.t_bins[t + 1].item()
                bin_mask = t_binned == t
                self.log(
                    f"{metric}_{bin_start:.2f}-{bin_end:.2f}",
                    values[bin_mask].mean().item(),
                    sync_dist=True,
                )


class UnpairedSequenceBFNSampler(SequenceSampler[UnpairedSequenceBFN]):
    """
    A sampler that uses a BFN to generate samples of unpaired data sequences.
    Conditioning is performed using the score-based method detailed in https://arxiv.org/pdf/2209.14687.
    """

    def __init__(
        self,
        num_steps: int,
        solver: DiscreteBFNSDESolver,
        t_start: float = 1e-6,
        temperature: float = 1.0,
        num_particles: int = 32,
        replace_receiver: bool = True,
    ):
        """
        :param num_steps: The number of steps to take in the generative process.
        :param solver: The type of solver (update rule) to use in the generative process.
        :param t_start: The starting time in the generative process.
        :param temperature: The temperature to use when sampling from the model.
        :param num_particles: The number of particles to use in SMC conditioning.
        :param replace_receiver: Whether to replace the predicted probabilities in the
            receiver distribution with the ground-truth sequence for the conditioning data.
            This typically guarantees that the model samples sequences that match the conditioning data.
        """
        self.num_steps = num_steps
        self.solver = solver
        self.t_start = t_start
        self.temperature = temperature
        self.num_particles = num_particles
        self.replace_receiver = replace_receiver

    def _step(
        self,
        model: UnpairedSequenceBFN,
        theta: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond_x: tuple[torch.Tensor, UnpairedStructureData | None] | None,
        cond_mask: torch.Tensor | None,
        downweight_pads: bool = False,
    ) -> torch.Tensor:
        """
        Perform a single step in the generative process, returning an updated tensor of parameters.

        :param model: The model to use to generate samples.
        :param theta: The current distribution parameters in the generative process as a
            rank-3 tensor of shape :code:`(batch_size, seq_len, num_classes)`. When using SMC-based
            conditional sampling, it is expected that the particle dimension has been combined
            with the batch dimension.
        :param t: The current time step [0, 1].
        :param t_next: The next time step in [0, 1].
        :param cond_x: A batch of data to condition on. The first element is the sequence conditioning
            information represented as integers (not one hot encoded), and the second element is the
            optional structural conditioning data.
        :param cond_mask: Optional mask for the sequences in :code:`cond_x`, specifying which of the elements
            to condition on (with :code:`mask=True`). If not provided, all elements of :code:`cond_x` are
            conditioned on. This should be rank-3: :code:`(batch_size, num_particles, seq_len)`.
        :param downweight_pads: Whether to downweight particle probabilities based on the likelihood of
            pad tokens in positions where :code:`cond_mask=False`. This can be useful to ensure the
            model conditionally samples sequences of a fixed length.
        :return: The updated distribution parameters.
        """
        beta = model.bfn.schedule.compute_beta(t)
        beta_next = model.bfn.schedule.compute_beta(t_next)

        if cond_x is not None:
            cond_seq, cond_data = cond_x
        else:
            cond_seq = None
            cond_data = None

        score = None
        if cond_seq is None:
            pred = model.forward((theta, cond_data), t)

            # Final predicted probabilities
            probs = F.softmax(pred / self.temperature, dim=-1)

        else:
            cond_seq, cond_data = cond_x

            # Replace sequence probabilities with one-hot encodings for the conditioning data
            cond_seq_one_hot = F.one_hot(cond_seq, num_classes=theta.shape[-1]).to(
                theta.dtype
            )[:, None, :, :]

            pred = model.forward((theta, cond_data), t)
            pred = rearrange(pred, "(b p) l d -> b p l d", p=self.num_particles)
            probs = F.softmax(pred / self.temperature, dim=-1)

            # Calculate sampling probabilities for the particles
            particle_logits = self._get_particle_logits(
                probs, cond_seq_one_hot, cond_mask, downweight_pads=downweight_pads
            )
            particle_probs = F.softmax(particle_logits, dim=-1)

            # Sample particles based on the probabilities
            sampled_particles = torch.multinomial(
                particle_probs, num_samples=self.num_particles, replacement=True
            )
            probs = probs[
                torch.arange(probs.shape[0], device=probs.device)[..., None],
                sampled_particles,
            ]

            # Overwrite the probabilities for conditioned data with the one-hot encoding
            if self.replace_receiver:
                if cond_mask is None:
                    cond_mask = torch.ones(
                        probs.shape[:-1], device=probs.device, dtype=torch.bool
                    )
                probs = torch.where(~cond_mask[..., None], probs, cond_seq_one_hot)

            # Collapse the particle dimension back into the batch
            probs = rearrange(probs, "b p l d -> (b p) l d")

        theta = self.solver(theta, probs, beta, beta_next, t, t_next, score)

        return theta

    def __call__(
        self,
        model: UnpairedSequenceBFN,
        num_samples: int = 1,
        cond_x: tuple[torch.Tensor, UnpairedStructureData | None] | None = None,
        cond_mask: torch.Tensor | None = None,
        fix_length: bool = False,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """Generate unpaired data sequence samples via the trained BFN model.

        :param model: The model to use to generate samples.
        :param num_samples: The number of samples to generate. Only used if :code:`cond_x` is :code:`None`.
            Otherwise, if :code:`cond_x` is provided, one sample is generated per batch element.
        :param cond_x: A batch of unpaired data sequences represented as integers (not one hot encoded)
            to perform conditioning on, as well as optional structural conditioning data.
        :param cond_mask: A mask for the conditional data specifying which of the elements to
            condition on (with :code:`mask=True`).
        :param fix_length: Whether to fix the length of generated sequences by ensuring the model
            only generates non-padding tokens at positions where :code:`cond_mask=False`.
        :param progress_bar: Whether to display a progress bar of the time in the generative process.
        :return: The generated unpaired sequence samples.
        """
        if cond_x is not None:
            cond_seq, cond_data = cond_x

            if len(cond_seq.shape) < 2:
                cond_seq = cond_seq[None]

            batch_size = cond_seq.shape[0] * self.num_particles

            # Broadcast the conditional data to match the number of particles
            batch_particle_idx = torch.arange(
                cond_seq.shape[0], device=cond_seq.device
            ).repeat_interleave(self.num_particles)

            if cond_data is not None:
                cond_data = cond_data[batch_particle_idx]

            cond_x = (cond_seq, cond_data)

            if cond_mask is not None:
                if len(cond_mask.shape) < 2:
                    cond_mask = cond_mask[None]

                cond_mask = cond_mask.to(cond_seq.device)
                cond_mask = rearrange(
                    cond_mask[batch_particle_idx],
                    "(b p) l -> b p l",
                    p=self.num_particles,
                )
        else:
            cond_data = None
            batch_size = num_samples

        theta = model.bfn.get_prior_input_distribution(
            self.t_start, batch_size=batch_size
        ).to(model.device)

        # Start at t_start
        time_steps = torch.linspace(
            self.t_start, 1, self.num_steps, device=model.device
        )

        iterator = range(self.num_steps - 1)
        if progress_bar:
            iterator = tqdm(iterator, desc=f"Time step in generative process")

        for i in iterator:
            t_start = time_steps[i].expand(theta.shape[0])
            t_end = time_steps[i + 1].expand(theta.shape[0])

            theta = self._step(
                model=model,
                theta=theta,
                t=t_start,
                t_next=t_end,
                cond_x=cond_x,
                cond_mask=cond_mask,
                downweight_pads=fix_length,
            )

        # Perform final forward pass at t=1
        t = time_steps[-1].expand(theta.shape[0])
        theta = model.forward((theta, cond_data), t)

        # For the final step of SMC sampling, take the particle with the highest probability
        if cond_x is not None:
            cond_seq, cond_data = cond_x
            cond_seq_one_hot = F.one_hot(cond_seq, num_classes=theta.shape[-1]).to(
                theta.dtype
            )

            theta = rearrange(theta, "(b p) l d -> b p l d", p=self.num_particles)

            probs = F.softmax(theta / self.temperature, dim=-1)
            particle_probs = self._get_particle_logits(
                probs,
                cond_seq_one_hot[:, None, :, :],
                cond_mask,
                downweight_pads=fix_length,
            )
            sample_idx = particle_probs.argmax(-1)

            theta = theta[torch.arange(theta.shape[0], device=theta.device), sample_idx]

        sample = model.bfn.sample_from_logits(theta)
        return sample


class PairedSequenceBFN(
    SequenceGenerativeModel[
        tuple[torch.Tensor, torch.Tensor], PairedSequenceTransformer
    ]
):
    """
    A BFN model that generates paired data sequences.
    """

    def __init__(
        self,
        cfg: PairedSequenceModelConfig,
        network: PairedSequenceTransformer,
        bfn: DiscreteBFN,
    ):
        """
        :param cfg: A configuration object containing the model configuration options.
        :param network: The prediction network containing all of the model's learnable weights.
        :param bfn: The Bayesian flow network used to generate samples.
        """
        super().__init__(cfg, network)

        self.bfn = bfn

        # These are set at the start of training using the datamodule's region indices
        self.vh_fwr_mask = None
        self.vl_fwr_mask = None

        if cfg.vh_ckpt_path is not None:
            vh_ckpt = torch.load(
                cfg.vh_ckpt_path, map_location=self.device, weights_only=True
            )
            network_key = (
                "ema_network.module." if self.network.freeze_heavy else "network."
            )
            vh_state_dict = {
                k.removeprefix(network_key): v
                for k, v in vh_ckpt["state_dict"].items()
                if k.startswith(network_key)
            }
            self.network.backbone_heavy.load_state_dict(vh_state_dict)

        if cfg.vl_ckpt_path is not None:
            vl_ckpt = torch.load(
                cfg.vl_ckpt_path, map_location=self.device, weights_only=True
            )
            network_key = (
                "ema_network.module." if self.network.freeze_heavy else "network."
            )
            vl_state_dict = {
                k.removeprefix(network_key): v
                for k, v in vl_ckpt["state_dict"].items()
                if k.startswith(network_key)
            }
            self.network.backbone_light.load_state_dict(vl_state_dict)

    def dummy_forward(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None],
    ) -> None:
        """
        Perform a dummy forward pass to initialise the weights of the model,
        using a batch directly loaded by the dataloader.

        :param batch: A batch of data to perform a forward pass on.
        """
        (vh, vl), cond_data = batch
        vh = F.one_hot(vh, num_classes=len(AA1_INDEX)).float()
        vl = F.one_hot(vl, num_classes=len(AA1_INDEX)).float()
        t = torch.rand(vh.shape[0], device=vh.device)
        self.network(vh, vl, t, cond_data)
        self.ema_network(vh, vl, t, cond_data)

    def on_fit_start(self) -> None:
        """Initialises the masks for the FWR regions."""
        self.vh_fwr_mask = torch.zeros(
            (self.trainer.datamodule.vh_length,), device=self.device, dtype=torch.bool
        )
        self.vl_fwr_mask = torch.zeros(
            (self.trainer.datamodule.vl_length,), device=self.device, dtype=torch.bool
        )

        vh_indices = self.trainer.datamodule.vh_region_indices
        vl_indices = self.trainer.datamodule.vl_region_indices

        fwr_regions = ["fwr1", "fwr2", "fwr3", "fwr4"]
        for region in fwr_regions:
            self.vh_fwr_mask[..., vh_indices[region]] = True
            self.vl_fwr_mask[..., vl_indices[region]] = True

    def transfer_batch_to_device(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], dict[str, Any]],
        device: torch.device,
        dataloader_idx: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None]:
        """
        Transfers a batch to the specified device. The batch is expected to be the output
        of the default collate function applied to the output of :class:`PairedSequenceDataset`
        (or child classes).

        :param batch: The batch to transfer, consisting of a tuple of VH/VL sequences and a
            dictionary containing structural conditioning data.
        :param device: The device to transfer the batch to.
        :param dataloader_idx: The index of the dataloader the batch came from.
        :return: The batch transferred to the specified device.
        """
        (vh, vl), cond_data = batch
        vh = vh.to(device)
        vl = vl.to(device)

        if cond_data:
            cond_data = PairedStructureData(
                vh=StructureData(**cond_data["vh"]),
                vl=StructureData(**cond_data["vl"]),
                epitope=StructureData(**cond_data["epitope"]),
            )
            cond_data = cond_data.to(device)
        else:
            cond_data = None

        return (vh, vl), cond_data

    def forward(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None],
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model, taking in input data and returning a tensor
        containing the predicted probabilities for all amino acids.

        :param batch: A tuple of the current amino acid logits for the VH/VL chains, which should be two
            rank-3 tensor of shape :code:`(batch_size, vh_seq_len, num_classes)`, and a dictionary
            of data to use for explicit conditioning.
        :param t: The current time in the generative process as a rank-1 tensor of shape :code:`(batch_size,)`.
        :return: A tuple of tensors containing the predicted amino acid logits for the VH and VL chains.
        """
        if self.training:
            network = self.network
        else:
            network = self.ema_network

        (inputs_vh, inputs_vl), cond_data = batch
        probs_vh = F.softmax(inputs_vh, dim=-1)
        probs_vl = F.softmax(inputs_vl, dim=-1)
        logits_vh, logits_vl = network(probs_vh, probs_vl, t, cond_data)

        return logits_vh, logits_vl

    def vlb(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None],
        num_time_steps: int = 10,
        num_flow_samples: int = 10,
    ) -> torch.Tensor:
        """
        Estimates a variational lower bound on the log likelihood of the input data :code:`batch`
        under the model, using the continuous time loss function and the reconstruction loss
        function. Specifically this estimates the integral of eq. 20 from the
        BFN paper (https://arxiv.org/pdf/2308.07037), where :math:`L^n` is replaced with :math:`L^{\infty}`,
        over t in [0, 1].

        :param batch: A batch of paired data sequences represented as integers (not one-hot encoded)
            and optional structural conditioning data.
        :param num_time_steps: The number of time steps to use to discretize the interval [0, 1].
        :param num_flow_samples: The number of samples to draw from the flow distribution for each time step.
        :return: A tensor containing the estimated VLB for each batch element.
        """
        (vh, vl), cond_data = batch
        x = torch.cat([vh, vl], dim=-1)

        # Compute the continuous time loss per batch element
        ct_loss = torch.zeros(
            (num_time_steps, num_flow_samples, x.shape[0]), device=x.device
        )

        time_steps = torch.linspace(1e-6, 1, num_time_steps, device=x.device)
        for i, t in enumerate(time_steps):

            # Expand t
            t = t[None].expand(x.shape[0])

            beta = self.bfn.schedule.compute_beta(t)
            alpha = self.bfn.schedule.compute_alpha(t)
            alpha_vh = alpha[..., : vh.shape[1]]
            alpha_vl = alpha[..., vh.shape[1] :]

            # Sample from the flow distribution for every sequence in the batch
            for j in range(num_flow_samples):
                theta = self.bfn.sample_flow(x, beta)
                theta_vh, theta_vl = theta[:, : vh.shape[1]], theta[:, vh.shape[1] :]

                with torch.no_grad():
                    logits_vh, logits_vl = self.forward(
                        ((theta_vh, theta_vl), cond_data), t
                    )

                ct_loss_vh, ct_loss_vl = self.continuous_time_loss(
                    (vh, vl),
                    (logits_vh, logits_vl),
                    (alpha_vh, alpha_vl),
                )

                mse = torch.sum(torch.cat([ct_loss_vh, ct_loss_vl], dim=-1), dim=-1)
                ct_loss[i, j] = mse

        # Compute the expectation over flow samples and integrate over time
        ct_loss = torch.trapezoid(ct_loss.mean(dim=1), time_steps, dim=0)

        # Compute reconstruction loss (at t=1)
        t_1 = torch.ones(x.shape[0], device=x.device)
        beta_1 = self.bfn.schedule.compute_beta(t_1)

        recon_loss = torch.zeros(x.shape[0], device=x.device)
        for _ in range(num_flow_samples):
            theta_1 = self.bfn.sample_flow(x, beta_1)
            theta_1_vh, theta_1_vl = (
                theta_1[:, : vh.shape[1]],
                theta_1[:, vh.shape[1] :],
            )

            with torch.no_grad():
                logits_1_vh, logits_1_vl = self.forward(
                    ((theta_1_vh, theta_1_vl), cond_data), t_1
                )

            logits_1 = torch.cat([logits_1_vh, logits_1_vl], dim=-2)
            recon_loss += -self.bfn.logL(logits_1, x, None)

        recon_loss /= num_flow_samples

        # Scale by the number of tokens
        return -(ct_loss + recon_loss) / x.shape[1]

    @staticmethod
    def continuous_time_loss(
        true: tuple[torch.Tensor, torch.Tensor],
        pred: tuple[torch.Tensor, torch.Tensor],
        alpha: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the continuous time loss function from the BFN paper (eq. 41).

        :param true: The true VH and VL chain sequences, encoded as rank-2 integer tensors.
        :param pred: The predicted VH and VL amino acid logits, encoded as rank-3 tensors.
        :param alpha: The alpha values for the VH and VL chains (derivative of the accuracy schedule).
            Used to scale the loss.
        :return: The continuous time loss for the VH and VL chains as a rank-2 tensor of shape :code:`(batch_size, seq_len)`.
        """
        true_vh, true_vl = true
        pred_vh, pred_vl = pred
        alpha_vh, alpha_vl = alpha

        probs_vh = F.softmax(pred_vh, dim=-1)
        probs_vl = F.softmax(pred_vl, dim=-1)

        vh_one_hot = F.one_hot(true_vh, num_classes=len(AA1_INDEX)).to(
            dtype=pred_vh.dtype
        )
        vh_mse = F.mse_loss(probs_vh, vh_one_hot, reduction="none")

        vl_one_hot = F.one_hot(true_vl, num_classes=len(AA1_INDEX)).to(
            dtype=pred_vl.dtype
        )
        vl_mse = F.mse_loss(probs_vl, vl_one_hot, reduction="none")

        # Sum over classes
        vh_loss = torch.sum(alpha_vh[..., None] * vh_mse, dim=-1)
        vl_loss = torch.sum(alpha_vl[..., None] * vl_mse, dim=-1)

        return vh_loss, vl_loss

    def _mask_fwr_structures(
        self, batch: tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData]
    ) -> tuple[PairedStructureData, torch.Tensor]:
        """
        Masks the structures in the input batch, either "all" (the entire structure is masked)
        or "fwrs" (only the framework regions are masked).

        :param batch: The batch to mask, containing VH and VL sequences and structural data.
        :return: The structure conditioning data with the padding masks updated and a boolean
            tensor denoting which structures in the batch had their framework regions masked.
        """
        (vh, vl), cond_data = batch
        batch_size = vh.shape[0]
        fwr_mask = torch.rand(batch_size, device=vh.device) < self._mask_fwr_rate

        # Mask only the framework regions of the structures given by the fwr_mask
        cond_data.vh.mask[fwr_mask] |= self.vh_fwr_mask[None]
        cond_data.vl.mask[fwr_mask] |= self.vl_fwr_mask[None]

        return cond_data, fwr_mask

    def training_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None],
    ) -> torch.Tensor:
        """
        A single training step for the model. This uses the continuous time loss function
        from the BFN paper.

        :param batch: Batch containing VH and VL sequences represented as integers from 0-20
            and optional structural conditioning data.
        :return: A scalar tensor containing the loss.
        """
        (vh, vl), cond_data = batch
        x = torch.cat([vh, vl], dim=-1)

        # Randomly mask structures
        if cond_data is not None:
            cond_data, _ = self._mask_fwr_structures(batch)

        # sample a time for each batch element
        t = torch.rand((x.shape[0],), device=x.device)

        beta = self.bfn.schedule.compute_beta(t)
        alpha = self.bfn.schedule.compute_alpha(t)
        alpha_vh = alpha[..., : vh.shape[1]]
        alpha_vl = alpha[..., vh.shape[1] :]

        theta = self.bfn.sample_flow(x, beta)
        theta_vh, theta_vl = theta[:, : vh.shape[1]], theta[:, vh.shape[1] :]
        pred_vh, pred_vl = self.forward(((theta_vh, theta_vl), cond_data), t)

        vh_loss, vl_loss = self.continuous_time_loss(
            (vh, vl), (pred_vh, pred_vl), (alpha_vh, alpha_vl)
        )

        # Sum over tokens but take the mean over sequences in the batch
        vh_loss = torch.sum(vh_loss, dim=-1).mean()
        vl_loss = torch.sum(vl_loss, dim=-1).mean()

        self.log("train_vh_loss", vh_loss.item(), sync_dist=True)
        self.log("train_vl_loss", vl_loss.item(), sync_dist=True)
        self.log(
            "train_loss",
            vh_loss.item() + vl_loss.item(),
            sync_dist=True,
        )

        return vh_loss + vl_loss

    def validation_step(
        self,
        batch: tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None],
    ):
        """
        A single validation step for the model. This uses the continuous time loss function
        from the BFN paper.

        :param batch: Batch containing VH and VL sequences represented as integers from 0-20
            and optional structural conditioning data.
        :return: A scalar tensor containing the loss.
        """
        (vh, vl), cond_data = batch
        x = torch.cat([vh, vl], dim=-1)

        # Randomly mask structures
        if cond_data is not None:
            cond_data, fwr_mask = self._mask_fwr_structures(batch)
        else:
            fwr_mask = None

        # sample a time for each batch element
        t = torch.rand((x.shape[0],), device=x.device)

        beta = self.bfn.schedule.compute_beta(t)
        alpha = self.bfn.schedule.compute_alpha(t)
        alpha_vh = alpha[..., : vh.shape[1]]
        alpha_vl = alpha[..., vh.shape[1] :]

        theta = self.bfn.sample_flow(x, beta)
        theta_vh, theta_vl = theta[:, : vh.shape[1]], theta[:, vh.shape[1] :]
        pred_vh, pred_vl = self.forward(((theta_vh, theta_vl), cond_data), t)

        vh_loss, vl_loss = self.continuous_time_loss(
            (vh, vl), (pred_vh, pred_vl), (alpha_vh, alpha_vl)
        )

        # when storing in the metrics dict, take the sum over tokens
        val_vh_loss = torch.sum(vh_loss, dim=-1)
        val_vl_loss = torch.sum(vl_loss, dim=-1)
        metrics = {
            "val_vh_loss": val_vh_loss,
            "val_vl_loss": val_vl_loss,
            "val_loss": val_vh_loss + val_vl_loss,
            "t": t,
        }

        # sum over the tokens for each individual region to track performance on different IMGT regions
        for region, idx in self.trainer.datamodule.vh_region_indices.items():
            metrics[f"val_H-{region}_loss"] = torch.sum(vh_loss[..., idx], dim=-1)

        for region, idx in self.trainer.datamodule.vl_region_indices.items():
            metrics[f"val_L-{region}_loss"] = torch.sum(vl_loss[..., idx], dim=-1)

        # If structure masking is used, store the loss when the framework was masked and not masked
        if fwr_mask is not None:
            metrics["fwr_masked/val_vh_loss"] = val_vh_loss[fwr_mask]
            metrics["fwr_masked/val_vl_loss"] = val_vl_loss[fwr_mask]
            metrics["fwr_masked/val_loss"] = (
                val_vh_loss[fwr_mask] + val_vl_loss[fwr_mask]
            )

            metrics["no_fwr_mask/val_vh_loss"] = val_vh_loss[~fwr_mask]
            metrics["no_fwr_mask/val_vl_loss"] = val_vl_loss[~fwr_mask]
            metrics["no_fwr_mask/val_loss"] = (
                val_vh_loss[~fwr_mask] + val_vl_loss[~fwr_mask]
            )

        self._val_outputs.append(metrics)

    def on_validation_epoch_end(self) -> None:
        """
        Logs the mean of the validation metrics over the entire validation set,
        as well as the mean of the metrics for each time bin.
        """

        # concatenate the metrics over all validation batches together
        metric_names = set(
            [key for outputs in self._val_outputs for key in outputs.keys()]
        )
        all_metrics = {
            k: torch.cat([metrics[k] for metrics in self._val_outputs if k in metrics])
            for k in metric_names
        }

        t_binned = torch.bucketize(all_metrics["t"], self.t_bins) - 1

        for metric, values in all_metrics.items():
            if metric == "t":
                continue

            self.log(metric, values.mean().item(), sync_dist=True)

            # if the metric is not the same length as the time, skip binning
            if len(values) != len(t_binned):
                continue

            for t in range(self._num_eval_time_bins):
                bin_start = self.t_bins[t].item()
                bin_end = self.t_bins[t + 1].item()
                bin_mask = t_binned == t
                self.log(
                    f"{metric}_{bin_start:.2f}-{bin_end:.2f}",
                    values[bin_mask].mean().item(),
                    sync_dist=True,
                )


class PairedSequenceBFNSampler(SequenceSampler[PairedSequenceBFN]):
    """
    A sampler that uses a BFN to generate samples of paired data sequences.
    Conditioning is performed using the score-based method detailed in https://arxiv.org/pdf/2209.14687.
    """

    def __init__(
        self,
        vh_len: int,
        num_steps: int,
        solver: DiscreteBFNSDESolver,
        t_start: float = 1e-6,
        temperature: float = 1.0,
        num_particles: int = 32,
        replace_receiver: bool = True,
    ):
        """
        :param vh_len: The length of the VH chain. Used to infer the length of the VL chain
            as well using the total length of the paired sequence tensor.
        :param num_steps: The number of steps to take in the generative process.
        :param solver: The type of solver (update rule) to use in the generative process.
        :param t_start: The starting time in the generative process.
        :param temperature: The temperature to use when sampling from the model.
        :param num_particles: The number of particles to use in SMC conditioning.
        :param replace_receiver: Whether to replace the predicted probabilities in the receiver distribution
            with the ground-truth sequence for the conditioning data. This typically guarantees that the model
            samples sequences that match the conditioning data.
        """
        self.vh_len = vh_len
        self.num_steps = num_steps
        self.solver = solver
        self.t_start = t_start
        self.temperature = temperature
        self.num_particles = num_particles
        self.replace_receiver = replace_receiver

    def _step(
        self,
        model: PairedSequenceBFN,
        theta: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond_x: tuple[torch.Tensor, PairedStructureData | None] | None,
        cond_mask: torch.Tensor | None,
        downweight_pads: bool = False,
    ) -> torch.Tensor:
        """
        Perform a single step in the generative process, returning an updated tensor of parameters.

        :param model: The model to use to generate samples.
        :param theta: The current distribution parameters in the generative process as a
            rank-3 tensor of shape :code:`(batch_size, seq_len, num_classes)`. When using SMC-based
            conditional sampling, it is expected that the particle dimension has been combined
            with the batch dimension.
        :param t: The current time step [0, 1].
        :param t_next: The next time step in [0, 1].
        :param cond_x: A batch of data to condition on. The first element is the sequence conditioning
            information represented as integers (not one hot encoded), and the second element is the optional
            structural conditioning data.
        :param cond_mask: Optional mask for the sequences in :code:`cond_x`, specifying which of the elements
            to condition on (with :code:`mask=True`). If not provided, all elements of :code:`cond_x` are
            conditioned on. This should be rank-3: :code:`(batch_size, num_particles, seq_len)`.
        :param downweight_pads: Whether to downweight particle probabilities based on the likelihood of
            pad tokens in positions where :code:`cond_mask=False`. This can be useful to ensure the
            model conditionally samples sequences of a fixed length.
        :return: The updated distribution parameters.
        """
        theta_vh, theta_vl = theta[..., : self.vh_len, :], theta[..., self.vh_len :, :]

        beta = model.bfn.schedule.compute_beta(t)
        beta_next = model.bfn.schedule.compute_beta(t_next)

        if cond_x is not None:
            cond_seq, cond_data = cond_x
        else:
            cond_seq = None
            cond_data = None

        # No conditioning data
        if cond_seq is None:
            pred_vh, pred_vl = model.forward(((theta_vh, theta_vl), cond_data), t)
            pred = torch.cat([pred_vh, pred_vl], dim=-2)

            # Final predicted probabilities
            probs = F.softmax(pred / self.temperature, dim=-1)

        # If conditioning data provided, use particle filtering
        else:

            # Replace sequence probabilities with one-hot encodings for the conditioning data
            cond_seq_one_hot = F.one_hot(cond_seq, num_classes=theta.shape[-1]).to(
                theta.dtype
            )[:, None, :, :]

            # Forward pass with conditioning data
            pred_vh, pred_vl = model.forward(
                (
                    (
                        theta[..., : self.vh_len, :],
                        theta[..., self.vh_len :, :],
                    ),
                    cond_data,
                ),
                t,
            )
            pred = torch.cat([pred_vh, pred_vl], dim=-2)
            pred = rearrange(pred, "(b p) l d -> b p l d", p=self.num_particles)
            probs = F.softmax(pred / self.temperature, dim=-1)

            # Calculate sampling probabilities for the particles
            particle_logits = self._get_particle_logits(
                probs, cond_seq_one_hot, cond_mask, downweight_pads=downweight_pads
            )
            particle_probs = F.softmax(particle_logits, dim=-1)

            # Sample particles based on the probabilities
            sampled_particles = torch.multinomial(
                particle_probs, num_samples=self.num_particles, replacement=True
            )
            probs = probs[
                torch.arange(probs.shape[0], device=probs.device)[..., None],
                sampled_particles,
            ]

            # Optionally replace the receiver distribution with the conditioning data
            if self.replace_receiver:
                if cond_mask is None:
                    cond_mask = torch.ones(
                        probs.shape[:-1], device=probs.device, dtype=torch.bool
                    )
                probs = torch.where(~cond_mask[..., None], probs, cond_seq_one_hot)

            # Collapse the particle dimension back into the batch
            probs = rearrange(probs, "b p l d -> (b p) l d")

        theta = self.solver(theta, probs, beta, beta_next, t, t_next)

        return theta

    def __call__(
        self,
        model: PairedSequenceBFN,
        num_samples: int = 1,
        cond_x: (
            tuple[tuple[torch.Tensor, torch.Tensor], PairedStructureData | None] | None
        ) = None,
        cond_mask: tuple[torch.Tensor, torch.Tensor] | None = None,
        fix_length: bool = False,
        progress_bar: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate paired sequence samples via the trained BFN model.

        :param model: The model to use to generate samples.
        :param num_samples: The number of samples to generate. Only used if :code:`cond_x` is :code:`None`.
            Otherwise, if :code:`cond_x` is provided, one sample is generated per batch element.
        :param cond_x: A batch of paired data with sequences represented as integers
            (not one hot encoded) to perform conditioning on, as well as optional structural
            conditioning data. The first tensor is for the VH chain and the second for the VL chain.
        :param cond_mask: A mask for the conditional data (the first for the VH and the second for the VL),
            specifying which of the elements to condition on (with :code:`mask=True`).
        :param fix_length: Whether to fix the length of generated sequences by ensuring the model
            only generates non-padding tokens at positions where :code:`cond_mask=False`.
        :param progress_bar: Whether to display a progress bar of the time in the generative process.
        :return: The generated samples for the VH and VL chains.
        """
        # Check the conditioning tensors have the correct dimensions and expand
        # to match the number of SMC particles
        if cond_x is not None:
            (cond_vh, cond_vl), cond_data = cond_x

            if len(cond_vh.shape) < 2:
                cond_vh = cond_vh[None]
            if len(cond_vl.shape) < 2:
                cond_vl = cond_vl[None]

            if cond_vh.shape[0] != cond_vl.shape[0]:
                raise ValueError("VH and VL chains must have the same batch size.")

            cond_seq = torch.cat([cond_vh, cond_vl], dim=-1)
            batch_size = cond_seq.shape[0] * self.num_particles

            # Expand the conditional data to match the number of particles
            batch_particle_idx = torch.arange(
                cond_seq.shape[0], device=cond_seq.device
            ).repeat_interleave(self.num_particles)

            if cond_data is not None:
                cond_data = cond_data[batch_particle_idx]

            cond_x = (cond_seq, cond_data)

            if cond_mask is not None:
                cond_mask_vh, cond_mask_vl = cond_mask
                if len(cond_mask_vh.shape) < 2:
                    cond_mask_vh = cond_mask_vh[None]
                if len(cond_mask_vl.shape) < 2:
                    cond_mask_vl = cond_mask_vl[None]

                if cond_mask_vh.shape[0] != cond_mask_vl.shape[0]:
                    raise ValueError("VH and VL chains must have the same batch size.")

                cond_mask = torch.cat([cond_mask_vh, cond_mask_vl], dim=-1)
                cond_mask = cond_mask.to(cond_seq.device)
                cond_mask = rearrange(
                    cond_mask[batch_particle_idx],
                    "(b p) l -> b p l",
                    p=self.num_particles,
                )

        else:
            cond_data = None
            batch_size = num_samples

        theta = model.bfn.get_prior_input_distribution(
            self.t_start, batch_size=batch_size
        ).to(model.device)

        # Start at t_start
        time_steps = torch.linspace(
            self.t_start, 1, self.num_steps, device=model.device
        )

        iterator = range(self.num_steps - 1)
        if progress_bar:
            iterator = tqdm(iterator, desc=f"Time step in generative process")

        for i in iterator:
            t = time_steps[i].expand(theta.shape[0])
            t_next = time_steps[i + 1].expand(theta.shape[0])

            theta = self._step(
                model=model,
                theta=theta,
                t=t,
                t_next=t_next,
                cond_x=cond_x,
                cond_mask=cond_mask,
                downweight_pads=fix_length,
            )

        # Perform final forward pass at t=1
        t = time_steps[-1].expand(theta.shape[0])
        theta_vh, theta_vl = (
            theta[..., : self.vh_len, :],
            theta[..., self.vh_len :, :],
        )
        pred_vh, pred_vl = model.forward(((theta_vh, theta_vl), cond_data), t)
        theta = torch.cat([pred_vh, pred_vl], dim=-2)

        # For the final step of conditional sampling, take the particle with the highest probability
        if cond_x is not None:
            cond_seq, cond_data = cond_x
            cond_seq_one_hot = F.one_hot(cond_seq, num_classes=theta.shape[-1]).to(
                theta.dtype
            )

            theta = rearrange(theta, "(b p) l d -> b p l d", p=self.num_particles)

            probs = F.softmax(theta / self.temperature, dim=-1)
            particle_probs = self._get_particle_logits(
                probs,
                cond_seq_one_hot[:, None, :, :],
                cond_mask,
                downweight_pads=fix_length,
            )
            sample_idx = particle_probs.argmax(-1)
            theta = theta[torch.arange(theta.shape[0], device=theta.device), sample_idx]

        sample = model.bfn.sample_from_logits(theta)
        vh_sample, vl_sample = sample[:, : self.vh_len], sample[:, self.vh_len :]
        return vh_sample, vl_sample
