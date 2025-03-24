"""
Contains the Transformer-based architecture that predicts data distribution in Bayesian Flow Networks.
"""

import math

import torch.nn as nn
import torch
from typing import Tuple

from .config import SequenceTransformerConfig, PairedSequenceTransformerConfig
from ..data.structure import PairedStructureData, UnpairedStructureData
from ..nn.geometric import PairedStructureEncoder
from ..nn.transformer import CrossAttentionBlock, RobertaHead, SelfAttentionBlock


class SequenceTransformer(nn.Module):
    """
    A backbone network that uses transformer layers to predict amino acid distributions
    for a single data chain. The network inputs noised amino acid probabilities and
    predicts the updated output probabilities.

    In addition to the standard transformer layers, this network uses a fourier-based encoding
    of the time step or entropy of the input probabilities at the embedding layer. See
    https://arxiv.org/pdf/2107.00630 (page 16) for more details.
    """

    def __init__(self, cfg: SequenceTransformerConfig):
        """
        Initializes the Transformer network.

        Args:
            cfg: Configuration object for the Transformer network - see :class:`SequenceTransformerConfig`
            for the required parameters.
        """
        super().__init__()

        self.num_layers = cfg.num_layers
        self.embed_dim = cfg.embed_dim
        self.use_entropy_encoding = cfg.use_entropy_encoding

        self.embedding = nn.LazyLinear(cfg.embed_dim, bias=False)

        self.trunk = nn.ModuleDict()
        for i in range(cfg.num_layers):
            self.trunk[f"self_attention_layer_{i}"] = SelfAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=cfg.num_heads,
                dropout_p=cfg.dropout_p,
            )

        self.output_head = RobertaHead(
            embed_dim=cfg.embed_dim, output_dim=cfg.output_dim
        )

        fourier_features = torch.pi * 2 ** torch.arange(
            cfg.fourier_n_min, cfg.fourier_n_max, dtype=torch.float32
        )
        self.register_buffer("fourier_features", fourier_features[None])

    def embed(self, probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input amino acid probabilities and performs either a time or entropy
        encoding (depending on if :code:`use_entropy_encoding` is :code:`True`).

        :param probs: The input amino acid probabilities of shape :code:`(batch_size, seq_len, num_classes)`.
        :param t: The time tensor of shape :code:`(batch_size,)` on the interval [0, 1] giving the
            current time in the generative process.
        :return: The embedded representation.
        """
        if self.use_entropy_encoding:
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
            max_entropy = -math.log(probs.shape[-1])

            # express entropy relative to maximum
            features = torch.sqrt(1 - (entropy / max_entropy))
        else:
            features = t[..., None, None]

        # (batch_size, fourier_n_max)
        fourier_embedding = torch.cat(
            [
                torch.sin(features * self.fourier_features),
                torch.cos(features * self.fourier_features),
            ],
            dim=-1,
        )

        # concatenate the probabilities with the fourier features
        inputs = torch.cat([probs, fourier_embedding], dim=-1)
        s_i = self.embedding(inputs)

        return s_i

    def forward(
        self,
        probs: torch.Tensor,
        t: torch.Tensor,
        cond_data: UnpairedStructureData | None,
    ) -> torch.Tensor:
        """
        :param probs: Input tensor of shape :code:`(batch_size, seq_len, num_classes)` with input
            amino acid probabilities.
        :param t: The time tensor of shape :code:`(batch_size,)` on the interval [0, 1] giving the
            current time in the generative process. Only used if :code:`use_entropy_encoding` is :code:`False`.
        :param cond_data: Optional structural data to condition on.
        :return: Output tensor of shape :code:`(batch_size, seq_len, num_classes)` with predicted
            amino acid logits.
        """

        s_i = self.embed(probs, t)

        for i in range(self.num_layers):
            s_i = self.trunk[f"self_attention_layer_{i}"](s_i)

        logits = self.output_head(s_i)

        return logits


class SigmoidGLU(nn.Module):
    """A sigmoid-gated linear unit."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        linear_bias: bool = True,
        gate_bias: bool = True,
        gate_bias_value: float = 0.0,
    ):
        """
        :param input_dim: The dimension of the input tensor.
        :param output_dim: The dimension of the output tensor.
        :param linear_bias: Whether to include a bias term in the linear layer.
        :param gate_bias: Whether to include a bias term in the gate layer.
        :param gate_bias_value: The initialisation value for the gate bias.
        """
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=linear_bias)
        self.linear_gate = nn.Linear(input_dim, output_dim, bias=gate_bias)

        self.linear_gate.bias.data.fill_(gate_bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param x: :code:`(..., input_dim)` tensor of embeddings.
        :return: :code:`(..., output_dim)` tensor of updated embeddings.
        """
        return self.linear(x) * nn.functional.sigmoid(self.linear_gate(x))


class PairedSequenceTransformer(nn.Module):
    """
    Paired Transformer network that applies cross-attention between two Transformer
    networks at each layer to model interactions between data heavy and light chains.

    Notes:
        - Both :code:`backbone_heavy` and :code:`backbone_light` must have the same number of layers.
        - Weights in heavy or light networks can optionally be frozen.
    """

    def __init__(
        self,
        cfg: PairedSequenceTransformerConfig,
        backbone_heavy: SequenceTransformer,
        backbone_light: SequenceTransformer,
        structure_encoder: PairedStructureEncoder | None = None,
    ):
        """
        Initializes the PairedSequenceTransformer network.

        Args:
            cfg: Configuration object for the PairedTransformer network - see :class:`PairedSequenceTransformerConfig`
                for the required parameters.
            backbone_heavy: Backbone network for the heavy chain.
            backbone_light: Backbone network for the light chain.
            structure_encoder: Optional structure encoder module to condition on structural data.
        """
        super().__init__()

        if backbone_heavy.num_layers != backbone_light.num_layers:
            raise ValueError(
                "Heavy and light networks must have the same number of layers."
            )

        self.num_layers = backbone_heavy.num_layers
        self.backbone_heavy = backbone_heavy
        self.backbone_light = backbone_light
        self.freeze_heavy = cfg.freeze_heavy
        self.freeze_light = cfg.freeze_light
        self.freeze_all = cfg.freeze_all

        # Projections before cross attention are a linear layer
        self.vh_pre_projections = nn.ModuleList(
            [
                nn.Linear(backbone_heavy.embed_dim, cfg.embed_dim, bias=False)
                for _ in range(self.num_layers)
            ]
        )
        self.vl_pre_projections = nn.ModuleList(
            [
                nn.Linear(backbone_light.embed_dim, cfg.embed_dim, bias=False)
                for _ in range(self.num_layers)
            ]
        )

        self.cross_attention_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    dropout_p=cfg.dropout_p,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Projections after cross attention are a sigmoid-gated linear unit
        self.vh_post_projections = nn.ModuleList(
            [
                SigmoidGLU(
                    cfg.embed_dim,
                    backbone_heavy.embed_dim,
                    linear_bias=False,
                    gate_bias_value=cfg.gate_bias_value,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.vl_post_projections = nn.ModuleList(
            [
                SigmoidGLU(
                    cfg.embed_dim,
                    backbone_light.embed_dim,
                    linear_bias=False,
                    gate_bias_value=cfg.gate_bias_value,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.freeze_all:
            for param in self.parameters():
                param.requires_grad = False
        else:
            if self.freeze_heavy:
                for param in self.backbone_heavy.parameters():
                    param.requires_grad = False
            if self.freeze_light:
                for param in self.backbone_light.parameters():
                    param.requires_grad = False

        self.structure_encoder = structure_encoder

    def forward(
        self,
        probs_vh: torch.Tensor,
        probs_vl: torch.Tensor,
        t: torch.Tensor,
        cond_data: PairedStructureData | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param probs_vh: The input tensor for the heavy chain of shape :code:`(batch_size, seq_len, embed_dim)`,
            consisting of the amino acid probabilities.
        :param probs_vl: The input tensor for the light chain of shape :code:`(batch_size, seq_len, embed_dim)`,
            consisting of the amino acid probabilities.
        :param t: The time tensor of shape :code:`(batch_size,)` on the interval [0, 1] giving the
            current time in the generative process.
        :param cond_data: Optional paired structural data to condition on.
        :return: A 2-tuple of output tensors of shape :code:`(batch_size, seq_len, num_classes)` with predicted
            amino acid logits for the heavy and light chains.
        """
        vh_repr = self.backbone_heavy.embed(probs_vh, t)
        vl_repr = self.backbone_light.embed(probs_vl, t)

        # If conditioning data is provided use the structure encoder
        if cond_data is not None and self.structure_encoder is not None:
            vh_repr, vl_repr = self.structure_encoder(vh_repr, vl_repr, cond_data)

        for i in range(self.num_layers):
            # Self-attention on each chain
            vh_repr = self.backbone_heavy.trunk[f"self_attention_layer_{i}"](vh_repr)
            vl_repr = self.backbone_light.trunk[f"self_attention_layer_{i}"](vl_repr)

            # Project to shared embedding space
            vh_repr_cross = self.vh_pre_projections[i](vh_repr)
            vl_repr_cross = self.vl_pre_projections[i](vl_repr)

            # Cross-attention between heavy and light representations
            vh_repr_cross, vl_repr_cross = self.cross_attention_blocks[i](
                vh_repr_cross, vl_repr_cross
            )

            # Project back to original embedding spaces, with residual connection
            vh_repr = vh_repr + self.vh_post_projections[i](vh_repr_cross)
            vl_repr = vl_repr + self.vl_post_projections[i](vl_repr_cross)

        logits_vh = self.backbone_heavy.output_head(vh_repr)
        logits_vl = self.backbone_light.output_head(vl_repr)

        return logits_vh, logits_vl
