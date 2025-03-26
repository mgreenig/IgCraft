"""
Layers for encoding protein structural data.
"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import einsum, rearrange

from ..data.structure import PairedStructureData
from .transformer import SwiGLUTransition


def sinusoidal_positional_encoding(x: torch.Tensor, base: int = 1000) -> torch.Tensor:
    """
    Adds a sinusoidal positional encoding to the input tensor, as in the original
    Transformer (https://arxiv.org/abs/1706.03762).

    :param x: Tensor of shape :code:`(B, N, D)` containing the input embeddings to which the positional
        encoding will be added.
    :param base: Value used to scale the sin/cos function outputs in the final encoding.
        The larger this number is, the less variability there will be between encodings for different values.
        The original Transformer paper used 10000 here but we prefer a smaller number to introduce
        more variation between encodings for shorter sequences.
    :return: Tensor of shape :code:`(B, N, D)` containing the input embeddings with the positional encoding added.
    """

    positions = torch.arange(x.shape[-2], dtype=torch.float32, device=x.device)
    channels = torch.arange(x.shape[-1], dtype=torch.float32, device=x.device)
    encoding = torch.where(
        channels % 2 == 0,
        torch.sin(positions[..., None] / (base ** (2 * channels / channels[-1]))[None]),
        torch.cos(positions[..., None] / (base ** (2 * channels / channels[-1]))[None]),
    )

    return x + encoding


class GeometricMHA(nn.Module):
    """
    Implements geometric multi-head attention from ESM3 (algorithm 6).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        """
        Initializes the geometric MHA layer.

        :param embed_dim: The hidden dimension of the input embeddings.
        :param num_heads: The number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # x15 for x5 features (Q_r, K_r, Q_d, K_d, V) and x3 dimensions
        qkv_outdim = self.num_heads * 15
        self.qkv_linear = nn.Linear(embed_dim, qkv_outdim, bias=False)
        self.out = nn.Linear(self.num_heads * 3, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.r_scale = nn.Parameter(torch.zeros((1, 1, 1, num_heads)))
        self.d_scale = nn.Parameter(torch.zeros((1, 1, 1, num_heads)))

    def forward(
        self,
        x: torch.Tensor,
        frames: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        For an input tensor of embeddings and a tuple of residue frames (rotations and translations),
        updates the input embeddings with an SE(3)-equivariant multi-head attention mechanism.

        :param x: Input node embeddings of shape :code:`(..., length, embed_dim)`.
        :param frames: Tuple of rotations and translations of shape :code:`(..., length, 3, 3)`
            and :code:`(..., length, 3)` respectively.
        :param mask: Optional padding mask for the input embeddings of shape :code:`(..., length)`.
            Attention weights will be set to 0 for elements with :code:`mask=True`.
        :return: Updated node embeddings of shape :code:`(..., length, embed_dim)`.
        """
        rotations, translations = frames

        x = self.layer_norm(x)

        QKV = self.qkv_linear(x)
        QKV = rearrange(
            QKV,
            "... r (f h d) -> ... r f h d",  # (..., residue, feature, head, dim)
            h=self.num_heads,
            f=5,
            d=3,
        )

        # Rotate all the QKV vectors
        QKV = einsum(rotations, QKV, "... r i d, ... r f h d -> ... r f h i")

        # Split into the queries, keys, and values via the "feature" dim
        Q_r, K_r, Q_d, K_d, V = QKV.unbind(dim=-3)

        # Rotational attention weights
        scale = 3**-0.5
        A_r = (
            einsum(Q_r, K_r, "... i h d, ... j h d -> ... i j h")
            * F.softplus(self.r_scale)
            * scale
        )

        # Distance attention weights
        query_d = Q_d + translations[:, :, None, :]
        key_d = K_d + translations[:, :, None, :]
        distances = torch.linalg.norm(query_d[:, :, None] - key_d[:, None], dim=-1)
        A_d = distances * F.softplus(self.d_scale) * scale

        attn_logits = A_r - A_d

        # Set attention logits to -inf where mask is True
        if mask is not None:
            attn_logit_mask = torch.where(
                mask[:, :, None] | mask[:, None, :], -1e9, 0.0
            )
            attn_logits += attn_logit_mask[..., None]

        # Combine attention weights and take weighted sum of values
        A = F.softmax(attn_logits, dim=2)
        outputs = einsum(A, V, "... i j h, ... j h d -> ... i h d")

        # Rotate back into the global frame and flatten (this is R^T @ v for every point v)
        outputs = rearrange(
            einsum(rotations, outputs, "... r j i, ... r h j -> ... r h i"),
            "... r h i -> ... r (h i)",
        )

        return self.out(outputs)


class PairedStructureEncoder(nn.Module):
    """Encodes paired VH/VL structures using geometric multi-head attention layers interleaved with MLPs."""

    def __init__(
        self,
        input_vh_dim: int,
        input_vl_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_p: float,
        gate_bias_value: float = -2.0,
    ):
        """
        Initializes the paired structure encoder.

        :param input_vh_dim: The dimension of the VH input embeddings. The structure encoder
            produces output embeddings of this dimension as well.
        :param input_vl_dim: The dimension of the VL input embeddings. The structure encoder
            produces output embeddings of this dimension as well.
        :param embed_dim: The hidden dimension of the geometric transformer.
        :param num_heads: The number of geometric attention heads in the geometric transformer.
        :param num_layers: The number of layers in the geometric transformer.
        :param dropout_p: The dropout probability for the SwiGLU transitions.
        :param gate_bias_value: The initialisation value for the output gate bias. Initialise to a large negative
            value to start with the output gate closed.
        """
        super().__init__()

        self.num_layers = num_layers

        self.vh_embedding = nn.Linear(input_vh_dim, embed_dim, bias=False)
        self.vl_embedding = nn.Linear(input_vh_dim, embed_dim, bias=False)
        self.target_embedding = nn.LazyLinear(embed_dim, bias=False)

        self.attention_layers = nn.ModuleList(
            [GeometricMHA(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.transition_layers = nn.ModuleList(
            [
                SwiGLUTransition(embed_dim, dropout_p=dropout_p)
                for _ in range(num_layers)
            ]
        )

        self.output_vh_layer = nn.Linear(embed_dim, input_vh_dim, bias=False)
        self.output_vh_gate = nn.Linear(embed_dim, input_vh_dim)
        self.output_vh_gate.bias.data.fill_(gate_bias_value)

        self.output_vl_layer = nn.Linear(embed_dim, input_vl_dim, bias=False)
        self.output_vl_gate = nn.Linear(embed_dim, input_vl_dim)
        self.output_vl_gate.bias.data.fill_(gate_bias_value)

    def forward(
        self,
        vh_x: torch.Tensor,
        vl_x: torch.Tensor,
        structure_data: PairedStructureData,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For an input tensor of embeddings and a structure data object,
        updates the input embeddings after performing geometric multi head attention
        on the input structure data.

        :param vh_x: Input embeddings for the VH chain of shape :code:`(batch_size, vh_length, embed_dim)`.
        :param vl_x: Input embeddings for the VL chain of shape :code:`(batch_size, vl_length, embed_dim)`.
        :param structure_data: Structural conditioning data for the VH/VL/epitope, stored under the
            keys :code:`vh`, :code:`vl`, and :code:`epitope` respectively. The :code:`epitope` key is optional.
        :return: Updated VH/VL embeddings of shape :code:`(batch_size, vh_length, embed_dim)`.
        """
        vh_rots, vh_trans = structure_data.vh.frames
        vl_rots, vl_trans = structure_data.vl.frames

        # Embed the residues and add positional encoding
        vh_emb = sinusoidal_positional_encoding(self.vh_embedding(vh_x))
        vl_emb = sinusoidal_positional_encoding(self.vl_embedding(vl_x))

        # Extract the target features
        target_rots, target_trans = structure_data.epitope.frames
        target_emb = self.target_embedding(structure_data.epitope.features)
        target_padding_mask = structure_data.epitope.mask

        emb = torch.cat([vh_emb, vl_emb, target_emb], dim=1)
        rots = torch.cat([vh_rots, vl_rots, target_rots], dim=1)
        trans = torch.cat([vh_trans, vl_trans, target_trans], dim=1)
        padding_mask = torch.cat(
            [structure_data.vh.mask, structure_data.vl.mask, target_padding_mask], dim=1
        )

        for i in range(self.num_layers):
            emb = emb + self.attention_layers[i](emb, (rots, trans), padding_mask)
            emb = emb + self.transition_layers[i](emb)

        # Project with the output gate
        vh_len = vh_x.shape[1]
        vl_len = vl_x.shape[1]
        vh_emb = emb[:, :vh_len]
        vh_emb = self.output_vh_layer(vh_emb) * F.sigmoid(self.output_vh_gate(vh_emb))
        vl_emb = emb[:, vh_len : vh_len + vl_len]
        vl_emb = self.output_vl_layer(vl_emb) * F.sigmoid(self.output_vl_gate(vl_emb))

        # Use the mask to update the non-pad positions
        vh_update_mask = ~padding_mask[..., :vh_len, None]
        vl_update_mask = ~padding_mask[..., vh_len : vh_len + vl_len, None]
        vh_x = torch.where(vh_update_mask, vh_x + vh_emb, vh_x)
        vl_x = torch.where(vl_update_mask, vl_x + vl_emb, vl_x)

        return vh_x, vl_x
