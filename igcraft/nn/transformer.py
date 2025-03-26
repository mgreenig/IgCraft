"""Attention-based network architectures.

Example usage:
    >>> import torch
    >>> from igcraft.model.attention import SelfAttentionBlock
    >>> s_i = torch.randn(1, 128, 64)
    >>> self_attention = SelfAttentionBlock(embed_dim=64, num_heads=8)
    >>> s_i = self_attention(s_i)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SwiGLU(nn.Module):
    """A swish-gated linear unit (SwiGLU)."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        """
        :param input_dim: The dimension of the input tensor.
        :param output_dim: The dimension of the output tensor.
        """
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, output_dim, bias=bias)
        self.linear_2 = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param x: :code:`(..., input_dim)` tensor of embeddings.
        :return: :code:`(..., output_dim)` tensor of updated embeddings.
        """
        a = self.linear_1(x)
        b = self.linear_2(x)
        x = F.silu(a) * b

        return x


class RotaryEncoding(nn.Module):
    """
    Rotary positional embeddings applied to queries and keys in a transformer.

    paper: RoFormer: Enhanced Transformer with Rotary Position Embedding
    link: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        """
        :param head_dim: The per head embedding dimension of the query or key vectors.
        :param theta: Scaling factor for the rotary frequencies.
        """
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                f"Rotary encoding requires an even embedding dimension, "
                f"but received head_dim={head_dim}. Please use an even number."
            )

        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer("freqs", freqs)

    def forward(self, v_i: torch.Tensor) -> torch.Tensor:
        """
        Applies rotary positional encoding to the input vector.
        A 2D case is: (x, y) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]] @ (x, y)

        :param v_i: Input tensor of shape :code:`(batch_size, num_heads, seq_len, head_dim)`, where head_dim should
            match `head_dim` in the constructor.
        :return: Tensor with rotary positional encoding applied, of the same shape as input.
        """
        _, _, seq_len, _ = v_i.shape

        t = torch.arange(seq_len, device=v_i.device, dtype=v_i.dtype).float()
        angles = torch.einsum("i,j->ij", t, self.freqs)
        emb_sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        emb_cos = torch.cos(angles).repeat_interleave(2, dim=-1)

        v_rotated = (v_i * emb_cos) + (self.rotate_half(v_i) * emb_sin)
        return v_rotated

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotates the pairwise elements of the input tensor by 90 degrees.
        (x1, x2) -> (-x2, x1)

        Example:
        x = tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        x = RotaryEncoding.rotate_half(x)
        x = tensor([-1,  0, -3,  2, -5,  4, -7,  6, -9,  8])
        """
        x = rearrange(x, "... (d r) -> ... d r", r=2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, "... d r -> ... (d r)")


class AdaptiveLayerNorm(nn.Module):
    """
    Implements adaptive layer normalisation similar to AlphaFold3 (algorithm 26),
    which performs layer normalisation on the primary tensor conditional on the
    secondary tensor.

    The key difference with the AlphaFold implementation is that the secondary tensor is
    aggregated via a mean over the token dimension before projecting, to allow for different
    numbers of tokens in the primary and secondary tensors.
    """

    def __init__(self, embed_dim: int):
        """
        :param embed_dim: The number of features in the input tensors.
        """
        super().__init__()

        self.ln_no_affine = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.ln_no_bias = nn.LayerNorm(embed_dim, bias=False)

        self.linear_s = nn.Linear(embed_dim, embed_dim)
        self.linear_no_bias_s = nn.Linear(embed_dim, embed_dim, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        :param a: The input tensor to apply layer norm on.
        :param s: A secondary tensor for adaptively modulating the normalization. It is assumed that the
            token dimension of this tensor is the second-to-last dimension.
        :return: The normalized tensor.
        """

        a = self.ln_no_affine(a)
        s = torch.mean(s, dim=-2, keepdim=True)
        s = self.ln_no_bias(s)
        a = self.sigmoid(self.linear_s(s)) * a + self.linear_no_bias_s(s)

        return a


class RobertaHead(nn.Module):
    """A 2-layer MLP with a layer norm."""

    def __init__(self, embed_dim: int, output_dim: int):
        """
        :param embed_dim: The dimension of the input representation.
        :param output_dim: The dimension of the outputs.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, s_i: torch.Tensor) -> torch.Tensor:
        """
        Projects the single representation to the output dimension.

        :param s_i: :code:`(..., embed_dim)` tensor.
        :return: :code:`(..., output_dim)` tensor.
        """
        return self.layers(s_i)


class GatedSelfAttention(nn.Module):
    """
    Gated self-attention layer with a rotary position embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        rotary_theta: float = 10000.0,
    ):
        """
        :param embed_dim: The dimensionality of the input/output embeddings.
        :param num_heads: The number of attention heads to use. The embedding dimension :code:`embed_dim`
            must be divisible by this number.
        :param dropout_p: Dropout probability for the attention weights.
        :param rotary_theta: Theta parameter for the rotary positional encoding.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension {embed_dim} should be divisible by the number of heads {num_heads}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.linear_no_bias_g = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False), nn.Dropout(dropout_p)
        )

        self.sigmoid = nn.Sigmoid()

        head_dim = embed_dim // num_heads
        self.rotary_encoding = RotaryEncoding(head_dim=head_dim, theta=rotary_theta)

    def forward(self, s_i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param s_i: :code:`(batch_size, seq_len, embed_dim)` tensor of embeddings.
        :return: Updated tensor of embeddings, same shape as the input.
        """

        # Input projections
        s_i = self.layer_norm(s_i)

        q_i = rearrange(self.linear_q(s_i), "b i (h d) -> b h i d", h=self.num_heads)
        k_i = rearrange(self.linear_k(s_i), "b i (h d) -> b h i d", h=self.num_heads)
        v_i = rearrange(self.linear_v(s_i), "b i (h d) -> b h i d", h=self.num_heads)
        g_i = rearrange(
            self.sigmoid(self.linear_no_bias_g(s_i)),
            "b i (h d) -> b h i d",
            h=self.num_heads,
        )

        # Rotary positional encoding
        q_i = self.rotary_encoding(q_i)
        k_i = self.rotary_encoding(k_i)

        # Attention
        attn_output = F.scaled_dot_product_attention(q_i, k_i, v_i) * g_i
        attn_output = rearrange(attn_output, "b h i d -> b i (h d)")

        # Output projections
        s_i = self.out(attn_output)
        return s_i


class SwiGLUTransition(nn.Module):
    """
    A transition block for backbones that uses a SwiGLU activation function.
    Implements the transition layer from AlphaFold3 (algorithm 11).
    """

    def __init__(self, embed_dim: int, n: int = 4, dropout_p: float = 0.1):
        """
        :param embed_dim: Dimension of the input embeddings.
        :param n: The "up-scale" factor in the hidden layer dimension (compared to the
            input dimension).
        """
        super().__init__()

        hidden_dim = n * embed_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param x: :code:`(..., embed_dim)` tensor of embeddings.
        :return: `(..., embed_dim)` tensor of updated embeddings.
        """

        return self.layers(x)


class SelfAttentionBlock(nn.Module):
    """A single block that performs self-attention followed by a SwiGLU transition layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
    ):
        """
        :param embed_dim: Dimension of the input embeddings.
        :param num_heads: Number of attention heads.
        :param dropout_p: Dropout probability for the attention weights.
        """
        super().__init__()

        self.attention = GatedSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )
        self.transition = SwiGLUTransition(embed_dim=embed_dim, dropout_p=dropout_p)

    def forward(self, s_i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on a single tensor of embeddings.

        :param s_i: :code:`(batch_size, seq_len, embed_dim)` tensor of embeddings.
        :return: Tensor of updated embeddings, same shape as the input.
        """
        s_i = s_i + self.attention(s_i)
        s_i = s_i + self.transition(s_i)

        return s_i


class ConditionedSwiGLUTransition(nn.Module):
    """
    A conditional SwiGLU-activated transition block. This implements Algorithm 25 from AlphaFold3,
    with the addition of dropout.
    """

    def __init__(
        self,
        embed_dim: int,
        n: int = 2,
        dropout_p: float = 0.1,
        gate_bias_value: float = -2.0,
    ):
        """
        :param embed_dim: Dimension of the input embeddings.
        :param n: The "up-scale" factor in the hidden layer dimension (compared to the
            input dimension).
        :param dropout_p: Dropout probability for the transition layer.
        :param gate_bias_value: The initialisation value for the gate bias. This can be initialised to a
            large negative value to prevent any cross-attention modulation at the start of training.
        """
        super().__init__()

        hidden_dim = n * embed_dim
        self.layer_norm = AdaptiveLayerNorm(embed_dim)
        self.layers = nn.Sequential(
            SwiGLU(embed_dim, hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.linear_gate = nn.Linear(embed_dim, embed_dim)
        self.linear_gate.bias.data.fill_(gate_bias_value)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to transform an input tensor :code:`x` conditional on a secondary tensor :code:`y`.
        This aggregates the secondary tensor via a mean over the token dimension before projecting.

        :param x: Input tensor to be transformed.
        :param y: Tensor containing conditioning information.
        :return: The transformed tensor.
        """
        x = self.layer_norm(x, y)
        out = self.layers(x)
        gate = F.sigmoid(self.linear_gate(torch.mean(y, dim=-2, keepdim=True)))

        return out * gate


class GatedCrossAttention(nn.Module):
    """
    Cross-attention layer for heavy/light chain embedding interactions. This layer uses
    an adaptive layer norm and a gating mechanism to modulate the attention weights obtained
    from the cross attention mechanism.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
    ):
        """
        :param embed_dim: The dimensionality of the input/output embeddings.
        :param num_heads: The number of attention heads to use. The embedding dimension :code:`embed_dim`
            must be divisible by this number.
        :param dropout_p: The dropout probability for the attention weights.
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p

        self.layer_norm_heavy = AdaptiveLayerNorm(embed_dim)
        self.layer_norm_light = AdaptiveLayerNorm(embed_dim)

        self.q_heavy = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_light = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_light = nn.Linear(embed_dim, embed_dim, bias=False)

        self.q_light = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_heavy = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_heavy = nn.Linear(embed_dim, embed_dim, bias=False)

        self.linear_g_heavy = nn.Linear(embed_dim, embed_dim)
        self.linear_g_light = nn.Linear(embed_dim, embed_dim)

        self.sigmoid = nn.Sigmoid()

        self.output_heavy = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout_p)
        )
        self.output_light = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout_p)
        )

    def forward(
        self, heavy_repr: torch.Tensor, light_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on a pair of heavy and light chain embeddings.

        :param heavy_repr: :code:`(batch_size, vh_seq_len, embed_dim)` tensor of heavy chain embeddings.
        :param light_repr: :code:`(batch_size, vl_seq_len, embed_dim)` tensor of light chain embeddings.
        :return: Tuple of updated heavy and light chain embeddings, same shape as the inputs.
        """

        heavy_repr = self.layer_norm_heavy(heavy_repr, light_repr)
        light_repr = self.layer_norm_light(light_repr, heavy_repr)

        # Heavy to Light cross-attention
        q_heavy = rearrange(
            self.q_heavy(heavy_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        k_light = rearrange(
            self.k_light(light_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        v_light = rearrange(
            self.v_light(light_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        g_heavy = rearrange(
            self.sigmoid(self.linear_g_heavy(heavy_repr)),
            "b i (h d) -> b h i d",
            h=self.num_heads,
        )

        attn_output_heavy = (
            F.scaled_dot_product_attention(q_heavy, k_light, v_light) * g_heavy
        )
        attended_heavy = rearrange(attn_output_heavy, "b h i d -> b i (h d)")

        # Light to heavy cross-attention
        q_light = rearrange(
            self.q_light(light_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        k_heavy = rearrange(
            self.k_heavy(heavy_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        v_heavy = rearrange(
            self.v_heavy(heavy_repr), "b i (h d) -> b h i d", h=self.num_heads
        )
        g_light = rearrange(
            self.sigmoid(self.linear_g_light(light_repr)),
            "b i (h d) -> b h i d",
            h=self.num_heads,
        )

        attn_output_light = (
            F.scaled_dot_product_attention(q_light, k_heavy, v_heavy) * g_light
        )
        attended_light = rearrange(attn_output_light, "b h i d -> b i (h d)")

        # Output projections
        updated_heavy = self.output_heavy(attended_heavy)
        updated_light = self.output_light(attended_light)

        return updated_heavy, updated_light


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block that combines cross-attention and transition layers for
    heavy and light chain interaction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        gate_bias_value: float = -2.0,
    ):
        """
        :param embed_dim: The dimensionality of the input/output embeddings.
        :param num_heads: The number of attention heads to use. The embedding dimension :code:`embed_dim`
            must be divisible by this number.
        :param dropout_p: The dropout probability for the attention weights.
        :param gate_bias_value: The initialisation value for the gate bias in the conditioned SwiGLU transition
            blocks. This can be initialised to a large negative value to prevent any cross-attention modulation
            at the start of training.
        """
        super().__init__()

        self.cross_attention = GatedCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout_p=dropout_p,
        )

        self.transition_heavy = ConditionedSwiGLUTransition(
            embed_dim=embed_dim, dropout_p=dropout_p, gate_bias_value=gate_bias_value
        )
        self.transition_light = ConditionedSwiGLUTransition(
            embed_dim=embed_dim, dropout_p=dropout_p, gate_bias_value=gate_bias_value
        )

    def forward(
        self, heavy_repr: torch.Tensor, light_repr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on a pair of heavy and light chain embeddings.

        :param heavy_repr: :code:`(batch_size, vh_seq_len, embed_dim)` tensor of heavy chain embeddings.
        :param light_repr: :code:`(batch_size, vl_seq_len, embed_dim)` tensor of light chain embeddings.
        :return: Tuple of updated heavy and light chain embeddings, same shape as the inputs.
        """

        updated_heavy, updated_light = self.cross_attention(heavy_repr, light_repr)

        heavy_repr = heavy_repr + updated_heavy
        light_repr = light_repr + updated_light

        heavy_repr = heavy_repr + self.transition_heavy(heavy_repr, light_repr)
        light_repr = light_repr + self.transition_light(light_repr, heavy_repr)

        return heavy_repr, light_repr
