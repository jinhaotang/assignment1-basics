from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
      super().__init__()
      self.eps = eps
      self.g = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
      torch.nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      in_dtype = x.dtype
      x = x.to(torch.float32)

      mean_square = x.pow(2).mean(dim=-1, keepdim=True)

      # 3. Add epsilon (1e-5) and take the square root [1]
      rms_a = torch.sqrt(mean_square + self.eps)
      result = x / rms_a * self.g
      return result.to(in_dtype)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(in_features=d_in, out_features=d_out, device=weights.device, dtype=weights.dtype)
    linear.load_state_dict({"weight": weights})
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embedding = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    embedding.load_state_dict({"weights": weights})
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    w1_x = torch.einsum("...i,oi->...o", in_features, w1_weight)
    w3_x = torch.einsum("...i,oi->...o", in_features, w3_weight)
    glu = w1_x * torch.sigmoid(w1_x) * w3_x
    return torch.einsum("...i,oi->...o", glu, w2_weight)
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... key s d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    attn_scores = torch.einsum("...qd,...kd->...qk", Q, K) / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32))
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == False, float('-inf'))
    attn_weights = run_softmax(attn_scores, dim=-1)
    return torch.einsum("...qk,...kv->...qv", attn_weights, V)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    seq_len = in_features.shape[-2]
    d_k = d_model // num_heads
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

    q = torch.einsum("...i,oi->...o", in_features, q_proj_weight)
    k = torch.einsum("...i,oi->...o", in_features, k_proj_weight)
    v = torch.einsum("...i,oi->...o", in_features, v_proj_weight)

    # split into heads: (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
    q = q.unflatten(-1, (num_heads, d_k)).transpose(-2, -3)
    k = k.unflatten(-1, (num_heads, d_k)).transpose(-2, -3)
    v = v.unflatten(-1, (num_heads, d_k)).transpose(-2, -3)

    attention = run_scaled_dot_product_attention(q, k, v, mask=causal_mask)
    # merge heads: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
    attention = attention.transpose(-2, -3).flatten(-2)
    return torch.einsum("...i,oi->...o", attention, o_proj_weight)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    seq_len = in_features.shape[-2]
    d_k = d_model // num_heads
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

    q = torch.einsum("...i,oi->...o", in_features, q_proj_weight)
    k = torch.einsum("...i,oi->...o", in_features, k_proj_weight)
    v = torch.einsum("...i,oi->...o", in_features, v_proj_weight)

    # split into heads: (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k)
    q = q.unflatten(-1, (num_heads, d_k)).transpose(-2, -3)
    k = k.unflatten(-1, (num_heads, d_k)).transpose(-2, -3)
    v = v.unflatten(-1, (num_heads, d_k)).transpose(-2, -3)
    q = run_rope(d_k, theta, max_seq_len, q, token_positions.unsqueeze(-2))
    k = run_rope(d_k, theta, max_seq_len, k, token_positions.unsqueeze(-2))

    attention = run_scaled_dot_product_attention(q, k, v, mask=causal_mask)
    # merge heads: (..., num_heads, seq_len, d_k) -> (..., seq_len, d_model)
    attention = attention.transpose(-2, -3).flatten(-2)
    return torch.einsum("...i,oi->...o", attention, o_proj_weight)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        j = torch.arange(d_k // 2, device=device)
        freqs = 1.0 / (theta ** (2 * j / d_k))  # (d_k/2,)

        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        angles = torch.outer(positions, freqs)  # (max_seq_len, d_k/2)

        cos = torch.cos(angles)  # (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # (max_seq_len, d_k/2)

        # Build block-diagonal rotation matrices: (max_seq_len, d_k, d_k)
        R = torch.zeros(max_seq_len, d_k, d_k, device=device)
        R[:, 2 * j, 2 * j] = cos
        R[:, 2 * j, 2 * j + 1] = -sin
        R[:, 2 * j + 1, 2 * j] = sin
        R[:, 2 * j + 1, 2 * j + 1] = cos

        self.register_buffer("R", R)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Ri = self.R[token_positions]  # (..., seq_len, d_in, d_out)
        # x: (..., seq_len, d_in)
        return torch.einsum("...i,...ji->...j", x, Ri)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
#     run_multihead_self_attention_with_rope(
#     d_model: int,
#     num_heads: int,
#     max_seq_len: int,
#     theta: float,
#     q_proj_weight: Float[Tensor, " d_model d_model"],
#     k_proj_weight: Float[Tensor, " d_model d_model"],
#     v_proj_weight: Float[Tensor, " d_model d_model"],
#     o_proj_weight: Float[Tensor, " d_model d_model"],
#     in_features: Float[Tensor, " ... sequence_length d_model"],
#     token_positions: Int[Tensor, " ... sequence_length"] | None = None,
# ) -> Float[Tensor, " ... sequence_length d_model"]:
    seq_len = in_features.shape[-2]
    batch_size = in_features.shape[0]
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)

    norm_in_features = run_rmsnorm(d_model, 1e-5, weights['ln1.weight'], in_features)

    attention = run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, weights['attn.q_proj.weight'], weights['attn.k_proj.weight'], weights['attn.v_proj.weight'], weights['attn.output_proj.weight'], norm_in_features, token_positions=token_positions)
    attention = in_features + attention

    norm_attention = run_rmsnorm(d_model, 1e-5, weights['ln2.weight'], attention)
    ffn = run_swiglu(d_model, d_ff, weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'], norm_attention)
    ffn = attention + ffn
    return ffn


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

#     run_embedding(
#     vocab_size: int,
#     d_model: int,
#     weights: Float[Tensor, " vocab_size d_model"],
#     token_ids: Int[Tensor, " ..."],
# ) -> Float[Tensor, " ... d_model"]:

    embedding_output = run_embedding(vocab_size, d_model, weights['token_embeddings.weight'], in_indices)
    # output batch_size, sequence_length, d_model
    for i in range(num_layers):
        embedding_output = run_transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, {k.replace(f'layers.{i}.', ''): v for k, v in weights.items() if k.startswith(f'layers.{i}.')}, embedding_output)
    # output batch_size, sequence_length, d_model
    embedding_output = run_rmsnorm(d_model, 1e-5, weights['ln_final.weight'], embedding_output)
    # output batch_size, sequence_length, d_model
    return run_linear(d_model, vocab_size, weights['lm_head.weight'], embedding_output)
    

#         run_transformer_block(
#     d_model: int,
#     num_heads: int,
#     d_ff: int,
#     max_seq_len: int,
#     theta: float,
#     weights: dict[str, Tensor],
#     in_features: Float[Tensor, " batch sequence_length d_model"],
# ) -> Float[Tensor, " batch sequence_length d_model"]:

    # in_indices
    # x = in_features
    # for i in range(num_layers):
    #     x = transformer_block(x)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_norm = RMSNorm(d_model, eps, device=weights.device, dtype=weights.dtype)
    rms_norm.load_state_dict({"g": weights})
    return rms_norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    inputs, labels = [], []
    for _ in range(batch_size):
        i = torch.randint(0, len(dataset) - context_length, (1,)).item()
        inputs.append(torch.tensor(dataset[i:i + context_length], dtype=torch.long))
        labels.append(torch.tensor(dataset[i + 1:i + context_length + 1], dtype=torch.long))
    return torch.stack(inputs).to(device), torch.stack(labels).to(device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_val = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - max_val
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # input is the probablitiy of each vocab word, target is the index of the correct vocab word
    max_val = inputs.max(dim=-1, keepdim=True).values
    inputs = inputs - max_val

    # exp = torch.exp(inputs[targets]) # ... batch_size
    exp_sum = torch.sum(torch.exp(inputs), dim=-1) # ... batch_size
    # log = -inputs[targets] + torch.log(exp_sum) # ... batch_size
    correct_logits = inputs[torch.arange(inputs.size(0)), targets]  # (batch_size,)
    loss = -correct_logits + torch.log(exp_sum)                     # (batch_size,)
    return loss.mean()
    # return 
    # raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total_norm >= max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # initialize state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0          # step count
                    state["m"] = torch.zeros_like(p.data)  # 1st moment
                    state["v"] = torch.zeros_like(p.data)  # 2nd moment

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]
                lr_adjust = lr * torch.sqrt(torch.tensor(1 - beta2 ** t)) / (1 - beta1 ** t)

                p.data = p.data - lr * weight_decay * p.data
                state["m"] = beta1 * m + (1 - beta1) * grad
                state["v"] = beta2 * v + (1 - beta2) * grad * grad
                p.data = p.data - lr_adjust * state["m"] / (torch.sqrt(state["v"]) + eps)
        return loss


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    elif it <= cosine_cycle_iters:
        t = it - warmup_iters
        T_c = cosine_cycle_iters - warmup_iters
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * t / T_c))
    else:
        return min_learning_rate


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        import regex
        self._regex = regex

        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Append any special tokens not already in vocab
        existing_bytes = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_bytes:
                self.vocab[len(self.vocab)] = token_bytes
                existing_bytes.add(token_bytes)

        # Reverse vocab for encoding: bytes -> id
        self._bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # Merge lookup: (left, right) -> merged, in priority order
        self._merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # GPT-2 pre-tokenization pattern
        self._GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Build special-token split pattern (longest first to avoid partial matches)
        sorted_special = sorted(self.special_tokens, key=len, reverse=True)
        import re
        self._special_pat = (
            re.compile("(" + "|".join(re.escape(t) for t in sorted_special) + ")")
            if sorted_special else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        import json
        with open(vocab_filepath, encoding="utf-8") as f:
            raw = json.load(f)
        vocab = {int(k): v.encode("latin-1") if isinstance(v, str) else v for k, v in raw.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                left, right = line.split(" ", 1)
                merges.append((left.encode("utf-8"), right.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, chars: list[bytes]) -> list[bytes]:
        import heapq
        n = len(chars)
        if n < 2:
            return chars

        # Represent sequence as a doubly-linked list over index arrays so
        # merges are O(1) instead of O(n) list rebuilds.
        tokens = list(chars)
        prev = list(range(-1, n - 1))   # prev[i]: previous active index (-1 = none)
        nxt  = list(range(1, n + 1))    # nxt[i]:  next active index (n = end)

        # Min-heap entries: (rank, left_idx, left_tok, right_tok).
        # Storing the token bytes at insertion time lets us detect stale entries
        # cheaply on pop — no separate "valid" set needed.
        heap: list = []
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = self._merge_rank.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, i, tokens[i], tokens[i + 1]))

        while heap:
            rank, i, tok_i, tok_j = heapq.heappop(heap)
            j = nxt[i]
            # Stale if either neighbour has been overwritten by a prior merge.
            if j >= n or tokens[i] != tok_i or tokens[j] != tok_j:
                continue
            merged = tok_i + tok_j
            tokens[i] = merged
            # Splice j out of the linked list and tombstone it so any stale
            # heap entries that reference j as a left position fail the check.
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            tokens[j] = None
            # Push new left pair if it has a merge rule.
            if prev[i] >= 0:
                pair = (tokens[prev[i]], merged)
                r = self._merge_rank.get(pair)
                if r is not None:
                    heapq.heappush(heap, (r, prev[i], tokens[prev[i]], merged))
            # Push new right pair if it has a merge rule.
            if nxt[i] < n:
                pair = (merged, tokens[nxt[i]])
                r = self._merge_rank.get(pair)
                if r is not None:
                    heapq.heappush(heap, (r, i, merged, tokens[nxt[i]]))

        # Collect surviving tokens in linked-list order.
        result, i = [], 0
        while i < n:
            result.append(tokens[i])
            i = nxt[i]
        return result

    def _encode_chunk(self, text: str) -> list[int]:
        tokens = self._regex.findall(self._GPT2_PAT, text)
        ids = []
        for token in tokens:
            chars = [bytes([b]) for b in token.encode("utf-8")]
            merged = self._apply_merges(chars)
            ids.extend(self._bytes_to_id[chunk] for chunk in merged)
        return ids

    def encode(self, text: str) -> list[int]:
        if self._special_pat is None:
            return self._encode_chunk(text)

        ids = []
        for part in self._special_pat.split(text):
            if not part:
                continue
            part_bytes = part.encode("utf-8")
            if part_bytes in self._bytes_to_id:
                ids.append(self._bytes_to_id[part_bytes])
            else:
                ids.extend(self._encode_chunk(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]):
        for text in iterable:
            yield from self.encode(text)

    def encode_file(
        self,
        path: str | os.PathLike,
        chunk_size: int = 1 << 20,
    ):
        """Encode a large file with O(chunk_size) memory.

        Splits chunks so that ``to_process`` never ends with whitespace.
        This is required because the GPT-2 regex pattern ``\\s+(?!\\S)``
        matches a whitespace run differently depending on whether it is
        followed by a non-whitespace character or by end-of-string, so a
        trailing whitespace in a chunk would produce wrong tokens.

        Yields token ids one by one.
        """
        import time
        file_size = os.path.getsize(path)
        bytes_done = 0
        tokens_done = 0
        t_start = time.time()
        t_last = t_start

        with open(path, encoding="utf-8") as f:
            leftover = ""
            while True:
                raw = f.read(chunk_size)
                if not raw:
                    break
                bytes_done += len(raw.encode("utf-8"))
                text = leftover + raw
                # Find the last word boundary: last position i where text[i-1]
                # is non-whitespace and text[i] is whitespace.  Splitting here
                # ensures to_process ends on a complete word (no mid-word cut)
                # AND the leftover starts with whitespace so the GPT-2 regex
                # can correctly compute \s+(?!\S) vs \s+ with full context.
                split = -1
                for i in range(len(text) - 1, 0, -1):
                    if text[i].isspace() and not text[i - 1].isspace():
                        split = i
                        break
                if split == -1:
                    to_process, leftover = text, ""
                else:
                    to_process, leftover = text[:split], text[split:]
                if to_process:
                    chunk_ids = list(self.encode(to_process))
                    tokens_done += len(chunk_ids)
                    yield from chunk_ids

                now = time.time()
                if now - t_last >= 10.0:
                    pct = bytes_done / file_size * 100 if file_size else 0
                    elapsed = now - t_start
                    eta = (elapsed / pct * (100 - pct)) if pct > 0 else 0
                    print(
                        f"  [{pct:5.1f}%] {bytes_done/1e6:.1f}/{file_size/1e6:.1f} MB  "
                        f"{tokens_done:,} tokens  "
                        f"elapsed {elapsed:.0f}s  eta {eta:.0f}s",
                        flush=True,
                    )
                    t_last = now

            if leftover:
                chunk_ids = list(self.encode(leftover))
                tokens_done += len(chunk_ids)
                yield from chunk_ids

        elapsed = time.time() - t_start
        print(f"  [100.0%] done — {tokens_done:,} tokens in {elapsed:.1f}s", flush=True)

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)


def _merge_pair(token_counts: dict[str, list], best_pair: tuple[bytes, bytes]) -> None:
    merged = best_pair[0] + best_pair[1]
    for token in token_counts:
        chars = token_counts[token][1]
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                new_chars.append(merged)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        token_counts[token][1] = new_chars


_GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _pretokenize_chunk(args):
    import re
    import regex
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Remove special tokens before applying GPT-2 regex
    if special_tokens:
        split_pat = "|".join(re.escape(t) for t in special_tokens)
        parts = re.split(split_pat, chunk)
    else:
        parts = [chunk]
    tokens = []
    for part in parts:
        tokens.extend(regex.findall(_GPT2_PAT, part))
    return tokens


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import multiprocessing
    import re
    import time
    from cs336_basics.pretokenization_example import find_chunk_boundaries

    t_start = time.time()
    print(f"[BPE] Started at {time.strftime('%H:%M:%S')}")

    num_processes = multiprocessing.cpu_count()
    print(f"[BPE] Pre-tokenizing with {num_processes} processes...")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with multiprocessing.Pool(num_processes) as pool:
        results = pool.map(_pretokenize_chunk, chunk_args)

    tokens = [tok for chunk_tokens in results for tok in chunk_tokens]
    print(f"Pre-tokenization done: {len(tokens):,} tokens, {len(set(tokens)):,} unique.")

    # Count occurrences of each unique token; values are [count, list_of_byte_chunks]
    # Each char in the list is a bytes object (initially one byte each)
    token_counts: dict[str, list] = {}
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = [0, [bytes([b]) for b in token.encode("utf-8")]]
        token_counts[token][0] += 1

    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    print(f"Training BPE: {num_merges} merges on {len(token_counts):,} unique tokens from {len(tokens):,} total tokens")

    # Build pair_counts and reverse index pair_to_tokens once
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_tokens: dict[tuple[bytes, bytes], set] = {}
    for token, (count, chars) in token_counts.items():
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(token)

    for merge_idx in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]
        merged = best_pair[0] + best_pair[1]

        # Only update tokens that contain best_pair
        for token in list(pair_to_tokens.get(best_pair, set())):
            count, chars = token_counts[token]
            # Compute new chars after merge
            new_chars = []
            i = 0
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == best_pair[0] and chars[i + 1] == best_pair[1]:
                    new_chars.append(merged)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            # Compute pair-count diffs between old and new chars
            old_pairs: dict[tuple, int] = {}
            for i in range(len(chars) - 1):
                p = (chars[i], chars[i + 1])
                old_pairs[p] = old_pairs.get(p, 0) + 1
            new_pairs: dict[tuple, int] = {}
            for i in range(len(new_chars) - 1):
                p = (new_chars[i], new_chars[i + 1])
                new_pairs[p] = new_pairs.get(p, 0) + 1
            # Apply diffs to pair_counts and pair_to_tokens
            for p, freq in old_pairs.items():
                pair_counts[p] = pair_counts.get(p, 0) - count * freq
                if pair_counts.get(p, 0) <= 0:
                    pair_counts.pop(p, None)
            for p, freq in new_pairs.items():
                pair_counts[p] = pair_counts.get(p, 0) + count * freq
            for p in set(old_pairs) - set(new_pairs):
                pair_to_tokens.get(p, set()).discard(token)
            for p in set(new_pairs) - set(old_pairs):
                pair_to_tokens.setdefault(p, set()).add(token)
            token_counts[token][1] = new_chars

        pair_to_tokens.pop(best_pair, None)

        merges.append(best_pair)
        if (merge_idx + 1) % 100 == 0 or merge_idx == 0:
            print(f"  merge {merge_idx+1:4d}/{num_merges} | best pair: {best_pair[0]!r} + {best_pair[1]!r} -> {merged!r} (count={best_count})")

    # Build vocab: start with 256 byte tokens, then special tokens, then merged tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for pair in merges:
        vocab[len(vocab)] = pair[0] + pair[1]

    elapsed = time.time() - t_start
    print(f"[BPE] Finished at {time.strftime('%H:%M:%S')} (took {elapsed/60:.1f} min)")
    return vocab, merges
