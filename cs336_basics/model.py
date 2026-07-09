from __future__ import annotations 

import json
import logging
import math
import os
import warnings

import einx
import torch
import torch.nn as nn
import torch.optim.optimizer as optimizer
from torch.utils.checkpoint import checkpoint
from jaxtyping import Float, Int, Bool
from torch import Tensor
from einops import einsum, rearrange
from cs336_basics.nn_utils import softmax

logger = logging.getLogger(__name__)

class Linear(nn.Module):
    def __init__(
        self, 
        d_in: int, 
        d_out: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None, 
    ):
        """Given the weights of a Linear layer, compute the transformation of a batched input.
        Args:
            in_features (int): The size of the input dimension
            out_features (int): The size of the output dimension
            weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        """
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_out, d_in, device=device, dtype=dtype), mean=0.0, std=std, a=-3 * std, b=3 * std
            ), requires_grad=True
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:   
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"

class Embedding(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ): 
        '''Construct an embedding module.
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (torch.device | None = None): Device to store the parameters on 
            dtype (torch.dtype | None = None): Data type of the parameters
        '''
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(vocab_size, d_model, device=device, dtype=dtype), mean=0.0, std=1.0, a=-3, b=3
            ), requires_grad=True
        )
        

    def forward(
        self, 
        token_ids: Int[Tensor, "..."],
    ) -> Float[Tensor, "... embedding_dim"]:
        '''Lookup the embedding vectors for the given token IDs.'''
        # advanced indexing
        # embedding_matrix: [num_embeddings, embedding_dim]
        # 对于任意形状的输入维度 D，输出维度为 [D, embedding_dim]
        # token_ids 为一堆 int 类型的数值，advanced indexing 会把 token_ids 每个标量 k
        # 替换成 embedding_matrix[k]，长度为 embedding_dim
        return self.weight[token_ids, :]

    def extra_repr(self):
        return f"num_embeddings={self.weight.shape[0]}, embedding_dim={self.weight.shape[1]}"

class RMSNorm(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ):
        r'''Construct the RMSNorm module. Given a vector a (d_model), RMSNorm(a_i) = a_i / RMS(a) * g_i
        RMS(a) = sqrt(1 / d_model * \sum_i^d_model a_i^2 + \epsilon)
        This function should accept the following parameters
        args: 
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability
            device (torch.device | None = None): Device to store the parameters on
            dtype: (torch.dtype | None = None): Data type of the parameters
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self, 
        x: Float[Tensor, "B S d_model"]
    ) -> Float[Tensor, "B S d_model"]:
        '''Process an input tensor of shape(batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.
        '''
        in_dtype = x.dtype
        
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # b s d_model
        out = x * inv_rms * self.weight # (b, s, d_model) * (d_model) -> (b, s, d_model)
        
        return out.to(in_dtype)
        
def _checkpoint_recursive(blocks, x, *args, use_reentrant=False):
    n = len(blocks)
    if n == 1:
        return blocks[0](x, *args)
    
    mid = n // 2
    
    def first_half(inp, *a):
        return _checkpoint_recursive(list(blocks[0:mid]), inp, *a, use_reentrant=use_reentrant)
    
    # 把前半段包进 checkpoint：前向丢掉它内部的中间值，反向时再递归重算
    x = checkpoint(first_half, x, *args, use_reentrant=use_reentrant)
    
    # 后半段继续递归处理
    return _checkpoint_recursive(list(blocks[mid:]), x, *args, use_reentrant=use_reentrant)

class BasicTransformerLM(nn.Module):
    """A Transformer language model.

    Args:
        vocab_size: int
            The number of unique items in the output vocabulary to be predicted.
        context_length: int,
            The maximum number of tokens to process at once.
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_layers: int
            The number of Transformer layers to use.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).

    Returns:
        FloatTensor of shape (batch size, sequence_length, vocab_size) with the
        predicted unnormalized next-word distribution for each token.
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = 10_000.0,
    ):
        # Store the model configuration for serialization / deserialization
        self.config = {
            k: v for k, v in locals().items() if k != "self" and not (k.startswith("__") and k.endswith("__"))
        } 
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = (
            RotaryEmbedding(context_length, d_head, rope_theta) if rope_theta is not None else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    positional_encoder=self.positional_encoder
                )
                for _ in range(num_layers)
            ] 
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        # Tie the weights, since the paper mentions that "we share the same weight
        # matrix between the two embedding layers and the pre-softmax linear transformation"
        # self.lm_head.weight = self.token_embeddings.weight
        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length vocab_size"]:
        """
        Args:
            x: Input IDs for language modeling.

        Returns: A FloatTensor of shape
            (batch size, sequence_length, vocab_size) with the predicted unnormalized next-word
            distribution for each token.
        """
        _, sequence_length = x.size()
        # (batch size, sequence_length, d_model)
        # NOTE: paper mentions "In the embedding layers, we multiply those
        # weights by sqrt(d_model)", but we aren't doing that here.
        embedded_tokens = self.token_embeddings(x)

        # x = self.positional_encoder(embedded_tokens, positions)
        x = embedded_tokens

        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
        
    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            # 只用来截断模型的输入部分
            x_cond = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self(x_cond)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # Apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with a lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill_(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # top_k 和 top_p 只能选一种方法
            if top_p:
                probs = next_token_probabilities[0] # (vocab_size,)
                sorted_next_token_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_next_token_probs, dim=-1)
                
                # 找到第一个 cumulative >= p 的位置，保留到这里
                # nucleus 包括这个位置（"smallest set such that sum >= p"）
                # 求第一个满足的下标，nonzero 返回非 0 的坐标
                # item() 会返回 Number 基类，不能直接用
                cutoff = (cumulative >= top_p).nonzero()[0].item()
                nucleus_probs = sorted_next_token_probs[:cutoff + 1] 
                nucleus_indices = sorted_indices[:cutoff + 1]
                 
                nucleus_probs = nucleus_probs / nucleus_probs.sum()
                sampled_pos = torch.multinomial(nucleus_probs, num_samples=1)
                next_token_id = nucleus_indices[sampled_pos].unsqueeze(0)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:].squeeze(0)
        return new_token_ids
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod, prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
            

# position-wise feed-forward network
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))
        
class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float=10000, device=None):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache", RotaryEmbedding._init_cache(context_length, dim, theta, device), persistent=False
        )
        self._freq_cis_cache: Float[Tensor, "2 context_length half_dim"]

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float, device: str | None) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0
        d = torch.arange(0, dim, 2, device=device) / dim 
        freqs = theta ** -d
        t = torch.arange(context_length, device=device)
        
        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))
        
        
    def forward(
        self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"] | None
    ) -> Float[Tensor, " ... seq d"]:
        # pos_ids 在 MultiHeadSelfAttention 中为多个 heads 做 RoPE，所以维度不定
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)

        # Standard 
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        if pos_ids is not None:
            cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids)
            # cos, sin = self._freq_cis_cache[:, pos_ids, :].unbind(0) 
            
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        # result = einx.id("... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot).contiguous()
        # 参考实现给的是:
        # result = torch.concat((x1_rot, x2_rot), dim=-1)
        result = rearrange([x1_rot, x2_rot], "xy ... half_d -> ... (half_d xy)")
        return result
       
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

class TransformerBlock(nn.Module):
    """A single Transformer layer.

    This implements a single layer of the Transformer, as described in section 3.1
    of the paper.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding | None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """
        # NOTE: this is a pre-norm Transformer, and differs from the original
        # description in the paper.
        # Apply the multi-head self-attention sublayer
        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        # Apply the feed-forward sublayer
        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output

class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    This function implements section 3.2.2 of the Transformer paper. In particular,
    given an input tensor of shape `(batch_size, sequence_length, d_model)`, we project
    it to create queries, keys, and values, and then perform causal multi-headed attention with
    those queries, keys, and values.

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """
    def __init__(
        self, 
        d_model: int, 
        num_heads: int,
        positional_encoder: RotaryEmbedding | None = None
    ):
        super().__init__()
        if positional_encoder is None:
            warnings.warn("No positional encoder provided", stacklevel=2)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder: RotaryEmbedding | None = positional_encoder

    def forward(
        self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_v"]:
        *batch_dims, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads = self.num_heads)
            for X in (Q, K, V)
        )

        if self.positional_encoder is not None: # RoPE is enabled
            if token_positions is not None: # We got explicit position ids
                # Duplicate token positions for each head
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
                
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        iota = torch.arange(sequence_length, device=x.device)
        qi = rearrange(iota, "query -> query 1")
        ki = rearrange(iota, "key   -> 1   key")
        causal_mask = qi >= ki  # (query, key)
        # 等价于: causal_mask = causal_mask[(None,) * len(batch_dims)]
        # for _ in batch_dims:
        #     causal_mask = causal_mask.unsqueeze(0)
        causal_mask = causal_mask.__getitem__((None,) * len(batch_dims))

        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask) 
        
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        
        output = self.output_proj(attn_output)
        return output 
    

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
