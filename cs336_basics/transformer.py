import torch
import torch.nn as nn
import torch.optim.optimizer as optimizer
import math
from jaxtyping import Float, Int, Bool
from torch import Tensor
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None, 
        weights: dict[str, Tensor] | None = None 
    ):
        """Given the weights of a Linear layer, compute the transformation of a batched input.
        Args:
            in_features (int): The size of the input dimension
            out_features (int): The size of the output dimension
            weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        """
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if weights is not None:
            self.weight.data = weights["weight"]

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:   
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        weights: dict[str, Tensor] | None = None 
    ): 
        '''Construct an embedding module.
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors
            device (torch.device | None = None): Device to store the parameters on 
            dtype (torch.dtype | None = None): Data type of the parameters
        '''
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)
        if weights is not None:
            self.weight.data = weights["weight"]

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
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
        weights: dict[str, Tensor] | None = None
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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        if weights is not None:
            self.weight.data = weights["weight"]

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

# position-wise feed-forward network
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weights: dict[str, Tensor] | None = None
    ):
        # canonically, d_ff = 8 / 3 * d_model
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        def make_weight(shape):
            w = torch.empty(*shape, device=device, dtype=dtype)
            d_out, d_in = shape
            std = math.sqrt(2 / (d_in + d_out))
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
            return nn.Parameter(w)

        self.w1 = make_weight((d_ff,    d_model))   # up-proj (gate)
        self.w2 = make_weight((d_model, d_ff   ))   # down-proj
        self.w3 = make_weight((d_ff,    d_model))   # up-proj

        if weights is not None:
            self.w1.data = weights["w1.weight"]
            self.w2.data = weights["w2.weight"]
            self.w3.data = weights["w3.weight"]

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # x: (..., d_model)
        gate = x @ self.w1.T              # (..., d_ff)
        up   = x @ self.w3.T              # (..., d_ff)
        mid  = self.silu(gate) * up       # (..., d_ff)  ← Hadamard, element-wise
        out  = mid @ self.w2.T            # (..., d_model)
        return out
        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float, 
        d_k: int, 
        max_seq_len: int, 
        device: torch.device | None = None
    ):
        '''Construct the RoPE module and create buffers if needed.
        args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None = None): Device to store the buffer on
        '''
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        # slow 
        # angle = torch.tensor([
        #     [i / theta ** (2 * k / d_k) for k in range(d_k // 2)] 
        #     for i in range(max_seq_len)
        # ], device=device)
        k_idx = torch.arange(d_k // 2, device=device, dtype=torch.float32) # (d_k // 2, )
        inv_freq = 1.0 / (theta ** (2 * k_idx / d_k))

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) # (max_seq_len, )

        # 一维张量使用外积构造 angles
        angles = torch.outer(positions, inv_freq) # (max_seq_len, d_k // 2)
        
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
        
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_k"], 
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos = self.cos_cached[token_positions].to(x.dtype)
        sin = self.sin_cached[token_positions].to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_odd * cos + x_even * sin

        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd

        return out

def softmax(x: torch.Tensor, dim: int, temperature: float=1.0) -> torch.Tensor:
    x_norm = (x - x.max(dim=dim, keepdim=True).values) / temperature
    return x_norm.exp() / x_norm.exp().sum(dim=dim, keepdim=True)

def scaled_dot_product_attn(
    q: Float[Tensor, "... queries d_k"], 
    k: Float[Tensor, "... keys d_k"], 
    v: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None
) -> Float[Tensor, "... queries d_v"]:
    # 注意力机制里，K和V来源于同一个序列，长度必然相等
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    
    # 和 None 比较只能用 is / is not，!= 对 Tensor 会做逐元素比较，返回一个 Bool Tensor，而不是单个 bool。
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    out = attn @ v

    return out
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        max_seq_len: int | None = None, 
        theta: float | None = None,
        use_rope: bool = False,
        weights: dict[str, Tensor] | None = None
    ): 
        '''causal multi-head self-attention
        args:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention. 
            max_seq_len (int): Maximum sequence length that will be inputted    
        '''
        # MultiHead(Q, K, V) = Concat(head_1, ... ,head_h)
        # head_i = Attention(Q_i, K_i, V_i)
        # MultiHeadSelfAttention(x) = W_O @ MultiHead(W_Q @ x, W_K @ x, W_V @ x) 
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        
        # d_k = d_v = d_model / h
        # w_q shape = w_k shape: (hd_k, d_model), w_v: (hd_v, d_model), w_o: (d_model, hd_v)
        self.w_qkv = nn.Parameter(torch.empty(3 * d_model, d_model)) 
        self.w_o = nn.Parameter(torch.empty(d_model, d_model)) 
        nn.init.kaiming_normal_(self.w_qkv, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.w_o, a=math.sqrt(5))
        
        mask_size = max_seq_len if max_seq_len else 2048
        self.register_buffer("causal", 
                             torch.tril(torch.ones(mask_size, mask_size, dtype=torch.bool)), persistent=False)
        
        if weights is not None:
            self.w_qkv.data = torch.cat(
                [weights["q_proj.weight"], 
                 weights["k_proj.weight"], 
                 weights["v_proj.weight"]], dim=-2
            )
            self.w_o.data = weights["output_proj.weight"]
            
        d_k = self.d_model // self.num_heads
        if use_rope:
            self.rotary = RotaryPositionalEmbedding(theta, d_k, max_seq_len) 
        
        
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        
        qkv = einsum(x, self.w_qkv, "... s d_in, d_out d_in -> ... s d_out")
        
        qkv = rearrange(qkv, "... s (three h d_k) -> three ... h s d_k", three=3, h=self.num_heads)
        query, key, value = qkv.unbind(0)

        # 测试使用的 token_positions 和这里的是一样的
        # token_positions: Int[Tensor, "... seq_len"]
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        
        key_rotary = self.rotary(key, token_positions) if self.use_rope else key
        query_rotary = self.rotary(query, token_positions) if self.use_rope else query
        
        # (..., S, d_model)
        S = x.shape[-2]
        mask = self.causal[:S, :S]
        
        result = scaled_dot_product_attn(query_rotary, key_rotary, value, mask)
        attn = rearrange(result, "... h s d_v -> ... s (h d_v)")
        
        # d_model 容易让人想成"模型隐藏维"，但在这里它其实是输入维度（h * d_v），用 d_in / d_out 来表达"投影前/后"更准确。
        out = einsum(attn, self.w_o, "... s d_in, d_out d_in -> ... s d_out")
        return out 

        
        # ----------------下面这些部分是历史遗留痕迹----------------
        
        # query, key, value = qkv.split(self.d_model, dim=-1) # (..., seq_len, d_model)
        # query = rearrange(query, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        # key = rearrange(key, "... s (h d_k) -> ... h s d_k", h=self.num_heads)
        # value = rearrange(value, "... s (h d_v) -> ... h s d_v", h=self.num_heads)
        
        # x: Float[Tensor, "... seq_len d_k"], 
        
        # kqv = x @ self.w_qkv.transpose(-2, -1) # (..., seq_len, 3 * d_model)
        # key shape: (..., S, d_model) -> (..., num_heads, S, d_k)
        # *batch_dims, S, d_model = x.shape 
        # d_k = self.d_model // self.num_heads
        # key = key.view(*batch_dims, S, self.num_heads, d_k).transpose(-3, -2)
        # query = query.view(*batch_dims, S, self.num_heads, d_k).transpose(-3, -2)
        
        # value shape: (..., S, d_model) -> (..., num_heads, S, d_v)
        # value = value.view(*batch_dims, S, self.num_heads, d_k).transpose(-3, -2)
        
        # attn shape: (..., S, d_model)
        # attn = scaled_dot_product_attn(query_rotary, key_rotary, value, mask).transpose(-3, -2).reshape(*batch_dims, S, d_model)
        
        # w_o shape: (... d_model, hd_v)
        # d_model == hd_v
        # 因为前面转置了 w_qkv，按照约定这里也需要转置 w_o
        # out = attn @ self.w_o.transpose(-2, -1)
        

def filter_prefix(weights: dict[str, Tensor] | None, prefix: str) -> dict[str, Tensor] | None:
    if weights is None: 
        return None 
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        weights: dict[str, Tensor] | None,

    ):
        '''pre-norm Transformer block
        args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
        '''
        # A pre-norm transformer_block (x shape (batch_size, seq_len, d_model)):
        # y = x + MultiHeadSelfAttention(RMSNorm(x)) 
        # out = y + position-wise feed-forward(RMSNorm(y))
        # out shape: (batch_size, seq_len, d_model)
        
        super().__init__()
        self.ln1 = RMSNorm(d_model, weights=filter_prefix(weights, "ln1."))
        self.ln2 = RMSNorm(d_model, weights=filter_prefix(weights, "ln2."))
        self.attn = \
        MultiHeadSelfAttention(
            d_model, 
            num_heads, 
            max_seq_len, 
            rope_theta, 
            use_rope=True, 
            weights=filter_prefix(weights, "attn.")
        )
        self.ffn = SwiGLU(d_model, d_ff, weights=filter_prefix(weights, "ffn."))
    
    def forward(
        self, 
        x: Float[Tensor, "B S d_model"], 
        token_positions: Int[Tensor, " ... seq_len"] | None = None
    ) -> Float[Tensor, "B S d_model"]:
        y = x + self.attn(self.ln1(x), token_positions=token_positions)
        out = y + self.ffn(self.ln2(y))
        return out 

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor] | None = None,
    ):
        '''Transformer Language model
        args:
            vocab_size (int): The size of the vocabulary, necessary for determining 
            the dimensionality of the token embedding matrix.
            context_length (int): The maximum context length, necessary for determining 
            the dimensionality ofthe position embedding matrix.
            num_layers(int): The number of Transformer blocks to use.
        '''
        super().__init__()
        self.embedding = Embedding(
            vocab_size, d_model, 
            weights=filter_prefix(weights, "token_embeddings.")
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, context_length, rope_theta, 
                weights=filter_prefix(weights, f"layers.{i}."),
            ) 
            for i in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, weights=filter_prefix(weights, "ln_final."))
        self.lm_head = Linear(d_model, vocab_size, weights=filter_prefix(weights, "lm_head."))

    def forward(
        self, 
        token_ids: Int[Tensor, "B S"],
        token_positions: Int[Tensor, " ... S"] | None = None
    ) -> Float[Tensor, "B S vocab_size"]:
        x = self.embedding(token_ids)
        for block in self.layers:
            x = block(x, token_positions)
            
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
        
def cross_entropy(
        logits: Float[Tensor, "... vocab_size"], 
        targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    
    # 减去最大元素保证 exp 不会溢出
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    log_sum_exp = shifted.exp().sum(dim=-1).log()
    
    # advanced indexing 索引的是最外层维度
    # gather 用来按索引在某一维取值，注意要求维度相同
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    loss = log_sum_exp - (target_logit - m.squeeze(-1))

    return loss.mean()
        
         
        

        
        
    
        
                
