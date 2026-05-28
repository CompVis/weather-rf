"""
Transformer implementation with local modifications.
© 2026 Johannes Schusterbauer, licensed under MIT.
"""


import os
import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from pydoc import locate
from jaxtyping import Float
from functools import reduce
from einops import rearrange
from functools import partial
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention



__all__ = [
    "zero_init", "LinearSwiGLU",
    "rms_norm", "RMSNorm", "AdaRMSNorm",
    "TokenMerge2D", "TokenSplitLast2D", "TokenMerge3D", "TokenSplitLast3D",
    "CondTokenMerge3D", "TokenSplit3D",
    "AxialRoPE1D", "AxialRoPE2D",
    "FourierFeatures", "TimestepEmbedder",
    "MappingFeedForwardBlock", "MappingNetwork", "FeedForwardBlock",
    "scale_for_cosine_sim", "AttentionBlock", "FactorizedAttentionBlock", "LegacyFactorizedAttentionBlock",
    "DimensionAttentionBlock",
    "CrossAttentionBlock", "TransformerLayer", "FactorizedAttentionLayer",
    "RegisterAttentionBlock", "RegisterCrossAttentionBlock", "TransformerLayerWithRegisters",
    "make_axial_pos_3d", "make_axial_pos_2d",
]
# ===================================================================================================


COMPILE = True
if torch.cuda.is_available():
    compile_fn = partial(torch.compile, fullgraph=False, backend='inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager')
else:
    compile_fn = lambda f: f

if COMPILE:
    flex_attention = compile_fn(flex_attention)

# ===================================================================================================

def instantiate_optional_class(class_ref, *args, **kwargs):
    """Instantiate class from dotted path or callable; return None if unavailable."""
    if class_ref is None:
        return None
    cls = locate(class_ref) if isinstance(class_ref, str) else class_ref
    if cls is None:
        print(
            f"[WARN] Could not locate optional class '{class_ref}'. "
            "Disabling corresponding positional embedding.",
            flush=True,
        )
        return None
    return cls(*args, **kwargs)


class AbstractPosEnc(nn.Module, ABC):
    def __init__(self, d_head, n_heads):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

    @abstractmethod
    def forward(self, pos):
        pass

    @abstractmethod
    def apply_emb(self, x, theta):
        pass


class AxialRoPEBase(AbstractPosEnc):
    def __init__(self, d_head, n_heads, in_place=False):
        super().__init__(d_head, n_heads)
        self.in_place = in_place

    def apply_emb(self, x, theta):
        if self.in_place:
            return apply_rotary_emb_(x, theta)
        return apply_rotary_emb(x, theta)

    @abstractmethod
    def forward(self, pos):
        pass


def apply_rotary_emb(x, theta, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)


class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        (theta,) = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)


class AxialRoPE2D(AxialRoPEBase):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        learnable_freqs: bool = False,
        relative_canvas: bool = True,
        in_place: bool = False,
        half_embedding: bool = True,
    ):
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        super().__init__(dim, n_heads, in_place=in_place)
        if learnable_freqs:
            assert not in_place, "In-place RoPE with learnable frequencies is not supported"
        self.learnable_freqs = learnable_freqs

        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.stack([torch.linspace(log_min, log_max, n_heads * dim // (2 * 2) + 1)[:-1].exp()] * 2)
        self.freqs = nn.Parameter(freqs.view(2, dim // (2 * 2), n_heads).mT.contiguous(), requires_grad=learnable_freqs)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-1] * (2 * 2)}, n_heads={self.freqs.shape[-2]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class AxialRoPE1D(AxialRoPEBase):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        learnable_freqs: bool = False,
        relative_canvas: bool = True,
        in_place: bool = False,
        half_embedding: bool = True,
    ):
        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        super().__init__(dim, n_heads, in_place=in_place)

        if learnable_freqs:
            assert not in_place, "In-place RoPE with learnable frequencies is not supported"

        self.learnable_freqs = learnable_freqs

        if relative_canvas:
            min_freq = math.pi
            max_freq = 10.0 * math.pi
        else:
            min_freq = 1 / 10_000
            max_freq = 1.0

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 2 + 1)[:-1].exp()
        self.freqs = nn.Parameter(freqs.view(dim // 2, n_heads).T.contiguous(), requires_grad=learnable_freqs)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-2] * 2}, n_heads={self.freqs.shape[-1]}"

    def forward(self, pos):
        return pos[..., None] * self.freqs.to(pos.dtype)


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2

def bounding_box(h, w, pixel_aspect_ratio=1.0):
    # Adjusted dimensions
    w_adj = w
    h_adj = h * pixel_aspect_ratio

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max

def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)

def make_axial_pos_2d(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2

    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)

def make_grid_3d(t_pos, h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(t_pos, h_pos, w_pos, indexing='ij'), dim=-1)
    t, h, w, d = grid.shape
    return grid.view(t * h * w, d)

def make_axial_pos_3d(t, h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None, relative_pos=True):
    if relative_pos:
        y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    else:
        y_min, y_max, x_min, x_max = -h / 2, h / 2, -w / 2, w / 2
    
    if align_corners:
        t_pos = torch.arange(t, dtype=dtype, device=device).float().to(dtype)
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        t_pos = torch.arange(t, dtype=dtype, device=device).float().to(dtype)
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid_3d(t_pos, h_pos, w_pos)

# ===================================================================================================

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        x = x @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        proj_out = self.linear(cond)
        if proj_out.ndim == 2:          # (bs, dim) -> (bs, 1, dim)
            proj_out = proj_out[:, None, :]
        else:
            assert proj_out.shape[1] == x.shape[1] or proj_out.shape[1] == 1, \
                f"mismatch in AdaRMSNorm shape: x={x.shape} proj_out={cond.shape}"
        return rms_norm(x, proj_out + 1, self.eps)


# ===================================================================================================
# patchification


class TokenMerge2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
        self.ph = patch_size[0]
        self.pw = patch_size[1]
        self.proj = nn.Linear(in_features * self.ph * self.pw, out_features, bias=False)

    def forward(self, x, pos):
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.ph, nw=self.pw)
        pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=self.ph, nw=self.pw)
        return self.proj(x), torch.mean(pos, dim=-2)


class TokenSplitLast2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2), zero_init: bool = True):
        super().__init__()
        if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
        self.ph = patch_size[0]
        self.pw = patch_size[1]
        self.norm = RMSNorm(in_features)
        self.proj = nn.Linear(in_features, out_features * self.ph * self.pw, bias=False)
        if zero_init: nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... h w (ph pw c) -> ... (h ph) (w pw) c", ph=self.ph, pw=self.pw)
        return x


class TokenMerge3D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w, out_features, bias=False)

    def forward(self, x, pos, **kwargs):
        x = rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        pos = rearrange(pos, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw) e", nt=self.t, nh=self.h, nw=self.w)

        return self.proj(x), torch.mean(pos, dim=-2)

class CondTokenMerge3D(nn.Module):
    def __init__(self, in_features, out_features, cond_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w + cond_features, out_features, bias=False)

    def forward(self, x, pos, cond_tokens, **kwargs):
        # print(f"[DEBUG] in CondTokenMerge3D: x={x.shape} cond_tokens={cond_tokens.shape}", flush=True)
        x = rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        pos = rearrange(pos, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw) e", nt=self.t, nh=self.h, nw=self.w)

        # print(f"[DEBUG] x before concat: {x.shape}", flush=True)
        x = torch.cat([x, cond_tokens], dim=-1)
        # print(f"[DEBUG] x after concat: {x.shape}", flush=True)
        return self.proj(x), torch.mean(pos, dim=-2)

class TokenSplit3D(nn.Module):

    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, pos, skip=None, **kwargs):
        x = self.proj(x)
        x = rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
        if skip is not None:
            skip = rearrange(skip, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
            x = x * self.fac + skip * (1 - self.fac)
        return x, pos

class TokenSplitLast3D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.norm = RMSNorm(in_features)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.proj(x)
        x = rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
        return x


# ===================================================================================================
# fourier/time embedding


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, depth: int, dim_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = dim
        self.depth = depth
        self.dim_mlp = dim_mlp
        self.dropout = dropout

        self.time_emb = FourierFeatures(1, dim)
        self.time_in_proj = nn.Linear(dim, dim, bias=False)
        self.mapping = MappingNetwork(depth, dim, dim_mlp, dropout=dropout)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, t: Float[torch.Tensor, 'b']) -> Float[torch.Tensor, 'b dim']:
        if t.ndim == 1:
            t = t[..., None]
        time_emb = self.time_in_proj(self.time_emb(t))
        time_emb = self.mapping(time_emb)
        return time_emb


# ===================================================================================================


class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm=None, dropout=0.0):
        super().__init__()
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond_norm=None):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


# ===================================================================================================
# Attention


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        rope_cls=AxialRoPE2D,
    ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.n_heads = d_model // d_head
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

        self.pos_emb = instantiate_optional_class(
            rope_cls,
            d_head,
            self.n_heads,
            relative_canvas=True,
            learnable_freqs=False,
        )

    def forward(self, x, pos, cond_norm=None, block_mask=None):
        skip = x

        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        if self.pos_emb is not None:
            pos = pos.to(qkv.dtype)
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)

        if block_mask is None:
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            x = flex_attention(q, k, v, scale=1.0, block_mask=block_mask)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip
    

class FactorizedAttentionBlock(nn.Module):
    """
    Factorized (spatial + temporal) attention over a 5D tensor (B, T, H, W, C).
    Applies cond-norm once, computes spatial and temporal attentions independently,
    concatenates both outputs, then applies a single dropout + out projection + skip.
    """
    def __init__(
        self,
        d_model: int,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        rope_cls_spatial=AxialRoPE2D,
        rope_cls_temporal=AxialRoPE1D,
    ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.n_heads = d_model // d_head
        assert self.n_heads * d_head == d_model, f"{d_model=} must be divisible by {d_head=}"

        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)

        self.spatial_qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.temporal_qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale_spatial = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.scale_temporal = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model * 2, d_model, bias=False))

        self.spatial_pos_emb = instantiate_optional_class(
            rope_cls_spatial,
            d_head,
            self.n_heads,
            relative_canvas=True,
            learnable_freqs=False,
        )

        self.temporal_pos_emb = instantiate_optional_class(
            rope_cls_temporal,
            d_head,
            self.n_heads,
            relative_canvas=True,
            learnable_freqs=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b t h w c"],
        pos: Float[torch.Tensor, "b t h w d"],
        cond_norm: Float[torch.Tensor, "b e"] | None = None,
        spatial_block_mask=None,
        temporal_block_mask=None,
        enable_temporal_attention: bool = True,
    ) -> Float[torch.Tensor, "b t h w c"]:
        skip = x
        B, T, H, W, C = x.shape

        x_ = rearrange(x, "b t h w c -> b (t h w) c")
        if cond_norm is not None:
            x_ = self.norm(x_, cond_norm)
        else:
            x_ = self.norm(x_)
        x_ = rearrange(x_, "b (t h w) c -> b t h w c", t=T, h=H, w=W)

        x_spatial = rearrange(x_, "b t h w c -> (b t) (h w) c")
        qkv_spatial = self.spatial_qkv_proj(x_spatial)
        q_spatial, k_spatial, v_spatial = rearrange(
            qkv_spatial, "n l (three nh e) -> three n nh l e", three=3, e=self.d_head
        )
        q_spatial, k_spatial = scale_for_cosine_sim(q_spatial, k_spatial, self.scale_spatial[:, None, None], 1e-6)

        if self.spatial_pos_emb is not None:
            spatial_pos = pos[..., -2:].to(qkv_spatial.dtype)
            spatial_pos = rearrange(spatial_pos, "b t h w d -> (b t) (h w) d")
            theta_spatial = self.spatial_pos_emb(spatial_pos).movedim(-2, -3)
            q_spatial = self.spatial_pos_emb.apply_emb(q_spatial, theta_spatial)
            k_spatial = self.spatial_pos_emb.apply_emb(k_spatial, theta_spatial)

        if spatial_block_mask is None:
            out_spatial = F.scaled_dot_product_attention(q_spatial, k_spatial, v_spatial, scale=1.0)
        else:
            out_spatial = flex_attention(q_spatial, k_spatial, v_spatial, scale=1.0, block_mask=spatial_block_mask)
        out_spatial = rearrange(out_spatial, "n nh l e -> n l (nh e)")
        out_spatial = rearrange(out_spatial, "(b t) (h w) c -> b t h w c", b=B, t=T, h=H, w=W)

        # Temporal branch: attend over t per spatial position
        if enable_temporal_attention:
            x_temporal = rearrange(x_, "b t h w c -> (b h w) t c")
            qkv_temporal = self.temporal_qkv_proj(x_temporal)
            q_temporal, k_temporal, v_temporal = rearrange(
                qkv_temporal, "n l (three nh e) -> three n nh l e", three=3, e=self.d_head
            )
            q_temporal, k_temporal = scale_for_cosine_sim(
                q_temporal, k_temporal, self.scale_temporal[:, None, None], 1e-6
            )

            if self.temporal_pos_emb is not None:
                temporal_pos = pos[..., 0:1]
                temporal_pos = rearrange(temporal_pos, "b t h w d -> (b h w) t d").to(qkv_temporal.dtype)
                theta_temporal = self.temporal_pos_emb(temporal_pos).movedim(-2, -3)
                q_temporal = self.temporal_pos_emb.apply_emb(q_temporal, theta_temporal)
                k_temporal = self.temporal_pos_emb.apply_emb(k_temporal, theta_temporal)

            if temporal_block_mask is None:
                out_temporal = F.scaled_dot_product_attention(q_temporal, k_temporal, v_temporal, scale=1.0)
            else:
                out_temporal = flex_attention(
                    q_temporal, k_temporal, v_temporal, scale=1.0, block_mask=temporal_block_mask
                )
            out_temporal = rearrange(out_temporal, "n nh l e -> n l (nh e)")
            out_temporal = rearrange(out_temporal, "(b h w) t c -> b t h w c", b=B, h=H, w=W)
        else:
            out_temporal = torch.zeros_like(out_spatial)

        # Merge branches, project once, then single residual skip over the full block
        out = torch.cat([out_spatial, out_temporal], dim=-1)
        out = self.dropout(out)
        out = self.out_proj(out)
        return out + skip


class LegacyFactorizedAttentionBlock(nn.Module):
    """
    Legacy/sequential factorized attention over (B, T, H, W, C):
    1) spatial attention (per frame), residual in spatial view
    2) temporal attention (per pixel trajectory), residual in temporal view
    """
    def __init__(
        self,
        d_model: int,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        rope_cls_spatial=AxialRoPE2D,
        rope_cls_temporal=AxialRoPE1D,
    ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.n_heads = d_model // d_head
        assert self.n_heads * d_head == d_model, f"{d_model=} must be divisible by {d_head=}"

        if d_cond_norm is not None:
            self.spatial_norm = AdaRMSNorm(d_model, d_cond_norm)
            self.temporal_norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.spatial_norm = RMSNorm(d_model)
            self.temporal_norm = RMSNorm(d_model)

        self.spatial_qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.temporal_qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.spatial_out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.temporal_out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

        self.spatial_pos_emb = instantiate_optional_class(
            rope_cls_spatial,
            d_head,
            self.n_heads,
            relative_canvas=True,
            learnable_freqs=False,
        )

        self.temporal_pos_emb = instantiate_optional_class(
            rope_cls_temporal,
            d_head,
            self.n_heads,
            relative_canvas=True,
            learnable_freqs=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b t h w c"],
        pos: Float[torch.Tensor, "b t h w d"],
        cond_norm: Float[torch.Tensor, "b e"] | None = None,
        spatial_block_mask=None,
        temporal_block_mask=None,
        enable_temporal_attention: bool = True,
    ) -> Float[torch.Tensor, "b t h w c"]:
        # print("[DEBUG] using legacy factorized attention block", flush=True)
        B, T, H, W, _ = x.shape
        HW = H * W

        # Spatial stage: attend over HxW within each frame independently.
        x_spatial = rearrange(x, "b t h w c -> (b t) (h w) c")
        skip_spatial = x_spatial
        if cond_norm is not None:
            if cond_norm.ndim == 2:
                cond_norm_spatial = cond_norm.repeat_interleave(T, dim=0)
            elif cond_norm.ndim == 3:
                n_tokens = cond_norm.shape[1]
                if n_tokens == 1:
                    cond_norm_spatial = cond_norm.repeat_interleave(T, dim=0)
                elif n_tokens == T:
                    cond_norm_spatial = rearrange(cond_norm, "b t d -> (b t) 1 d")
                elif n_tokens == HW:
                    cond_norm_spatial = cond_norm.repeat_interleave(T, dim=0)
                elif n_tokens == T * HW:
                    cond_norm_spatial = rearrange(cond_norm, "b (t hw) d -> (b t) hw d", t=T, hw=HW)
                else:
                    raise AssertionError(
                        f"Unsupported cond_norm shape for sequential spatial stage: {cond_norm.shape=}, "
                        f"expected second dim in {{1, {T}, {HW}, {T * HW}}}"
                    )
            else:
                raise AssertionError(f"Expected cond_norm with 2 or 3 dims, got {cond_norm.shape=}")
            x_spatial = self.spatial_norm(x_spatial, cond_norm_spatial)
        else:
            x_spatial = self.spatial_norm(x_spatial)

        qkv_spatial = self.spatial_qkv_proj(x_spatial)
        q_spatial, k_spatial, v_spatial = rearrange(
            qkv_spatial, "n l (three nh e) -> three n nh l e", three=3, e=self.d_head
        )
        q_spatial, k_spatial = scale_for_cosine_sim(q_spatial, k_spatial, self.scale[:, None, None], 1e-6)

        if self.spatial_pos_emb is not None:
            spatial_pos = pos[..., -2:].to(qkv_spatial.dtype)
            spatial_pos = rearrange(spatial_pos, "b t h w d -> (b t) (h w) d")
            theta_spatial = self.spatial_pos_emb(spatial_pos).movedim(-2, -3)
            q_spatial = self.spatial_pos_emb.apply_emb(q_spatial, theta_spatial)
            k_spatial = self.spatial_pos_emb.apply_emb(k_spatial, theta_spatial)

        if spatial_block_mask is None:
            out_spatial = F.scaled_dot_product_attention(q_spatial, k_spatial, v_spatial, scale=1.0)
        else:
            out_spatial = flex_attention(q_spatial, k_spatial, v_spatial, scale=1.0, block_mask=spatial_block_mask)
        out_spatial = rearrange(out_spatial, "n nh l e -> n l (nh e)")
        out_spatial = self.dropout(out_spatial)
        out_spatial = self.spatial_out_proj(out_spatial)
        out_spatial = out_spatial + skip_spatial

        if not enable_temporal_attention:
            return rearrange(out_spatial, "(b t) (h w) c -> b t h w c", b=B, t=T, h=H, w=W)

        # Temporal stage: attend over T per spatial position.
        x_temporal = rearrange(out_spatial, "(b t) (h w) c -> (b h w) t c", b=B, t=T, h=H, w=W)
        skip_temporal = x_temporal
        if cond_norm is not None:
            if cond_norm.ndim == 2:
                cond_norm_temporal = cond_norm.repeat_interleave(HW, dim=0)
            else:
                n_tokens = cond_norm.shape[1]
                if n_tokens == 1:
                    cond_norm_temporal = cond_norm.repeat_interleave(HW, dim=0)
                elif n_tokens == T:
                    cond_norm_temporal = rearrange(
                        cond_norm[:, None, :, :].expand(B, HW, T, cond_norm.shape[-1]),
                        "b hw t d -> (b hw) t d",
                    )
                elif n_tokens == HW:
                    cond_norm_temporal = rearrange(cond_norm, "b hw d -> (b hw) 1 d")
                elif n_tokens == T * HW:
                    cond_norm_temporal = rearrange(cond_norm, "b (t hw) d -> (b hw) t d", t=T, hw=HW)
                else:
                    raise AssertionError(
                        f"Unsupported cond_norm shape for sequential temporal stage: {cond_norm.shape=}, "
                        f"expected second dim in {{1, {T}, {HW}, {T * HW}}}"
                    )
            x_temporal = self.temporal_norm(x_temporal, cond_norm_temporal)
        else:
            x_temporal = self.temporal_norm(x_temporal)

        qkv_temporal = self.temporal_qkv_proj(x_temporal)
        q_temporal, k_temporal, v_temporal = rearrange(
            qkv_temporal, "n l (three nh e) -> three n nh l e", three=3, e=self.d_head
        )
        q_temporal, k_temporal = scale_for_cosine_sim(q_temporal, k_temporal, self.scale[:, None, None], 1e-6)

        if self.temporal_pos_emb is not None:
            temporal_pos = pos[..., 0:1]
            temporal_pos = rearrange(temporal_pos, "b t h w d -> (b h w) t d").to(qkv_temporal.dtype)
            theta_temporal = self.temporal_pos_emb(temporal_pos).movedim(-2, -3)
            q_temporal = self.temporal_pos_emb.apply_emb(q_temporal, theta_temporal)
            k_temporal = self.temporal_pos_emb.apply_emb(k_temporal, theta_temporal)

        if temporal_block_mask is None:
            out_temporal = F.scaled_dot_product_attention(q_temporal, k_temporal, v_temporal, scale=1.0)
        else:
            out_temporal = flex_attention(
                q_temporal, k_temporal, v_temporal, scale=1.0, block_mask=temporal_block_mask
            )
        out_temporal = rearrange(out_temporal, "n nh l e -> n l (nh e)")
        out_temporal = self.dropout(out_temporal)
        out_temporal = self.temporal_out_proj(out_temporal)
        out_temporal = out_temporal + skip_temporal
        return rearrange(out_temporal, "(b h w) t c -> b t h w c", b=B, h=H, w=W)


class DimensionAttentionBlock(AttentionBlock):
    """
    Expects (b, ..., dim), reshapes in forward, and applies attention over all '...' dimensions.
    This allows you to apply AdaRMSNorm only to specific dimensions.
    """
    def forward(
        self,
        x: Float[torch.Tensor, 'b ... c'],
        pos: Float[torch.Tensor, 'b ... d'],
        cond_norm: Float[torch.Tensor, 'b ... e'] = None,
    ):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        B, *DIMS, C = x.shape
        qkv = self.qkv_proj(x)
        x = rearrange(x, "b ... c -> b (...) c")
        pos = rearrange(pos, "b ... c -> b (...) c")
        qkv = rearrange(qkv, "b ... c -> b (...) c")
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
        theta = theta.movedim(-2, -3)
        q = self.pos_emb.apply_emb(q, theta)
        k = self.pos_emb.apply_emb(k, theta)
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        x = x.view(B, *DIMS, C)
        return x + skip


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cross: int,
        d_head: int = 64,
        d_cond_norm: int | None = None,
        dropout: float = 0.0,
        rope_cls=AxialRoPE2D,
    ):
        super().__init__()
        self.d_head = d_head
        self.d_model = d_model
        self.n_heads = d_model // d_head
        # TODO check if ada norm makes sense for keys and values
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.norm_cross = RMSNorm(d_cross)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_cross, d_model * 2, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))

        self.pos_emb = instantiate_optional_class(
            rope_cls,
            d_head,
            self.n_heads,
            relative_canvas=True,
            learnable_freqs=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "b l d"],
        pos: Float[torch.Tensor, "b l nc"],
        x_cross: Float[torch.Tensor, "b l' d'"],
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ) -> Float[torch.Tensor, "b ... d"]:
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x_cross = self.norm_cross(x_cross)
        q = self.q_proj(x)
        kv = self.kv_proj(x_cross)

        q = rearrange(q, "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        if self.pos_emb is not None:
            pos = pos.to(q.dtype)
            theta = self.pos_emb(pos)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
        
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip
    

class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_cross=None,
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
        rope_cls=AxialRoPE2D,
        compile: bool = False
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
        
        d_ff = d_model * ff_expand

        self.self_attn = AttentionBlock(
            d_model=d_model,
            d_head=d_head,
            d_cond_norm=d_cond_norm,
            dropout=dropout,
            rope_cls=rope_cls,
        )

        self.cross_attn = None
        if d_cross is not None:
            self.cross_attn = CrossAttentionBlock(
                d_model=d_model,
                d_cross=d_cross,
                d_head=d_head,
                d_cond_norm=d_cond_norm,
                dropout=dropout,
                rope_cls=rope_cls,
            )
        
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(
        self,
        x: Float[torch.Tensor, "b n c"],
        pos: Float[torch.Tensor, "b n d"],
        cond_norm: Float[torch.Tensor, "b 1|n e"] = None,
        x_cross: Float[torch.Tensor, "b m k"] = None,
        block_mask = None,
    ):
        x = self.self_attn(x, pos, cond_norm=cond_norm, block_mask=block_mask)
        if self.cross_attn is not None:
            x = self.cross_attn(x, pos, x_cross=x_cross, cond_norm=cond_norm)
        x = self.ff(x, cond_norm=cond_norm)
        return x


class FactorizedAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
        rope_cls_spatial=AxialRoPE2D,
        rope_cls_temporal=AxialRoPE1D,
        attention_impl: str = "parallel",
        compile: bool = False,
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile

        d_ff = d_model * ff_expand
        if attention_impl == "parallel":
            self.self_attn = FactorizedAttentionBlock(
                d_model=d_model,
                d_head=d_head,
                d_cond_norm=d_cond_norm,
                dropout=dropout,
                rope_cls_spatial=rope_cls_spatial,
                rope_cls_temporal=rope_cls_temporal,
            )
        elif attention_impl in ("legacy", "sequential"):
            self.self_attn = LegacyFactorizedAttentionBlock(
                d_model=d_model,
                d_head=d_head,
                d_cond_norm=d_cond_norm,
                dropout=dropout,
                rope_cls_spatial=rope_cls_spatial,
                rope_cls_temporal=rope_cls_temporal,
            )
        else:
            raise ValueError(f"Unknown factorized attention implementation: {attention_impl}")
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

        if COMPILE:
            self.forward = compile_fn(self.forward)

    def forward(
        self,
        x: Float[torch.Tensor, "b t h w c"],
        pos: Float[torch.Tensor, "b t h w d"],
        cond_norm: Float[torch.Tensor, "b e"] | None = None,
        spatial_block_mask=None,
        temporal_block_mask=None,
        enable_temporal_attention: bool = True,
    ):
        x = self.self_attn(
            x=x,
            pos=pos,
            cond_norm=cond_norm,
            spatial_block_mask=spatial_block_mask,
            temporal_block_mask=temporal_block_mask,
            enable_temporal_attention=enable_temporal_attention,
        )

        B, T, H, W, C = x.shape
        x_ff = rearrange(x, "b t h w c -> b (t h w) c")
        x_ff = self.ff(x_ff, cond_norm=cond_norm)
        x = rearrange(x_ff, "b (t h w) c -> b t h w c", t=T, h=H, w=W)
        return x


# ===================================================================================================


class RegisterAttentionBlock(AttentionBlock):
    """ [register tokens, ... other tokens] """
    def __init__(self, *args, n_registers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_registers = n_registers
        assert self.n_registers >= 0, "n_registers must be non-negative"

    def forward(self, x, pos, cond_norm=None):
        skip = x

        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        qkv = self.qkv_proj(x)
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)

        q, k, v = rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        # exclude register tokens from RoPE (registers first)
        if self.n_registers > 0:
            q_r = q[:, :, :self.n_registers, :]
            k_r = k[:, :, :self.n_registers, :]
            q = q[:, :, self.n_registers:, :]
            k = k[:, :, self.n_registers:, :]

            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)

            # concatenate back
            q = torch.cat([q_r, q], dim=-2)
            k = torch.cat([k_r, k], dim=-2)
        else:
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class RegisterCrossAttentionBlock(CrossAttentionBlock):
    """ [register tokens, ... other tokens] """
    def __init__(self, *args, n_registers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_registers = n_registers
        assert self.n_registers >= 0, "n_registers must be non-negative"

    def forward(
        self,
        x: Float[torch.Tensor, "b l d"],
        pos: Float[torch.Tensor, "b l nc"],
        x_cross: Float[torch.Tensor, "b l' d'"],
        cond_norm: Float[torch.Tensor, "b d"] | None = None,
    ) -> Float[torch.Tensor, "b ... d"]:
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)

        x_cross = self.norm_cross(x_cross)
        kv = self.kv_proj(x_cross)
        q = self.q_proj(x)

        pos = pos.to(q.dtype)
        theta = self.pos_emb(pos)

        q = rearrange(q, "n l (nh e) -> n nh l e", e=self.d_head)
        k, v = rearrange(kv, "n l (t nh e) -> t n nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)

        theta = theta.movedim(-2, -3)
        if self.n_registers > 0:
            q_r = q[:, :, :self.n_registers, :]
            q = q[:, :, self.n_registers:, :]
            q = self.pos_emb.apply_emb(q, theta)
            q = torch.cat([q_r, q], dim=-2)
        else:
            q = self.pos_emb.apply_emb(q, theta)
        
        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = rearrange(x, "n nh l e -> n l (nh e)")

        x = self.dropout(x)
        x = self.out_proj(x)
        return x + skip


class TransformerLayerWithRegisters(nn.Module):
    def __init__(
        self,
        d_model,
        d_cross=None,
        d_head=64,
        d_cond_norm=None,
        dropout=0.0,
        ff_expand=3,
        rope_cls=AxialRoPE2D,
        compile: bool = False,
        n_registers: int = 1,
    ):
        super().__init__()
        global COMPILE
        COMPILE = compile
        self.n_registers = n_registers
        
        d_ff = d_model * ff_expand

        self.self_attn = RegisterAttentionBlock(
            d_model=d_model,
            d_head=d_head,
            d_cond_norm=d_cond_norm,
            dropout=dropout,
            rope_cls=rope_cls,
            n_registers=n_registers,
        )

        self.cross_attn = None
        if d_cross is not None:
            self.cross_attn = RegisterCrossAttentionBlock(
                d_model=d_model,
                d_cross=d_cross,
                d_head=d_head,
                d_cond_norm=d_cond_norm,
                dropout=dropout,
                rope_cls=rope_cls,
                n_registers=n_registers,
            )
        
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm, dropout)

        if COMPILE: self.forward = compile_fn(self.forward)

    def forward(
        self,
        x: Float[torch.Tensor, "b n c"],
        pos: Float[torch.Tensor, "b n d"],
        cond_norm: Float[torch.Tensor, "b 1|n e"] = None,
        x_cross: Float[torch.Tensor, "b m k"] = None,
    ):
        x = self.self_attn(x, pos, cond_norm=cond_norm)
        if self.cross_attn is not None:
            x = self.cross_attn(x, pos, x_cross=x_cross, cond_norm=cond_norm)
        x = self.ff(x, cond_norm=cond_norm)
        return x


if __name__ == "__main__":
    transformer = TransformerLayer(768, d_cond_norm=128, d_cross=64)
    kwargs = dict(
        x=torch.randn((1, 256, 768)),
        pos=torch.randn((1, 256, 2)),
        cond_norm=torch.randn((1, 1, 128)),
        x_cross=torch.randn((1, 256, 64)),
    )
    out = transformer(**kwargs)
    print(out.shape)
