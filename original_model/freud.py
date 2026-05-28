from typing import Union
from jaxtyping import Float

import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import einops
from functools import reduce

"""
#########################
Utilities
#########################
"""

"""Layer Helpers"""

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def bounding_box(h, w):
    w_adj = w
    h_adj = h
    ar_adj = w_adj / h_adj
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj
    return y_min, y_max, x_min, x_max

"""Positional Encoding"""

def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)

def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2

def make_axial_pos_2d(h, w, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w)
    h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
    w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)

def make_grid_3d(t_pos, h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(t_pos, h_pos, w_pos, indexing='ij'), dim=-1)
    t, h, w, d = grid.shape
    return grid.view(t * h * w, d)

def make_axial_pos_3d(t, h, w, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w)
    t_pos = torch.arange(t, dtype=dtype, device=device).float().to(dtype)
    h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
    w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid_3d(t_pos, h_pos, w_pos)

"""
#########################
AxialRoPE
#########################
"""

class AxialRoPE(nn.Module):
    def __init__(self, dim, n_heads, n_dims):
        super().__init__()
        assert n_dims in (2, 3)
        self.n_dims = n_dims
        min_freq = math.pi
        max_freq = 10.0 * math.pi
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)

        n_freqs = n_heads * dim // 4

        spatial_freqs = torch.stack([torch.linspace(log_min, log_max, n_freqs + 1)[:-1].exp()] * 2)
        if self.n_dims == 2:
            # Keep 2D key name compatible with existing checkpoints.
            self.register_buffer("freqs", spatial_freqs.view(2, dim // 4, n_heads).mT.contiguous())
        else:
            self.register_buffer("spatial_freqs", spatial_freqs.view(2, dim // 4, n_heads).mT.contiguous())

        if self.n_dims == 3:
            num_time_theta = 10_000
            temporal_freqs = 1.0 / (num_time_theta ** (torch.arange(0, n_freqs).float() / n_freqs))
            self.register_buffer("temporal_freqs", temporal_freqs.view(dim // 4, n_heads).mT.contiguous())

    @staticmethod
    def apply_emb(x, theta):
        out_dtype = x.dtype
        dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
        d = theta.shape[-1]
        assert d * 2 <= x.shape[-1]
        x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
        x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
        cos, sin = torch.cos(theta), torch.sin(theta)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
        return torch.cat((y1, y2, x3), dim=-1)

    def forward(self, pos):
        if self.n_dims == 2:
            theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
            theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
            return torch.cat((theta_h, theta_w), dim=-1)

        theta_t = pos[..., None, 0:1] * self.temporal_freqs.to(pos.dtype)
        theta_h = pos[..., None, 1:2] * self.spatial_freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 2:3] * self.spatial_freqs[1].to(pos.dtype)
        return torch.cat((theta_t, theta_h, theta_w), dim=-1)

class AxialRoPE2D(AxialRoPE):
    def __init__(self, dim, n_heads):
        super().__init__(dim, n_heads, n_dims=2)

class AxialRoPE3D(AxialRoPE):
    def __init__(self, dim, n_heads):
        super().__init__(dim, n_heads, n_dims=3)

"""
#########################
Layers
#########################
"""

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)

def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
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
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)

class LinearGEGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        x = x @ self.weight.mT
        if self.bias is not None:
            x = x + self.bias
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        x = x @ self.weight.mT
        if self.bias is not None:
            x = x + self.bias
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

class MappingFeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = LinearGEGLU(d_model, d_ff, bias=False)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return x + skip

class MappingNetwork(nn.Module):
    def __init__(self, d_model, n_layers=2, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 3
        self.width = d_model
        self.depth = n_layers
        self.d_ff = d_ff
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([MappingFeedForwardBlock(d_model, d_ff) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x

"""Attention Helpers"""

def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)

def scale_for_cosine_sim_qkv(qkv, scale, eps):
    q, k, v = qkv.unbind(2)
    q, k = scale_for_cosine_sim(q, k, scale[:, None], eps)
    return torch.stack((q, k, v), dim=2)

def mk_1d_flex_natten_mod(seq_len, kernel_size):
    
    def get_block_mask(b, h, q_idx, kv_idx):
        kernel_x = q_idx.clamp(kernel_size // 2, (seq_len-1) - kernel_size//2)
        mask = (kernel_x - kv_idx).abs() <= kernel_size//2
        return mask
    
    def get_score_mod(score, b, h, q_idx, kv_idx):
        mask = get_block_mask(b, h, q_idx, kv_idx)
        return torch.where(mask, score, -float("inf"))
    
    return get_block_mask, get_score_mod

def mk_2d_flex_natten_mod(dims, kernel_size):
    # Build 2D local-neighborhood masks in flattened token space.
    H, W = dims
    K_H, K_W = kernel_size

    def get_x_y(idx):
        return idx // W, idx % W

    def get_block_mask(b, h, q_idx, kv_idx):
        q_x, q_y = get_x_y(q_idx)
        kv_x, kv_y = get_x_y(kv_idx)
        kernel_x = q_x.clamp(K_W // 2, (W - 1) - K_W // 2)
        kernel_y = q_y.clamp(K_H // 2, (H - 1) - K_H // 2)
        hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
        vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
        return hori_mask & vert_mask
    
    def get_score_mod(score, b, h, q_idx, kv_idx):
        mask = get_block_mask(b,h,q_idx, kv_idx)
        return torch.where(mask, score, -float("inf"))
    
    return get_block_mask, get_score_mod

def mk_3d_flex_natten_mod(dims, kernel_size):
    # Build 3D local-neighborhood masks in flattened token space.
    T, H, W = dims
    K_T, K_H, K_W = kernel_size

    def get_t_x_y(idx):
        t_idx = idx // (H * W)
        remaining = idx % (H * W)
        x_idx = remaining // W
        y_idx = remaining % W
        return t_idx, x_idx, y_idx

    def get_block_mask(b, h, q_idx, kv_idx):
        q_t, q_x, q_y = get_t_x_y(q_idx)
        kv_t, kv_x, kv_y = get_t_x_y(kv_idx)

        kernel_t = q_t.clamp(K_T // 2, (T - 1) - K_T // 2)
        kernel_x = q_x.clamp(K_H // 2, (H - 1) - K_H // 2)
        kernel_y = q_y.clamp(K_W // 2, (W - 1) - K_W // 2)
        time_mask = (kernel_t - kv_t).abs() <= K_T // 2
        hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
        vert_mask = (kernel_y - kv_y).abs() <= K_H // 2

        return time_mask & hori_mask & vert_mask

    def get_score_mod(score, b, h, q_idx, kv_idx):
        mask = get_block_mask(b, h, q_idx, kv_idx)
        return torch.where(mask, score, -float("inf"))

    return get_block_mask, get_score_mod

POS_ENC = {
    "AxialRoPE2D": AxialRoPE2D,
    "AxialRoPE3D": AxialRoPE3D
}

"""Transformer Layers"""

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, d_cond_norm=None):
        super().__init__()
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond_norm=None):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return x + skip


class GenericAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_cls,
        d_head: int = 64,
        d_cond_norm: int | None = None,
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
        self.pos_emb = POS_ENC[pos_enc_cls](d_head // 2, self.n_heads)
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.block_mask = None
        self.flex_attention = flex_attention

    def extra_repr(self):
        return f"d_head={self.d_head},"

    def forward(
        self,
        x,
        pos,
        cond_norm=None,
        neighborhood_attn: bool=False,
        neighborhood_size: int=7,
        temporal_neighborhood_size: int=3,
        image_size: tuple | None = None,
    ):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        qkv = self.qkv_proj(x)
        pos = pos.to(qkv.dtype)
        theta = self.pos_emb(pos)

        if neighborhood_attn and x.is_cuda:
            qkv = einops.rearrange(qkv, "n l (t nh e) -> n l t nh e", t=3, e=self.d_head)
            _, seq_len, _, _, _ = qkv.size()
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta = torch.stack((theta, theta, torch.zeros_like(theta)), dim=-3)
            qkv = self.pos_emb.apply_emb(qkv, theta)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if self.block_mask is None:
                image_size = image_size if image_size is not None else (128, 128)
                if len(image_size) == 2:
                    mask_mod, _ = mk_2d_flex_natten_mod(
                        image_size,
                        (neighborhood_size, neighborhood_size)
                    )
                    self.block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=qkv.device)
                else:
                    mask_mod, _ = mk_3d_flex_natten_mod(
                        image_size,
                        (temporal_neighborhood_size, neighborhood_size, neighborhood_size)
                    )
                    self.block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=qkv.device)
            x = self.flex_attention(q, k, v, block_mask=self.block_mask, scale=1.)
            x = einops.rearrange(x, "n l nh e -> n l (nh e)")
        else:
            q, k, v = einops.rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta = theta.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta)
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x = einops.rearrange(x, "n nh l e -> n l (nh e)")

        x = self.out_proj(x)
        return x + skip


class FactorizedAttentionBlock(GenericAttentionBlock):
    def __init__(
        self,
        d_model: int,
        pos_enc_cls,
        d_head: int = 64,
        d_cond_norm: int | None = None,
    ) -> None:
        super().__init__(d_model, pos_enc_cls, d_head, d_cond_norm)

        self.spatial_qkv_proj = self.qkv_proj
        self.spatial_norm = self.norm
        self.spatial_out_proj = self.out_proj
        self.spatial_block_mask = None

        self.temporal_qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        if d_cond_norm is not None:
            self.temporal_norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.temporal_norm = RMSNorm(d_model)
        self.temporal_out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.temporal_block_mask = None

    def forward(
        self,
        x,
        pos,
        cond_norm=None,
        image_size: tuple | None = None,
        neighborhood_attn: bool=False,
        neighborhood_size: int=7,
        temporal_neighborhood_size: int=3,
    ):
        assert image_size is not None and len(image_size) == 3, "FactorizedAttention requires image_size parameter with shape (frames, height, width)"
        t, h, w = image_size

        x_spatio_temporal = x.reshape(x.size(0), t, h, w, x.size(-1))
        x_spatial = einops.rearrange(x_spatio_temporal, "b t h w c -> (b t) (h w) c")

        skip = x_spatial
        if cond_norm is not None:
            cond_norm_spatial = cond_norm.unsqueeze(1).repeat(1, t, 1).view(-1, cond_norm.size(-1))
            x_spatial = self.spatial_norm(x_spatial, cond_norm_spatial)
        else:
            x_spatial = self.spatial_norm(x_spatial)
        qkv = self.spatial_qkv_proj(x_spatial)
        pos_spatio_temporal = pos.reshape(pos.size(0), t, h, w, pos.size(-1))
        pos = pos.to(qkv.dtype)
        pos_spatial = einops.rearrange(pos_spatio_temporal, "b t h w c -> (b t) (h w) c")
        theta_spatial = self.pos_emb(pos_spatial)

        if neighborhood_attn and x_spatial.is_cuda:
            qkv = einops.rearrange(qkv, "n l (t nh e) -> n l t nh e", t=3, e=self.d_head)
            _, seq_len, _, _, _ = qkv.size()
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta_spatial = torch.stack((theta_spatial, theta_spatial, torch.zeros_like(theta_spatial)), dim=-3)
            qkv = self.pos_emb.apply_emb(qkv, theta_spatial)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if self.spatial_block_mask is None:
                mask_mod, _ = mk_2d_flex_natten_mod(
                    (h, w),
                    (neighborhood_size, neighborhood_size)
                )
                self.spatial_block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=qkv.device)
            x = self.flex_attention(q, k, v, block_mask=self.spatial_block_mask, scale=1.)
            x = einops.rearrange(x, "n l nh e -> n l (nh e)")
        else:
            q, k, v = einops.rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta_spatial = theta_spatial.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta_spatial)
            k = self.pos_emb.apply_emb(k, theta_spatial)
            x_spatial = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x_spatial = einops.rearrange(x_spatial, "n nh l e -> n l (nh e)")

        x_spatial = self.spatial_out_proj(x_spatial)
        x_spatial = x_spatial + skip

        x_spatio_temporal = einops.rearrange(x_spatial, "(b t) (h w) c -> b t h w c", t=t, h=h, w=w)
        x_temporal = einops.rearrange(x_spatio_temporal, "b t h w c -> (b h w) t c")
        pos_temporal = einops.rearrange(pos_spatio_temporal, "b t h w c -> (b h w) t c")

        skip = x_temporal
        if cond_norm is not None:
            cond_norm_temporal = cond_norm.unsqueeze(1).repeat(1, h * w, 1).view(-1, cond_norm.size(-1))
            x_temporal = self.temporal_norm(x_temporal, cond_norm_temporal)
        else:
            x_temporal = self.temporal_norm(x_temporal)
        qkv = self.temporal_qkv_proj(x_temporal)
        theta_temporal = self.pos_emb(pos_temporal)

        if neighborhood_attn and x_temporal.is_cuda:
            qkv = einops.rearrange(qkv, "n l (t nh e) -> n l t nh e", t=3, e=self.d_head)
            _, seq_len, _, _, _ = qkv.size()
            qkv = scale_for_cosine_sim_qkv(qkv, self.scale, 1e-6)
            theta_temporal = torch.stack((theta_temporal, theta_temporal, torch.zeros_like(theta_temporal)), dim=-3)
            qkv = self.pos_emb.apply_emb(qkv, theta_temporal)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if self.temporal_block_mask is None:
                mask_mod, _ = mk_1d_flex_natten_mod(t, temporal_neighborhood_size)
                self.temporal_block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=qkv.device)
            x = self.flex_attention(q, k, v, block_mask=self.temporal_block_mask, scale=1.)
            x = einops.rearrange(x, "n l nh e -> n l (nh e)")
        else:
            q, k, v = einops.rearrange(qkv, "n l (t nh e) -> t n nh l e", t=3, e=self.d_head)
            q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], 1e-6)
            theta_temporal = theta_temporal.movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta_temporal)
            k = self.pos_emb.apply_emb(k, theta_temporal)
            x_temporal = F.scaled_dot_product_attention(q, k, v, scale=1.0)
            x_temporal = einops.rearrange(x_temporal, "n nh l e -> n l (nh e)")

        x_temporal = self.temporal_out_proj(x_temporal)
        x_temporal = x_temporal + skip

        return einops.rearrange(x_temporal, "(b h w) t c -> b (t h w) c", h=h, w=w)

class GenericTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        pos_enc_cls="AxialRoPE3D",
        d_head=64,
        d_cond_norm=None,
        ff_expand=3,
        neighborhood_attn: bool=False,
        neighborhood_size: int=7,
        temporal_neighborhood_size: int=3,
        use_factorized_attention: bool=False,
    ):
        super().__init__()
        d_ff = d_model * ff_expand

        self.neighborhood_attn = neighborhood_attn
        self.neighborhood_size = neighborhood_size
        self.temporal_neighborhood_size = temporal_neighborhood_size

        if use_factorized_attention:
            self.self_attn = FactorizedAttentionBlock(d_model, pos_enc_cls, d_head, d_cond_norm)
        else:
            self.self_attn = GenericAttentionBlock(d_model, pos_enc_cls, d_head, d_cond_norm)
        self.ff = FeedForwardBlock(d_model, d_ff, d_cond_norm)

    def forward(self, x, pos, cond_norm=None, image_size: tuple | None = None, **kwargs):
        x = self.self_attn(
            x,
            pos,
            cond_norm=cond_norm,
            image_size=image_size,
            neighborhood_attn=self.neighborhood_attn,
            neighborhood_size=self.neighborhood_size,
            temporal_neighborhood_size=self.temporal_neighborhood_size,
        )
        x = self.ff(x, cond_norm=cond_norm)
        return x

"""Splits and Merges"""


class SimpleProj(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x)


class TokenMerge2D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = nn.Linear(in_features * self.h * self.w, out_features, bias=False)

    def downscale_pos(self, pos, **kwargs):
        pos = einops.rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=self.h, nw=self.w)
        return torch.mean(pos, dim=-2)

    def forward(self, x, **kwargs):
        x = einops.rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenMerge3D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w, out_features, bias=False)

    def downscale_pos(self, pos, **kwargs):
        pos = einops.rearrange(pos, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw) e", nt=self.t, nh=self.h, nw=self.w)
        return torch.mean(pos, dim=-2)

    def forward(self, x, **kwargs):
        x = einops.rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        return self.proj(x)


class CondTokenMerge3D(nn.Module):
    def __init__(self, in_features, out_features, cond_features, patch_size=(2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features * self.t * self.h * self.w + cond_features, out_features, bias=False)

    def downscale_pos(self, pos, **kwargs):
        pos = einops.rearrange(pos, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw) e", nt=self.t, nh=self.h, nw=self.w)
        return torch.mean(pos, dim=-2)

    def forward(self, x, cond_tokens, **kwargs):
        x = einops.rearrange(x, "... (t nt) (h nh) (w nw) e -> ... t h w (nt nh nw e)", nt=self.t, nh=self.h, nw=self.w)
        x = torch.cat((x, cond_tokens), dim=-1)
        return self.proj(x)


class TokenSplit3D(nn.Module):
    def __init__(self, in_features, out_features, patch_size=(1, 2, 2)):
        super().__init__()
        self.t = patch_size[0]
        self.h = patch_size[1]
        self.w = patch_size[2]
        self.proj = nn.Linear(in_features, out_features * self.t * self.h * self.w, bias=False)
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip=None, **kwargs):
        x = self.proj(x)
        x = einops.rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
        if skip is None:
            return x
        return torch.lerp(skip, x, self.fac.to(x.dtype))

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
        x = einops.rearrange(x, "... t h w (nt nh nw e) -> ... (t nt) (h nh) (w nw) e", nt=self.t, nh=self.h, nw=self.w)
        return x
"""
#########################
Transformer Backbone
#########################
"""

class Transformer(nn.Module):
    def __init__(
        self,
        down_merges: list[nn.Module],
        down_levels: list[nn.ModuleList],
        up_splits: list[nn.Module] | None = None,
        up_levels: list[nn.ModuleList] | None = None,
        mid_merge: nn.Module | None = None,
        mid_level: nn.ModuleList | None = None,
        mid_split: nn.Module | None = None,
    ):
        super().__init__()
        if up_splits is None:
            up_splits = []
        if up_levels is None:
            up_levels = []
        assert len(down_merges) == len(down_levels), "down_merges and down_levels must have same length"
        assert len(up_splits) == len(up_levels), "up_splits and up_levels must have same length"
        self.merges = nn.ModuleList(down_merges)
        self.down_levels = nn.ModuleList(down_levels)
        self.splits = nn.ModuleList(up_splits)
        self.up_levels = nn.ModuleList(up_levels)
        self.mid_merge = mid_merge
        self.mid_level = mid_level
        self.mid_split = mid_split

    def forward(self, x: Float[torch.Tensor, "B C *DIMS"], pos: Float[torch.Tensor, "B cn *DIM"], **kwargs):
        h, w = x.size(-2), x.size(-1)

        x = einops.rearrange(x, "b c ... -> b ... c")
        pos = einops.rearrange(pos, "b cn ... -> b ... cn")

        C_pos = pos.shape[-1]

        skips, poses = [], []
        for merge, level in zip(self.merges, self.down_levels):
            skips.append(x)
            x = merge(x, **kwargs)
            pos = merge.downscale_pos(pos, **kwargs)
            poses.append(pos)
            B, *DIMS, C = x.shape
            x = x.reshape(B, -1, C)
            pos = pos.reshape(B, -1, C_pos)
            h, w = DIMS[-2], DIMS[-1]
            if len(DIMS) > 2:
                t = DIMS[-3]
            image_size = (h, w) if len(DIMS) == 2 else (t, h, w)
            for layer in level:
                x = layer(x, pos=pos, image_size=image_size, **kwargs)
            x = x.reshape(B, *DIMS, C)
            pos = pos.reshape(B, *DIMS, C_pos)

        if self.mid_level is not None:
            skip = x
            x = self.mid_merge(x, **kwargs)
            pos = self.mid_merge.downscale_pos(pos, **kwargs)
            B, *DIMS, C = x.shape
            x = x.reshape(B, -1, C)
            pos = pos.reshape(B, -1, C_pos)
            h, w = DIMS[-2], DIMS[-1]
            if len(DIMS) > 2:
                t = DIMS[-3]
            image_size = (h, w) if len(DIMS) == 2 else (t, h, w)
            for layer in self.mid_level:
                x = layer(x, pos=pos, image_size=image_size, **kwargs)
            x = x.reshape(B, *DIMS, C)
            pos = pos.reshape(B, *DIMS, C_pos)
            x = self.mid_split(x, skip=skip, **kwargs)

        for split, level, skip, pos in reversed(list(zip(self.splits, self.up_levels, skips, poses))):
            B, *DIMS, C = x.shape
            x = x.reshape(B, -1, C)
            pos = pos.reshape(B, -1, C_pos)
            h, w = DIMS[-2], DIMS[-1]
            if len(DIMS) > 2:
                t = DIMS[-3]
            image_size = (h, w) if len(DIMS) == 2 else (t, h, w)
            for layer in level:
                x = layer(x, pos=pos, image_size=image_size, **kwargs)
            x = x.reshape(B, *DIMS, C)
            pos = pos.reshape(B, *DIMS, C_pos)
            x = split(x, skip=skip, **kwargs)

        x = einops.rearrange(x, "b ... c -> b c ...")

        return x

"""
#########################
FREUD First-Stage Model
#########################
"""

class FreudDiffusionAE(nn.Module):
    """Implements a Frame-wise Encoder Unified Decoder model as a DiffusionAE (FreudDiffAE).
    Therefore, the encoder model is a 2D image space DiT/SiT while the decoder is a 3D video space DiT/SiT.
    The assumption is that decoding the entire sequence of frames leads to higher temporal consistency, while frame-wise encoding allows for downstream RaMViD training.
    """

    def __init__(
            self,
            encoder: nn.Module,
            unet: nn.Module,
            mapping: MappingNetwork,
            img_size: tuple[int, int],
            num_channels: int,
            num_timesteps: int,
            use_outlier_punishment: bool=True,
            outlier_threshold: float=3,
            outlier_punishment_weight: float=1e5,
            reg_weight: float=1e-3,
        ) -> None:
        super().__init__()
        self.unet = unet
        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)
        self.mapping = mapping
        self.encoder = encoder
        self.img_size = img_size
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.reg_weight = reg_weight

        self.use_outlier_punishment = use_outlier_punishment
        self.outlier_threshold = outlier_threshold
        self.outlier_punishment_weight = outlier_punishment_weight

    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        B, _, *DIMS = x.shape
        pos = make_axial_pos_3d(*DIMS, device=x.device).view(1, *DIMS, -1).expand(B, -1, -1, -1, -1)
        return pos.movedim(-1, 1)
    
    def get_pos2d(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        B, _, *DIMS = x.shape
        pos = make_axial_pos_2d(*DIMS, device=x.device).view(1, *DIMS, -1).expand(B, -1, -1, -1)
        return pos.movedim(-1, 1)

    def get_conditioning(self, t: Float[torch.Tensor, "b"], c_vid: torch.Tensor=None, c_latent: torch.Tensor=None, **kwargs) -> dict[str, torch.Tensor]:
        time_emb = self.time_in_proj(self.time_emb(t[..., None]))
        cond_time = self.mapping(time_emb)
        cond_dict = {"cond_norm": cond_time}
        if c_latent is None:
            c_latent = self.encode(c_vid=c_vid, **kwargs)
        cond_dict["cond_tokens"] = self._latent_to_cond_tokens(c_latent)
        return cond_dict

    def _latent_to_cond_tokens(self, c_latent: torch.Tensor) -> torch.Tensor:
        assert c_latent.ndim == 5, f"Expected 5D latent, got shape {tuple(c_latent.shape)}"
        if c_latent.size(1) == self.num_timesteps:
            return c_latent
        if c_latent.size(2) == self.num_timesteps:
            return c_latent.movedim(1, -1)
        raise ValueError(
            f"Could not infer latent layout from shape {tuple(c_latent.shape)} for num_timesteps={self.num_timesteps}"
        )

    def _get_zt(self, x: Float[torch.Tensor, "b ..."]) -> Float[torch.Tensor, "b ..."]:
        B = x.size(0)
        t = torch.rand((B,), device=x.device)
        texp = t.view([B, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        dtype = x.dtype
        zt, t = zt.to(dtype), t.to(dtype)
        return zt, t, z1

    def encode(self, c_vid: torch.Tensor, reg_weight: float=None, **kwargs) -> torch.Tensor:
        _, ch, t, h, w = c_vid.size()
        if reg_weight is None:
            reg_weight = self.reg_weight

        c_img = einops.rearrange(c_vid, "b c t h w -> (b t) c h w")
        pos = self.get_pos2d(c_img)
        latent = self.encoder(c_img, pos)
        latent = einops.rearrange(latent, "(b t) c h w -> b c t h w", t=t)

        if reg_weight >= 0:
            latent = torch.nn.functional.tanh(latent)
            latent = latent + reg_weight * torch.randn_like(latent)

        return latent
    
    def decode(self, latent: torch.Tensor, noise: torch.Tensor=None, sample_steps: int=50, **kwargs) -> torch.Tensor:
        if noise is None:
            noise = torch.randn((latent.size(0), self.num_channels, self.num_timesteps, self.img_size[0], self.img_size[1])).to(latent.device).to(latent.dtype)
        return self.sample(noise, c_latent=latent, sample_steps=sample_steps)

    def get_loss(self, x: Float[torch.Tensor, "b c t h w"], noise: Float[torch.Tensor, "b ..."], vtheta: Float[torch.Tensor, "b ..."], cond_dict: dict) -> Float[torch.Tensor, "b"]:
        recon_loss = ((noise - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))

        z = cond_dict["cond_tokens"]
        z_b, z_t, z_w, z_h, z_c = z.size()
        z_opl = einops.rearrange(z, "b t h w c -> (b c t) h w")
        outlier_loss = torch.mean(torch.clamp(torch.abs(torch.mean(z_opl, dim=(1, 2), keepdim=True) - z_opl) - self.outlier_threshold * z_opl.std(dim=(1, 2), keepdim=True), min=0), dim=list(range(1, len(z_opl.shape))))
        outlier_loss = einops.rearrange(outlier_loss, "(b c t) -> b c t", c=z_c, t=z_t)
        outlier_loss = torch.sum(outlier_loss, dim=list(range(1, len(outlier_loss.shape))))
        weight = self.outlier_punishment_weight

        return recon_loss + weight * outlier_loss

    def forward(self, x: Float[torch.Tensor, "b c t h w"], **data_kwargs) -> Float[torch.Tensor, ""]:
        zt, t, z1 = self._get_zt(x)
        cond_dict = self.get_conditioning(t, c_vid=x, **data_kwargs)
        pos = self.get_pos(zt)
        vtheta = self.unet(zt, pos=pos, **cond_dict)
        return self.get_loss(x, z1, vtheta, cond_dict)
    
    @torch.no_grad()
    def sample(
        self,
        x: Float[torch.Tensor, "b c ..."],
        c_vid: Float[torch.Tensor, "b c ..."] = None,
        c_latent: torch.Tensor = None,
        sample_steps=50,
        return_list: bool = False,
        **data_kwargs,
    ) -> Float[torch.Tensor, "b ..."] | list[Float[torch.Tensor, "b ..."]]:
        assert (c_vid != None and c_latent == None) or (c_vid == None and c_latent != None), 'Provide either an image or latents for sampling'
        if c_vid is not None:
            c_latent = self.encode(c_vid)

        B = x.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * B, device=x.device, dtype=x.dtype).view([B, *([1] * len(x.shape[1:]))])
        if return_list:
            samples = [x]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * B, device=x.device, dtype=x.dtype)
            cond_dict = self.get_conditioning(t, c_latent=c_latent, **data_kwargs)
            pos = self.get_pos(x)
            vc = self.unet(x, pos=pos, **cond_dict)
            x = x - dt * vc
            if return_list:
                samples.append(x)

        if return_list:
            return samples
        return x

"""
######################################
Inference Functions
######################################
"""

def load_freud_dit_small_p4_rain(
    checkpoint_path: str=None,
    img_size: tuple[int, int] = (384, 384),
    num_channels: int = 1,
    num_timesteps: int = 25,
):
    mapping_width = 384

    encoder = Transformer(
        down_merges=[ TokenMerge2D(in_features=1, out_features=96, patch_size=(4, 4)), ],
        down_levels=[ nn.ModuleList([ GenericTransformerLayer(d_model=96, d_head=32, pos_enc_cls="AxialRoPE2D", neighborhood_attn=True) for _ in range(2) ]), ],
        mid_merge=TokenMerge2D(in_features=96, out_features=mapping_width, patch_size=(2, 2)),
        mid_level=nn.ModuleList([ GenericTransformerLayer(d_model=mapping_width, d_head=64, pos_enc_cls="AxialRoPE2D") for _ in range(4) ]),
        mid_split=SimpleProj(in_features=mapping_width, out_features=4),
    )

    unet = Transformer(
        down_merges=[ TokenMerge3D(in_features=1, out_features=96, patch_size=(1, 4, 4)), ],
        down_levels=[ nn.ModuleList([
                GenericTransformerLayer(d_model=96, d_head=32, d_cond_norm=mapping_width, neighborhood_attn=True, use_factorized_attention=True, ) for _ in range(2) ]),
        ],
        mid_merge=CondTokenMerge3D(in_features=96, out_features=mapping_width, cond_features=4, patch_size=(1, 2, 2)),
        mid_level=nn.ModuleList([ GenericTransformerLayer(d_model=mapping_width, d_head=64, d_cond_norm=mapping_width, use_factorized_attention=True) for _ in range(12) ]),
        mid_split=TokenSplit3D(in_features=mapping_width, out_features=96, patch_size=(1, 2, 2)),
        up_splits=[ TokenSplitLast3D(in_features=96, out_features=1, patch_size=(1, 4, 4)), ],
        up_levels=[ nn.ModuleList([
                GenericTransformerLayer( d_model=96, d_head=32, d_cond_norm=mapping_width, neighborhood_attn=True, use_factorized_attention=True, ) for _ in range(2) ]),
        ],
    )

    model = FreudDiffusionAE(
        encoder=encoder, mapping=MappingNetwork(d_model=mapping_width), unet=unet,
        img_size=img_size, num_channels=num_channels, num_timesteps=num_timesteps,
    )

    if checkpoint_path is None:
        return model

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('._orig_mod', '') if '_orig_mod' in key and key.replace('._orig_mod', '') not in state_dict.keys() else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    return model

@torch.no_grad()
@torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16)
def sample_recon_freud(
    freud: FreudDiffusionAE,
    im: Float[torch.Tensor, "b c t h w"],
    decode_sample_steps: int=10,
    ensemble_size: int=1,
    device='cuda' if torch.cuda.is_available() else 'cpu'
) -> Union[Float[torch.Tensor, "b e c t h w"], Float[torch.Tensor, "b c t h w"]]:
    """Assumption: im Tensor is normalized to [-1, 1] range"""
    freud.eval()
    freud.to(device).to(torch.bfloat16)
    im = im.to(device).to(torch.bfloat16)
    encoded = freud.encode(im)

    ensemble_sample = []
    for i in range(ensemble_size):
        noise = torch.randn_like(im)
        noise = noise.to(device).to(torch.bfloat16)
        decoded = freud.decode(encoded, noise, sample_steps=decode_sample_steps).detach().cpu()
        ensemble_sample.append(decoded)
    ensemble_sample = torch.stack(ensemble_sample, dim=1)
    ensemble_sample = ensemble_sample.squeeze(1)
    return ensemble_sample

