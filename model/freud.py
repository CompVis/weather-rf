import torch
import einops
import math
import wandb
import numpy as np

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from tqdm import tqdm
from typing import Dict, Union, Any
from jaxtyping import Float
from torch.nn.attention.flex_attention import create_block_mask
from .transformer import *

"""
Helpers
"""

def mk_2d_flex_natten_mod(dims, kernel_size):
    # Build 2D local-neighborhood masks in flattened token space.
    H, W = dims
    K_H, K_W = kernel_size

    def get_x_y(idx):
        return idx // W, idx % W

    def get_block_mask(b, h, q_idx, kv_idx):
        q_x, q_y = get_x_y(q_idx)
        kv_x, kv_y = get_x_y(kv_idx)
        # q_x/kv_x index rows (height), q_y/kv_y index cols (width)
        kernel_x = q_x.clamp(K_H // 2, (H - 1) - K_H // 2)
        kernel_y = q_y.clamp(K_W // 2, (W - 1) - K_W // 2)
        hori_mask = (kernel_y - kv_y).abs() <= K_W // 2
        vert_mask = (kernel_x - kv_x).abs() <= K_H // 2
        return hori_mask & vert_mask
    
    def get_score_mod(score, b, h, q_idx, kv_idx):
        mask = get_block_mask(b,h,q_idx, kv_idx)
        return torch.where(mask, score, -float("inf"))
    
    return get_block_mask, get_score_mod

def mk_1d_flex_natten_mod(seq_len, kernel_size):
    
    def get_block_mask(b, h, q_idx, kv_idx):
        kernel_x = q_idx.clamp(kernel_size // 2, (seq_len-1) - kernel_size//2)
        mask = (kernel_x - kv_idx).abs() <= kernel_size//2
        return mask
    
    def get_score_mod(score, b, h, q_idx, kv_idx):
        mask = get_block_mask(b, h, q_idx, kv_idx)
        return torch.where(mask, score, -float("inf"))
    
    return get_block_mask, get_score_mod

def dict_to(d: Dict[str, Union[torch.Tensor, Any]], **to_kwargs) -> Dict[str, Union[torch.Tensor, Any]]:
    return {k: (v.to(**to_kwargs) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}

def sample_to_im(sample: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(sample, torch.Tensor):
        sample = ((sample.clip(-1, 1) / 2 + 0.5) * 255).round()
        sample = sample.detach().float().cpu().numpy().astype(np.uint8)
    else:
        sample = np.clip(sample, -1, 1)
        sample = ((sample / 2 + 0.5) * 255).round().astype(np.uint8)

    # Expected frame layout is HWC (channels-last). For grayscale C=1, return HxW for wandb.Image(mode="L").
    if sample.ndim == 3 and sample.shape[-1] == 1:
        sample = sample[..., 0]
    elif sample.ndim == 3 and sample.shape[0] == 1:
        # Legacy CHW grayscale fallback
        sample = sample[0]

    assert sample.ndim == 2, f"sample_to_im expects grayscale frame with shape [H,W,1] or [H,W], got {sample.shape=}"
    return sample


def sample_to_vid(sample: torch.Tensor | np.ndarray) -> np.ndarray:
    # Expected video layout is THWC (channels-last).
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu()
    assert sample.ndim in (3, 4), f"sample_to_vid expects [T,H,W,C] or [T,H,W], got {sample.shape=}"
    if sample.ndim == 3:
        # Already T,H,W
        return sample.numpy() if isinstance(sample, torch.Tensor) else sample
    return np.array([sample_to_im(sample[t]) for t in range(sample.shape[0])])

def compute_grid_dims(N: int):
    cols = math.ceil(np.sqrt(N))  # Number of columns (rounded up)
    rows = math.ceil(N / cols)    # Number of rows (adjusted based on columns)
    return rows, cols

def add_padding(imgs: list, padding: int = 10, mark_frames: list = [], mode: str = 'L') -> list:
    if mark_frames is None:
        mark_frames = []
    for i, im in enumerate(imgs):
        if mode == 'L':
            imgs[i] = np.pad(
                im,
                pad_width=((padding, padding), (padding, padding)),
                mode='constant',
                constant_values=255 if i not in mark_frames else 128,
            )
        elif mode == 'RGB':
            color = (255, 255, 255) if i not in mark_frames else (128, 0, 0)
            padded_img = np.ones(
                (im.shape[0], im.shape[1] + 2 * padding, im.shape[2] + 2 * padding),
                dtype=im.dtype
            ) * np.array(color, dtype=im.dtype)[:, None, None]

            padded_img[
                :,
                padding:-padding,
                padding:-padding,
            ] = im

            imgs[i] = padded_img
        else:
            raise NotImplementedError(f'Mode {mode} is not supported, choose L or RGB')
    return imgs

def pad_vid(video: np.array, padding: int=10, mark_frames: list=None, mode: str='L') -> np.array:
    if mark_frames is None:
        mark_frames = []
    if mode != 'L' and len(video.shape) == 3:
        video = np.repeat(video[:,  np.newaxis, :, :], 3, axis=1)
    frames = [video[i] for i in range(video.shape[0])]
    padded = add_padding(frames, padding, mark_frames, mode)
    return np.array(padded)

def get_grid_image(imgs: np.array, rows: int, cols: int, mode: str='L') -> np.array:
    grid = []
    for r in range(rows):
        row_images = imgs[r * cols:(r+1) * cols]

        if len(row_images) < cols:
            padding = [np.ones_like(imgs[0])*255 for _ in range(cols - len(row_images))]
            row_images += padding
        if mode == 'L':
            grid.append(np.concatenate(row_images, axis=1))
        else:
            grid.append(np.concatenate(row_images, axis=2))

    grid_image = np.concatenate(grid, axis=0) if mode == 'L' else np.concatenate(grid, axis=1)
    return grid_image

def get_grid_video(videos: list, rows, cols, mode: str='L') -> np.array:
    grid_imgs = []
    for i in range(videos[0].shape[0]):
        grid_frame = get_grid_image([v[i] for v in videos], rows, cols, mode)
        grid_imgs.append(grid_frame)
    return np.array(grid_imgs)

"""
Frame-wise Encoder Transformer
"""

class FreudEncoderLevel(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        depth: int,
        patch_size: int = 4,
        neighborhood_size: tuple[int, int] = None,
    ):
        super().__init__()

        self.neighborhood_size = neighborhood_size
        self.block_mask = None

        self.token_merge = TokenMerge2D(
            in_features=in_features,
            out_features=out_features,
            patch_size=patch_size,
        )
        self.transformer_layers = torch.nn.ModuleList([
            TransformerLayer(
                out_features,
                d_head=out_features // num_heads,
            ) for _ in range(depth)
        ])

    def init_block_mask(self, h: int, w: int, device: torch.device | None = None) -> None:
        if self.neighborhood_size is None or self.block_mask is not None:
            return
        image_size = (h, w)
        seq_len = h * w
        mask_mod, _ = mk_2d_flex_natten_mod(image_size, self.neighborhood_size)
        if device is None:
            device = torch.device("cpu")
        self.block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    
    def forward(self, x: Float[torch.Tensor, "B H W C"], pos: Float[torch.Tensor, "B H W C"]) -> Float[torch.Tensor, "B H' W' C'"]:
        x, pos = self.token_merge(x, pos)

        B, H, W, C = x.shape

        if self.neighborhood_size is not None and self.block_mask is None:
            self.init_block_mask(H, W, x.device)

        for layer in self.transformer_layers:
            x = einops.rearrange(x, "b h w c -> b (h w) c")
            pos = einops.rearrange(pos, "b h w c -> b (h w) c")
            x = layer(x, pos, block_mask=self.block_mask)
            x = einops.rearrange(x, "b (h w) c -> b h w c", h=H, w=W)
            pos = einops.rearrange(pos, "b (h w) c -> b h w c", h=H, w=W)
        return x, pos

class FreudEncoder(nn.Module):

    def __init__(
        self,
        in_features: int=1,
        out_features: int=4,
        widths: list=[96, 384],
        num_heads: list=[3, 6],
        depths: list=[2, 4],
        neighborhood_sizes: list = [(7, 7), None],
        patch_sizes: list=[4, 2],
    ):
        super().__init__()

        self.sub_modules = torch.nn.ModuleList()

        # construct downsampling transformer
        c_features = in_features
        for width, depth, patch_size, num_head, neighborhood_size in zip(widths, depths, patch_sizes, num_heads, neighborhood_sizes):
            self.sub_modules.append(
                FreudEncoderLevel(
                    in_features=c_features,
                    out_features=width,
                    num_heads=num_head,
                    depth=depth,
                    patch_size=patch_size,
                    neighborhood_size=neighborhood_size
                )
            )
            c_features = width

        self.out_proj = nn.Linear(c_features, out_features)

    def init_block_masks(self, h: int, w: int, device: torch.device | None = None) -> None:
        c_h, c_w = h, w
        for sm in self.sub_modules:
            c_h = c_h // sm.token_merge.ph
            c_w = c_w // sm.token_merge.pw
            sm.init_block_mask(c_h, c_w, device=device)

    def get_pos(self, x: Float[torch.Tensor, "B *DIM C"]) -> Float[torch.Tensor, "B *DIM C"]:
        # 2D pos
        B, H, W, _ = x.shape
        pos = make_axial_pos_2d(H, W, device=x.device).view(1, H, W, -1).expand(B, -1, -1, -1)
        return pos

    def forward(
        self,
        x: Float[torch.Tensor, "B H W C"]
    ) -> Float[torch.Tensor, "B H' W' C'"]:
        pos = self.get_pos(x)

        for sm in self.sub_modules:
            x, pos = sm(x, pos)

        x = self.out_proj(x)
        return x

"""
Unified Video Diffusion Transformer
"""

class FreudDecoderLevel(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        depth: int,
        d_cond_norm: int,
        merge: bool=False,
        d_cond_tokens: int = 4,
        temporal_patch_size: int = 1,
        patch_size: int = 4,
        neighborhood_size: tuple[int, int, int] = None,
        activate_temporal_attention: bool = True,
        factorized_attention_impl: str = "parallel",
    ):
        super().__init__()
        self.activate_temporal_attention = activate_temporal_attention

        if neighborhood_size is not None:
            self.temporal_neighborhood_size = neighborhood_size[0]
            self.spatial_neighborhood_size = neighborhood_size[1:]
        else:
            self.temporal_neighborhood_size = None
            self.spatial_neighborhood_size = None
        self.temporal_block_mask = None
        self.spatial_block_mask = None

        self.token_merge = None
        self.merge = merge
        if patch_size < 0:
            self.token_merge = TokenSplit3D(
                in_features=in_features,
                out_features=out_features,
                patch_size=(temporal_patch_size, -1*patch_size, -1*patch_size),
            )
        else:
            if self.merge:
                self.token_merge = CondTokenMerge3D(
                    in_features=in_features,
                    out_features=out_features,
                    cond_features=d_cond_tokens,
                    patch_size=(temporal_patch_size, patch_size, patch_size),
                )
            else:
                self.token_merge = TokenMerge3D(
                    in_features=in_features,
                    out_features=out_features,
                    patch_size=(temporal_patch_size, patch_size, patch_size),
                )

        self.factorized_transformer_layers = torch.nn.ModuleList([
            FactorizedAttentionLayer(
                out_features,
                d_head=out_features // num_heads,
                d_cond_norm=d_cond_norm,
                attention_impl=factorized_attention_impl,
            ) for _ in range(depth)
        ])

        if not self.activate_temporal_attention:
            for layer in self.factorized_transformer_layers:
                for name, param in layer.self_attn.named_parameters():
                    if name.startswith("temporal_") or name.startswith("scale_temporal"):
                        param.requires_grad_(False)

    def init_block_masks(self, t: int, h: int, w: int, device: torch.device | None = None) -> None:
        if device is None:
            device = torch.device("cpu")
        if self.temporal_neighborhood_size is not None and self.temporal_block_mask is None:
            mask_mod, _ = mk_1d_flex_natten_mod(t, self.temporal_neighborhood_size)
            self.temporal_block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=t, KV_LEN=t, device=device)
        if self.spatial_neighborhood_size is not None and self.spatial_block_mask is None:
            seq_len = h * w
            mask_mod, _ = mk_2d_flex_natten_mod((h, w), self.spatial_neighborhood_size)
            self.spatial_block_mask = create_block_mask(mask_mod, B=1, H=1, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

    def forward(
        self,
        x: Float[torch.Tensor, "B T H W C"],
        pos: Float[torch.Tensor, "B T H W C"],
        cond_norm: Float[torch.Tensor, "B D"],
        cond_tokens: Float[torch.Tensor, "B T H' W' C'"],
        skip: Float[torch.Tensor, "B T H' W' C"] | None = None,
    ) -> Float[torch.Tensor, "B T H' W' C'"]:
        x, pos = self.token_merge(x, pos, cond_tokens=cond_tokens, skip=skip)

        B, T, H, W, C = x.shape

        if (
            (self.temporal_neighborhood_size is not None and self.temporal_block_mask is None)
            or (self.spatial_neighborhood_size is not None and self.spatial_block_mask is None)
        ):
            self.init_block_masks(T, H, W, x.device)
    
        for layer in self.factorized_transformer_layers:
            x = layer(
                x=x,
                pos=pos,
                cond_norm=cond_norm,
                spatial_block_mask=self.spatial_block_mask,
                temporal_block_mask=self.temporal_block_mask,
                enable_temporal_attention=self.activate_temporal_attention,
            )

        return x, pos
    

class FreudDecoder(nn.Module):
    
    def __init__(
        self,
        in_features: int=1,
        widths: list=[96, 384, 96],
        num_heads: list=[3, 6, 3],
        depths: list=[2, 12, 2],
        merge_level: int = 1,
        neighborhood_sizes: list = [(3, 7, 7), None, (3, 7, 7)],
        patch_sizes: list=[4, 2, -2, -4],
        d_cond_norm: int = 384,
        latent_size: int=4,
        activate_temporal_attention: bool = True,
        factorized_attention_impl: str = "parallel",
    ):
        super().__init__()

        c_features = in_features
        self.sub_modules = torch.nn.ModuleList()
        self.is_downsampling_layer = []

        for i, (width, depth, patch_size, num_head, neighborhood_size) in enumerate(zip(widths, depths, patch_sizes[:-1], num_heads, neighborhood_sizes)):
            do_merge = i == merge_level
            self.sub_modules.append(
                FreudDecoderLevel(
                    in_features=c_features,
                    out_features=width,
                    num_heads=num_head,
                    depth=depth,
                    d_cond_norm=d_cond_norm,
                    merge=do_merge,
                    d_cond_tokens=latent_size if do_merge else 0,
                    temporal_patch_size=1,
                    patch_size=patch_size,
                    neighborhood_size=neighborhood_size,
                    activate_temporal_attention=activate_temporal_attention,
                    factorized_attention_impl=factorized_attention_impl,
                )
            )
            self.is_downsampling_layer.append(patch_size > 0)
            c_features = width

        # out proj unpatchifies the input
        self.out_proj = TokenSplitLast3D(
            in_features=c_features,
            out_features=in_features,
            patch_size=(1, -1*patch_sizes[-1], -1*patch_sizes[-1]),
        )

    def init_block_masks(self, t: int, h: int, w: int, device: torch.device | None = None) -> None:
        c_t, c_h, c_w = t, h, w
        for sm in self.sub_modules:
            tm = sm.token_merge
            if isinstance(tm, TokenSplit3D):
                c_t = c_t * tm.t
                c_h = c_h * tm.h
                c_w = c_w * tm.w
            else:
                c_t = c_t // tm.t
                c_h = c_h // tm.h
                c_w = c_w // tm.w
            sm.init_block_masks(c_t, c_h, c_w, device=device)

    def get_pos(self, x: Float[torch.Tensor, "B *DIM C"]) -> Float[torch.Tensor, "B *DIM C"]:
        # 3D pos
        B, T, H, W, _ = x.shape
        pos = make_axial_pos_3d(T, H, W, device=x.device).view(1, T, H, W, -1).expand(B, -1, -1, -1, -1)
        return pos

    def forward(
        self,
        x: Float[torch.Tensor, "B T H W C"],
        cond_norm: Float[torch.Tensor, "B D"],
        cond_tokens: Float[torch.Tensor, "B T H' W' C'"],
    ) -> Float[torch.Tensor, "B T H W C"]:
        pos = self.get_pos(x)
        
        skips = []
        skip_pos = []
        for i, sm in enumerate(self.sub_modules):
            if self.is_downsampling_layer[i]:
                skip = None
                pos_for_upsample = pos
            else:
                skip = skips.pop() if len(skips) > 0 else None
                pos = skip_pos.pop() if len(skip_pos) > 0 else pos
            x, pos = sm(x, pos, cond_norm, cond_tokens, skip=skip)
            if self.is_downsampling_layer[i]:
                skips.append(x)
                skip_pos.append(pos_for_upsample)

        return self.out_proj(x)


"""
FREUD Diffusion Autoencoder
with Tanh regularization and Outlier Punishment loss
"""

class FreudDiffusionAE(nn.Module):

    def __init__(
        self,
        encoder: FreudEncoder,
        decoder: FreudDecoder,
        mapping: MappingNetwork,
        img_size: tuple[int, int] = (384, 384),
        num_channels: int=1,
        num_timesteps: int=25,
        use_outlier_punishment: bool=True,
        outlier_threshold: float=3,
        outlier_punishment_weight: float=1e5,
        reg_weight: float=1e-3,
    ) -> None:
        super().__init__()

        self.unet = decoder
        self.mapping = mapping
        self.encoder = encoder

        self.img_size = img_size
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps

        self.use_outlier_punishment = use_outlier_punishment
        self.outlier_threshold = outlier_threshold
        self.outlier_punishment_weight = outlier_punishment_weight

        self.reg_weight = reg_weight

        mapping_width = getattr(mapping, "width", None)
        if mapping_width is None:
            # Fallback for MappingNetwork implementations that do not expose `.width`
            mapping_width = int(mapping.in_norm.scale.shape[-1])
        self.time_emb = FourierFeatures(1, mapping_width)
        self.time_in_proj = nn.Linear(mapping_width, mapping_width, bias=False)

    def init_block_masks(self, device: torch.device | None = None) -> None:
        if device is None:
            device = next(self.parameters()).device
        self.encoder.init_block_masks(self.img_size[0], self.img_size[1], device=device)
        self.unet.init_block_masks(self.num_timesteps, self.img_size[0], self.img_size[1], device=device)
    
    def encode(
        self,
        c_vid: Float[torch.Tensor, "B T H W C"],
        reg_weight: float = None,
        **kwargs
    ) -> Float[torch.Tensor, "B T H' W' C'"]:
        if c_vid.ndim != 5:
            c_vid = c_vid.unsqueeze(1) # add time dimension if not present
        assert c_vid.ndim == 5, f"c_vid must have shape [B T H W C] but got {c_vid.shape=}"

        _, t, _, _, _ = c_vid.shape
        if reg_weight is None:
            reg_weight = self.reg_weight
        
        # frame-wise encoding => merge batch and time dimensions, unmerge after encoding
        # print(f"[DEBUG] c_vid before rearrange: {c_vid.shape=}", flush=True)
        x = einops.rearrange(c_vid, "b t h w c -> (b t) h w c")
        # print(f"[DEBUG] c_vid after rearrange: {x.shape=}", flush=True)
        latent = self.encoder(x)
        # print(f"[DEBUG] latent before rearrange: {latent.shape=}", flush=True)
        latent = einops.rearrange(latent, "(b t) hl wl cl -> b t hl wl cl", b=c_vid.shape[0], t=t)
        # print(f"[DEBUG] latent after rearrange: {latent.shape=}", flush=True)

        # stochastic tanh regularization
        latent = torch.nn.functional.tanh(latent)
        latent = latent + reg_weight * torch.randn_like(latent)
        return latent

    def decode(
        self,
        latent: Float[torch.Tensor, "B T H' W' C'"],
        noise: Float[torch.Tensor, "B T H W C"] = None,
        sample_steps: int=50,
        **kwargs
    ) -> Float[torch.Tensor, "B T H W C"]:
        # decoding is achieved by sampling from the diffusion model conditioned on the latent representation
        if noise is None:
            noise = torch.randn(
                (
                    latent.shape[0],
                    latent.shape[1],
                    self.img_size[0],
                    self.img_size[1],
                    self.num_channels
                ), device=latent.device, dtype=latent.dtype
            )
        return self.sample(noise, c_latent=latent, sample_steps=sample_steps, **kwargs)

    def get_conditioning(
        self,
        t: Float[torch.Tensor, "B"],
        c_vid: Float[torch.Tensor, "B *DIMS C"] = None,
        c_latent: Float[torch.Tensor, "B *DIMS C"] = None,
        **data_kwargs
    ) -> dict[str, torch.Tensor]:
        # diffusion time embedding
        time_emb = self.time_in_proj(self.time_emb(t[..., None]))
        cond_dict = {"cond_norm": self.mapping(time_emb)}

        # get latent if not provided
        if c_latent is None:
            assert c_vid is not None, "Either c_vid or c_latent must be provided"
            c_latent = self.encode(c_vid=c_vid)
        if c_latent.ndim == 4:
            c_latent = c_latent.unsqueeze(1) # add time dimension
        assert c_latent.ndim == 5, f"c_latent must have shape [B T H W C] but got {c_latent.shape=}"
        cond_dict["cond_tokens"] = c_latent
        return cond_dict

    def lopl_2d(self, x: torch.Tensor, r: float=3, eps: float = 1e-6) -> torch.Tensor:
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sigma = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
        return torch.relu((x-mu).abs() - r*sigma).mean(dim=(-2, -1))

    def get_loss(
        self,
        x: Float[torch.Tensor, "B T H W C"],
        noise: Float[torch.Tensor, "B T H W C"],
        vtheta: Float[torch.Tensor, "B T H W C"],
        latent: Float[torch.Tensor, "B T H' W' C'"],
    ) -> Float[torch.Tensor, "B"]:
        # Recon is standard FM loss
        loss = ((noise - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))

        # Outlier punishment
        if self.use_outlier_punishment:
            z = latent
            _, z_t, _, _, z_c = z.size()
            z_opl = einops.rearrange(z, "b t h w c -> (b c t) h w")
            outlier_loss = torch.mean(
                torch.clamp(
                    torch.abs(torch.mean(z_opl, dim=(1, 2), keepdim=True) - z_opl)
                    - self.outlier_threshold * z_opl.std(dim=(1, 2), keepdim=True),
                    min=0,
                ),
                dim=(1, 2),
            )
            outlier_loss = einops.rearrange(outlier_loss, "(b c t) -> b c t", c=z_c, t=z_t)
            outlier_loss = torch.sum(outlier_loss, dim=(1, 2))
            loss = loss + self.outlier_punishment_weight * outlier_loss
        return loss

    def _get_zt(
        self,
        x: Float[torch.Tensor, "B ..."]
    ) -> Float[torch.Tensor, "B ..."]:
        # Rectified Flow: linear interpolation between noise and data, with t sampled uniformly from [0, 1]
        B = x.size(0)
        t = torch.rand((B,), device=x.device)
        texp = t.view([B, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        dtype = x.dtype
        zt, t = zt.to(dtype), t.to(dtype)
        return zt, t, z1

    def forward(
        self,
        x: Float[torch.Tensor, "B T H W C"],
        **data_kwargs
    ) -> Float[torch.Tensor, "B"]:
        # forward returns loss for training
        zt, t, z1 = self._get_zt(x)
        cond = self.get_conditioning(t, c_vid=x, **data_kwargs)
        vtheta = self.unet(zt, **cond)
        return self.get_loss(x, z1, vtheta, cond["cond_tokens"])

    @torch.no_grad()
    def sample(
        self,
        noise: Float[torch.Tensor, "B T H W C"],
        c_latent: Float[torch.Tensor, "B T H' W' C'"] = None,
        c_vid: Float[torch.Tensor, "B T H W C"] = None,
        sample_steps: int=50,
        **kwargs
    ):
        if c_latent is None:
            assert c_vid is not None, "Either c_latent or c_vid must be provided for sampling"
            c_latent = self.encode(c_vid=c_vid)
        
        # rectified flow sampling with timesteps in [0, 1] range
        dt = 1.0 / sample_steps
        dt = torch.tensor(dt, device=noise.device, dtype=noise.dtype)
        dt = dt * torch.ones((noise.shape[0],), device=noise.device, dtype=noise.dtype)
        repeat_string = "B -> B" + " 1" * (len(noise.shape) - 1)  # repeat for all dimensions except batch
        dt = einops.repeat(dt, repeat_string, B=noise.shape[0])

        # Initialize sample with noise
        x = noise

        # Rectified Flow Euler sampling loop
        for i in range(sample_steps, 0, -1):
            t = float(i) / sample_steps
            t = torch.tensor([t] * noise.shape[0], device=noise.device, dtype=noise.dtype)
            cond_dict = self.get_conditioning(t, c_latent=c_latent)
            vtheta = self.unet(x, **cond_dict)
            x = x - dt * vtheta
        return x

    @torch.no_grad()
    def validate(
        self,
        dataloader_val: "torch.utils.data.DataLoader",
        global_rank: int,
        max_steps: int | None,
        device,
    ):
        """Logs max_steps frames and max_steps videos to wandb"""
        if global_rank != 0:  # only perform validation on 0-rank GPU
            return
        
        # generate samples
        samples = []
        conditions = []

        with tqdm(desc="Validating", disable=(global_rank != 0), total=max_steps) as pbar:
            for batch in dataloader_val:
                batch = dict_to(batch, device=device)
                batch = batch['x']
                batch = batch[0:min(batch.size(0), max_steps - len(samples))]

                # do validation
                sample_tensor = torch.randn_like(batch)
                res = self.sample(sample_tensor, c_vid=batch)
                samples += [res[i].cpu() for i in range(res.size(0))]
                conditions += [batch[i].cpu() for i in range(batch.size(0))]

                pbar.update(batch.size(0))

                if len(samples) >= max_steps:
                    break
        
        samples = torch.stack(samples)
        conditions = torch.stack(conditions)

        # extract and log images
        image_samples = [sample_to_im(x[0]) for x in samples]
        image_conditions = [sample_to_im(x[0]) for x in conditions]
        
        combined_images = []
        for i, im_sample in enumerate(image_samples):
            # channels as horizontal dim, top row: conditions, bottom row: samples
            combined = np.vstack([image_conditions[i], im_sample])
            combined_images.append(combined)
        
        num_images = len(combined_images)
        rows, cols = compute_grid_dims(num_images)
        combined_images = add_padding(combined_images)
        grid_image = get_grid_image(combined_images, rows, cols)

        rmse = torch.sqrt(torch.mean((conditions - samples)**2)).item()

        log_dict = {
            "Val/Vis_Img/sample": wandb.Image(grid_image, mode='L'),
            "Val/rmse": rmse,
        }

        # log video only if there is a temporal dimension larger than 1
        if samples.shape[1] > 1:
            sample_videos = [sample_to_vid(x) for x in samples]
            condition_videos = [sample_to_vid(x) for x in conditions]

            combined_videos = []
            for i, vid_sample in enumerate(sample_videos):
                # For THW videos, concatenate along width
                combined = np.concatenate([condition_videos[i], vid_sample], axis=2)
                combined = pad_vid(combined, 5)
                combined_videos.append(combined)

            num_videos = len(combined_videos)
            rows, cols = compute_grid_dims(num_videos)
            reel = get_grid_video(combined_videos, rows, cols)
            reel = np.repeat(reel[:, np.newaxis, :, :], 3, axis=1)  # ensure video has 3 channels for wandb
            log_dict["Val/Vis_Vid/reel"] = wandb.Video(reel, fps=5, format="mp4")

        wandb.log(log_dict)
        
"""
Inference Helpers
"""

@torch.inference_mode()
def get_recon(
    model: FreudDiffusionAE,
    x: Float[torch.Tensor, "B T H W C"],
    decoding_sample_steps: int=10,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
) -> Float[torch.Tensor, "B T H W C"]:
    with torch.autocast(device_type=device.type, dtype=dtype):
        latent = model.encode(x)
        recon = model.decode(latent, sample_steps=decoding_sample_steps)
    return recon, latent

def get_freud_model(
    ckpt_path: str | None = None,
) -> FreudDiffusionAE:
    model =  FreudDiffusionAE(
        img_size=(384, 384),
        num_channels=1,
        num_timesteps=25,
        use_outlier_punishment=True,
        outlier_threshold=3,
        outlier_punishment_weight=1e5,
        reg_weight=1e-3,
        mapping=MappingNetwork(
            n_layers=2,
            d_model=384,
            d_ff = 384*3,
            dropout=0.0,
        ),
        encoder=FreudEncoder(
            in_features=1,
            out_features=4,
            widths=[96, 384],
            num_heads=[3, 6],
            depths=[2, 4],
            neighborhood_sizes=[(7, 7), None],
            patch_sizes=[4, 2],
        ),
        decoder=FreudDecoder(
            in_features=1,
            latent_size=4,
            d_cond_norm=384,
            widths=[96, 384, 96],
            num_heads=[3, 6, 3],
            depths=[2, 12, 2],
            merge_level=1,
            neighborhood_sizes=[(3, 7, 7), None, (3, 7, 7)],
            patch_sizes=[4, 2, -2, -4],
            factorized_attention_impl="sequential",
            activate_temporal_attention=True,
        )
    )

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model

