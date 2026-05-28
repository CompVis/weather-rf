from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Union
import random

import einops
import torch
from jaxtyping import Float
from scipy.optimize import linear_sum_assignment
from torch import nn

from .freud import (
    FourierFeatures,
    GenericTransformerLayer,
    MappingNetwork,
    TokenMerge3D,
    TokenSplitLast3D,
    Transformer,
    load_freud_dit_small_p4_rain,
    make_axial_pos_3d,
)


class DiffusionScheduleBase(Protocol):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class MappingSpec:
    depth: int = 3
    width: int = 512
    d_ff: int = 1536
    dropout: float = 0.0


class RF(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        mapping: MappingSpec | MappingNetwork,
        train_timestep_sampling: Literal["logit_sigmoid", "uniform"] = "logit_sigmoid",
        time_cond_type: Literal["sigma", "rf_t"] = "rf_t",
        immiscible: bool = False,
        noise_type: Literal["gaussian", "heavy_tail", "forecast"] = "gaussian",
        heavy_tail_dof: float = 3.0,
        compile_mapping: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.train_timestep_sampling = train_timestep_sampling
        self.immiscible = immiscible
        self.compile_mapping = compile_mapping

        self.noise_type = noise_type
        self.heavy_tail_dof = heavy_tail_dof
        if noise_type == "heavy_tail":
            self.chi2 = torch.distributions.chi2.Chi2(torch.tensor([heavy_tail_dof]))

        self.mapping = mapping if isinstance(mapping, MappingNetwork) else MappingNetwork(mapping.width, mapping.depth, mapping.d_ff)
        self.time_emb = FourierFeatures(1, self.mapping.width)
        self.time_in_proj = nn.Linear(self.mapping.width, self.mapping.width, bias=False)
        self.time_cond_type = time_cond_type

        self.compile_self = kwargs.pop("compile", True)
        self.compile_backend = kwargs.pop("compile_backend", "inductor")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        raise NotImplementedError

    def compile(self) -> None:
        if self.compile_mapping and hasattr(self.mapping, "compile"):
            self.mapping.compile()
        if self.compile_self and hasattr(self.unet, "compile"):
            self.unet.compile()

    def get_conditioning(self, t: Float[torch.Tensor, "b"], **kwargs: Any) -> dict[str, torch.Tensor]:
        if self.time_cond_type == "sigma":
            c_noise = torch.log(t) / 4
        elif self.time_cond_type == "rf_t":
            c_noise = t
        else:
            raise NotImplementedError(f'Unknown time conditioning type "{self.time_cond_type}".')

        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        cond_time = self.mapping(time_emb)
        return {"cond_norm": cond_time}

    def _get_noise(self, x: Float[torch.Tensor, "b ..."]) -> Float[torch.Tensor, "b ..."]:
        if self.noise_type == "gaussian" or self.noise_type == "forecast":
            return torch.randn_like(x)
        if self.noise_type == "heavy_tail":
            gauss_sample = torch.randn_like(x).to(x.device).to(x.dtype)
            chi2_sample = torch.sqrt(self.chi2.sample(x.size()).squeeze(-1) / self.heavy_tail_dof).to(x.device).to(x.dtype)
            return gauss_sample / chi2_sample
        raise NotImplementedError(f"Noise type {self.noise_type} is not implemented")

    def _get_zt(self, x: Float[torch.Tensor, "b ..."]):
        b = x.size(0)
        if self.train_timestep_sampling == "logit_sigmoid":
            t = torch.sigmoid(torch.randn((b,), device=x.device))
        elif self.train_timestep_sampling == "uniform":
            t = torch.rand((b,), device=x.device)
        else:
            raise ValueError(f'Unknown train timestep sampling method "{self.train_timestep_sampling}".')
        texp = t.view([b, *([1] * len(x.shape[1:]))])

        z1 = self._get_noise(x)
        if self.immiscible:
            cost = torch.cdist(x.flatten(1), z1.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.float().cpu())
            z1 = z1[torch.from_numpy(reorder_indices).to(cost.device)]

        zt = (1 - texp) * x + texp * z1
        return zt.to(x.dtype), t.to(x.dtype), z1

    def get_logits(self, x: Float[torch.Tensor, "b ..."], **data_kwargs):
        zt, t, z1 = self._get_zt(x)
        cond_dict = self.get_conditioning(t, **data_kwargs)
        pos = self.get_pos(zt)
        vtheta = self.unet(zt, pos=pos, **cond_dict)
        return vtheta, z1, cond_dict

    def get_loss(
        self,
        x: Float[torch.Tensor, "b ..."],
        noise: Float[torch.Tensor, "b ..."],
        vtheta: Float[torch.Tensor, "b ..."],
        cond_dict: dict,
    ) -> Float[torch.Tensor, "b"]:
        return ((noise - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))

    def forward(self, x: Float[torch.Tensor, "b ..."], **data_kwargs) -> Float[torch.Tensor, "b"]:
        vtheta, z1, cond_dict = self.get_logits(x, **data_kwargs)
        return self.get_loss(x, z1, vtheta, cond_dict)

    @torch.no_grad()
    def sample(
        self,
        z: Float[torch.Tensor, "b c ..."],
        sample_steps: int = 50,
        return_list: bool = False,
        diffusion_schedule: DiffusionScheduleBase | None = None,
        **data_kwargs,
    ) -> Union[Float[torch.Tensor, "b ..."], list[Float[torch.Tensor, "b ..."]]]:
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b, device=z.device, dtype=z.dtype).view([b, *([1] * len(z.shape[1:]))])
        if return_list:
            images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b, device=z.device, dtype=z.dtype)

            cond_dict = self.get_conditioning(t, **data_kwargs)
            pos = self.get_pos(z)
            vc = self.unet(z, pos=pos, **cond_dict)

            if diffusion_schedule is None:
                z = z - dt * vc
            else:
                sigma_t = 1 - t + 1e-5
                score = (t * vc - z) / sigma_t
                gt = diffusion_schedule(t)
                z = z - dt * (vc - 0.5 * (gt**2) * score) + gt * torch.sqrt(dt) * torch.randn_like(z)

            if return_list:
                images.append(z)

        if return_list:
            return images
        return z


class LatentRF3D(RF):
    def __init__(
        self,
        unet: nn.Module,
        mapping: MappingSpec | MappingNetwork,
        first_stage: nn.Module,
        train_timestep_sampling: Literal["logit_sigmoid", "uniform"] = "logit_sigmoid",
        time_cond_type: Literal["sigma", "rf_t"] = "rf_t",
        immiscible: bool = False,
        fs_is_framewise: bool = False,
        encoded_data: bool = False,
        validation_fps: int = 3,
        image_pretraining_mode: bool = False,
        only_train_temporal_layers: bool = False,
        noise_type: Literal["gaussian", "heavy_tail", "forecast"] = "gaussian",
        heavy_tail_dof: float = 3.0,
        orig_image_size: tuple[int, int] = (128, 128),
        orig_num_channels: int = 10,
        orig_num_timesteps: int = 18,
        m_norm: float = 0.0,
        std_norm: float = 1.0,
        vae_dtype: str = "bfloat16",
        fs_batch_size: int | None = None,
        vae_range: tuple[float, float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            unet=unet,
            mapping=mapping,
            train_timestep_sampling=train_timestep_sampling,
            time_cond_type=time_cond_type,
            immiscible=immiscible,
            noise_type=noise_type,
            heavy_tail_dof=heavy_tail_dof,
            **kwargs,
        )

        self.first_stage = first_stage
        self.fs_is_framewise = fs_is_framewise
        self.encoded_data = encoded_data
        self.m_norm = m_norm
        self.std_norm = std_norm
        self.validation_fps = validation_fps
        self.image_size = orig_image_size
        self.num_channels = orig_num_channels
        self.num_timesteps = orig_num_timesteps if not image_pretraining_mode else 1
        self.fs_bs = fs_batch_size
        self.vae_range = vae_range
        self.vae_dtype = torch.float32 if vae_dtype == "float32" else torch.bfloat16

        for param in first_stage.parameters():
            param.requires_grad = False
        first_stage.eval()

        self.image_pretraining_mode = image_pretraining_mode
        if only_train_temporal_layers:
            for name, param in self.unet.named_parameters():
                if "temporal" not in name:
                    param.requires_grad = False

    def compile(self) -> None:
        super().compile()
        if hasattr(self.first_stage, "compile"):
            self.first_stage.compile()

    @torch.no_grad()
    def _encode(self, x: Float[torch.Tensor, "b c t h w"]) -> Float[torch.Tensor, "b c t h w"]:
        orig_dtype = x.dtype
        if self.encoded_data:
            latent = x.detach().clone()
        else:
            bs, ch, t, h, w = x.size()
            if self.fs_is_framewise:
                x = einops.rearrange(x, "b c t h w -> (b t) c h w")

            with torch.autocast(device_type=self.device.type, dtype=self.vae_dtype):
                x = x.to(self.vae_dtype)

                if self.vae_range is not None:
                    x = (x + 1) / 2
                    x *= (self.vae_range[1] - self.vae_range[0]) + self.vae_range[0]

                self.fs_bs = x.size(0) if self.fs_bs is None else self.fs_bs
                latent_chunks = []
                for i in range(0, x.size(0), self.fs_bs):
                    x_chunk = x[i : i + self.fs_bs]
                    latent_chunks.append(self.first_stage.encode(x_chunk))
                latent = torch.cat(latent_chunks, dim=0)

            if self.fs_is_framewise:
                latent = einops.rearrange(latent, "(b t) c h w -> b c t h w", t=t)

        latent = latent.to(orig_dtype)
        latent = (latent - self.m_norm) / self.std_norm
        return latent

    @torch.no_grad()
    def _decode(
        self,
        latent: Union[Float[torch.Tensor, "b ..."], list[Float[torch.Tensor, "b ..."]]],
        sample_steps: int = 50,
    ) -> Union[Float[torch.Tensor, "b ..."], list[Float[torch.Tensor, "b ..."]]]:
        if isinstance(latent, list):
            latents = [l * self.std_norm + self.m_norm for l in latent]
        else:
            latents = [latent * self.std_norm + self.m_norm]

        if not self.fs_is_framewise:
            aligned_latents = []
            for l in latents:
                if l.ndim != 5:
                    raise ValueError(f"Expected 5D latent tensor, got shape {tuple(l.shape)}")
                if l.size(2) == self.num_timesteps:
                    # Already in B C T H W layout.
                    aligned_latents.append(l)
                elif l.size(1) == self.num_timesteps:
                    # Convert B T H W C -> B C T H W.
                    aligned_latents.append(l.movedim(-1, 1))
                else:
                    raise ValueError(
                        f"Could not infer latent layout from shape {tuple(l.shape)} for num_timesteps={self.num_timesteps}"
                    )
            latents = aligned_latents

        decoded = []
        for l in latents:
            bs, ch, t, h, w = l.size()
            original_dtype = l.dtype

            if self.fs_is_framewise:
                l = einops.rearrange(l, "b c t h w -> (b t) c h w")
                with torch.autocast(device_type=self.device.type, dtype=self.vae_dtype):
                    l = l.to(self.vae_dtype)
                    self.fs_bs = l.size(0) if self.fs_bs is None else self.fs_bs
                    x_chunks = []
                    for i in range(0, l.size(0), self.fs_bs):
                        l_chunk = l[i : i + self.fs_bs]
                        x_chunks.append(
                            self.first_stage.decode(
                                l_chunk,
                                image_size=self.image_size,
                                num_channels=self.num_channels,
                                sample_steps=sample_steps,
                            )
                        )
                    x = torch.cat(x_chunks, dim=0)
                x = einops.rearrange(x, "(b t) c h w -> b c t h w", t=t)
                x = x.to(l.dtype)
            else:
                with torch.autocast(device_type=self.device.type, dtype=self.vae_dtype):
                    l = l.to(self.vae_dtype)
                    self.fs_bs = l.size(0) if self.fs_bs is None else self.fs_bs
                    x_chunks = []
                    for i in range(0, l.size(0), self.fs_bs):
                        l_chunk = l[i : i + self.fs_bs]
                        x_chunks.append(
                            self.first_stage.decode(
                                l_chunk,
                                image_size=self.image_size,
                                num_channels=self.num_channels,
                                num_timesteps=self.num_timesteps,
                                sample_steps=sample_steps,
                            )
                        )
                    x = torch.cat(x_chunks, dim=0)
                x = x.to(original_dtype)

            decoded.append(x)

        if self.vae_range is not None:
            decoded = [(x - self.vae_range[0]) / (self.vae_range[1] - self.vae_range[0]) * 2 - 1 for x in decoded]

        if isinstance(latent, list):
            return decoded
        return decoded[-1]

    def get_pos(self, x: Float[torch.Tensor, "B C *DIM"]) -> Float[torch.Tensor, "B *DIM c"]:
        b, _, *dims = x.shape
        pos = make_axial_pos_3d(*dims, device=x.device).view(1, *dims, -1).expand(b, -1, -1, -1, -1)
        return pos.movedim(-1, 1)

    def sample(
        self,
        z: Float[torch.Tensor, "b c ..."],
        sample_steps: int = 50,
        return_list: bool = False,
        **data_kwargs,
    ) -> Union[Float[torch.Tensor, "b ..."], list[Float[torch.Tensor, "b ..."]]]:
        latent = super().sample(z, sample_steps=sample_steps, return_list=return_list, **data_kwargs)
        return self._decode(latent, sample_steps=10)

    def forward(self, x: Float[torch.Tensor, "b c t h w"], **data_kwargs) -> Float[torch.Tensor, "b"]:
        if self.image_pretraining_mode:
            x = x.unsqueeze(2)
        x = self._encode(x)
        vtheta, z1, cond_dict = self.get_logits(x, **data_kwargs)
        return self.get_loss(x, z1, vtheta, cond_dict)


class LatentRF3DRamvid(LatentRF3D):
    def __init__(
        self,
        unet: nn.Module,
        mapping: MappingSpec | MappingNetwork,
        first_stage: nn.Module,
        train_timestep_sampling: Literal["logit_sigmoid", "uniform"] = "logit_sigmoid",
        time_cond_type: Literal["sigma", "rf_t"] = "rf_t",
        immiscible: bool = False,
        fs_is_framewise: bool = False,
        encoded_data: bool = False,
        prob_unconditional: float = 0.25,
        max_conditional_frames: int = 3,
        validation_fps: int = 3,
        image_pretraining_mode: bool = False,
        only_train_temporal_layers: bool = False,
        only_prediction: bool = False,
        sigma_min: float = 0.0,
        sigma: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            unet=unet,
            mapping=mapping,
            first_stage=first_stage,
            train_timestep_sampling=train_timestep_sampling,
            time_cond_type=time_cond_type,
            immiscible=immiscible,
            fs_is_framewise=fs_is_framewise,
            encoded_data=encoded_data,
            validation_fps=validation_fps,
            image_pretraining_mode=image_pretraining_mode,
            only_train_temporal_layers=only_train_temporal_layers,
            **kwargs,
        )

        self.prob_unconditional = prob_unconditional
        self.max_conditional_frames = max_conditional_frames
        self.only_prediction = only_prediction
        self.sigma_min = sigma_min
        self.sigma = sigma

    def _get_conditional_idx(self, x: Float[torch.Tensor, "b c t h w"]) -> list[list[int]]:
        bs = x.size(0)
        conditional_idx: list[list[int]] = []
        max_cond = max(1, min(self.max_conditional_frames, x.size(2) - 1))

        if not self.only_prediction:
            unconditional = torch.rand(bs) < self.prob_unconditional
            num_cond_frames = torch.randint(0, max_cond, (bs,))
            for i in range(bs):
                if unconditional[i]:
                    conditional_idx.append([])
                    continue
                conditional_idx.append(random.sample(range(x.size(2)), num_cond_frames[i].item()))
        else:
            for _ in range(bs):
                conditional_idx.append(list(range(max_cond)))

        return conditional_idx

    @staticmethod
    def _mask_indices_inplace(x: torch.Tensor, conditional_idx: list[list[int]], reference: torch.Tensor | None = None) -> None:
        for i, indices in enumerate(conditional_idx):
            if reference is None:
                x[i, :, indices, :, :] = 0
            else:
                x[i, :, indices, :, :] = reference[i, :, indices, :, :]

    @torch.no_grad()
    def sample(
        self,
        z: Float[torch.Tensor, "b c ..."],
        conditional_idx: list[list[int]] | None = None,
        sample_steps: int = 50,
        decode_sample_steps: int = 50,
        return_list: bool = False,
        diffusion_schedule: DiffusionScheduleBase | None = None,
        **data_kwargs,
    ) -> Union[Float[torch.Tensor, "b ..."], list[Float[torch.Tensor, "b ..."]]]:
        assert conditional_idx is None or len(conditional_idx) == z.size(0), "Give conditional indices for all elements in the batch or give None for unconditional sampling"

        if self.image_pretraining_mode:
            return super().sample(z, sample_steps, return_list, **data_kwargs)

        if conditional_idx is None:
            conditional_idx = [[] * z.size(0)]

        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b, device=z.device, dtype=z.dtype).view([b, *([1] * len(z.shape[1:]))])
        if return_list:
            images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b, device=z.device, dtype=z.dtype)

            cond_dict = self.get_conditioning(t, **data_kwargs)
            pos = self.get_pos(z)
            vc = self.unet(z, pos=pos, **cond_dict)

            self._mask_indices_inplace(vc, conditional_idx)

            if diffusion_schedule is None:
                z = z - dt * vc
            else:
                sigma_t = 1 - t + 1e-5
                score = (t * vc - z) / sigma_t
                gt = diffusion_schedule(t)

                noise = gt * torch.sqrt(dt) * torch.randn_like(z)
                self._mask_indices_inplace(noise, conditional_idx)
                z = z - dt * (vc - 0.5 * (gt**2) * score) + noise

            if return_list:
                images.append(z)

        if return_list:
            return self._decode(images, sample_steps=decode_sample_steps)
        return self._decode(z, sample_steps=decode_sample_steps)

    @staticmethod
    def _get_next_neighbor_index(indices: list[int]) -> list[int]:
        s = set(indices)
        output = []

        for x in indices:
            found = False
            for y in range(x - 1, -1, -1):
                if y not in s:
                    output.append(y)
                    found = True
                    break

            if not found:
                y = x + 1
                while True:
                    if y not in s:
                        output.append(y)
                        break
                    y += 1

        return output

    def get_z1(self, x: torch.Tensor, conditional_idx: list[list[int]], texp: torch.Tensor) -> torch.Tensor:
        if self.noise_type != "forecast":
            z1 = self._get_noise(x)
            self._mask_indices_inplace(z1, conditional_idx, reference=x)
            return z1

        added_noise = torch.randn_like(x) * (self.sigma_min + self.sigma * texp * (1 - texp))
        z1 = x.clone()
        for i, indices in enumerate(conditional_idx):
            selected_indices = self._get_next_neighbor_index(indices)
            z1[i, :, indices, :, :] = x[i, :, selected_indices, :, :] + added_noise[i, :, indices, :, :]
        return z1

    def get_logits(self, x: Float[torch.Tensor, "b ..."], **data_kwargs):
        if self.image_pretraining_mode:
            return super().get_logits(x, **data_kwargs)

        b = x.size(0)
        if self.train_timestep_sampling == "logit_sigmoid":
            t = torch.sigmoid(torch.randn((b,), device=x.device))
        elif self.train_timestep_sampling == "uniform":
            t = torch.rand((b,), device=x.device)
        else:
            raise ValueError(f'Unknown train timestep sampling method "{self.train_timestep_sampling}".')
        texp = t.view([b, *([1] * len(x.shape[1:]))])

        conditional_idx = self._get_conditional_idx(x)
        z1 = self.get_z1(x, conditional_idx, texp)

        if self.immiscible:
            cost = torch.cdist(x.flatten(1), z1.flatten(1))
            _, reorder_indices = linear_sum_assignment(cost.float().cpu())
            z1 = z1[torch.from_numpy(reorder_indices).to(cost.device)]

        zt = (1 - texp) * x + texp * z1
        zt, t = zt.to(x.dtype), t.to(x.dtype)

        cond_dict = self.get_conditioning(t, **data_kwargs)
        pos = self.get_pos(zt)
        vtheta = self.unet(zt, pos=pos, **cond_dict)

        self._mask_indices_inplace(vtheta, conditional_idx)
        return vtheta, z1, cond_dict

    def forward(self, x: Float[torch.Tensor, "b c t h w"], **data_kwargs) -> Float[torch.Tensor, "b"]:
        if self.image_pretraining_mode:
            return super().forward(x, **data_kwargs)
        x = self._encode(x)
        vtheta, z1, cond_dict = self.get_logits(x, **data_kwargs)
        return self.get_loss(x, z1, vtheta, cond_dict)

"""
Inference Helpers and Utilities
"""

def _strip_compiled_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "") if "_orig_mod" in key and key.replace("._orig_mod", "") not in state_dict else key
        cleaned[new_key] = value
    return cleaned


def load_rf_ramvid_dit_L(
    checkpoint_path: str | None = None,
    first_stage_checkpoint_path: str | None = None,
    img_size: tuple[int, int] = (384, 384),
    num_channels: int = 1,
    num_timesteps: int = 25,
    m_norm: float = 0.0295775,
    std_norm: float = 0.508012,
) -> LatentRF3DRamvid:
    first_stage = load_freud_dit_small_p4_rain(
        checkpoint_path=first_stage_checkpoint_path,
        img_size=img_size,
        num_channels=num_channels,
        num_timesteps=num_timesteps,
    )

    mapping_width = 512
    latent_channels = first_stage.encoder.mid_split.proj.out_features

    unet = Transformer(
        down_merges=[],
        down_levels=[],
        mid_merge=TokenMerge3D(in_features=latent_channels, out_features=1024, patch_size=(1, 2, 2)),
        mid_level=nn.ModuleList(
            [
                GenericTransformerLayer(
                    d_model=1024,
                    d_head=64,
                    d_cond_norm=mapping_width,
                    use_factorized_attention=True,
                    pos_enc_cls="AxialRoPE3D",
                )
                for _ in range(24)
            ]
        ),
        mid_split=TokenSplitLast3D(in_features=1024, out_features=latent_channels, patch_size=(1, 2, 2)),
    )

    model = LatentRF3DRamvid(
        unet=unet,
        mapping=MappingNetwork(d_model=mapping_width, n_layers=3, d_ff=mapping_width * 3),
        first_stage=first_stage,
        prob_unconditional=0.25,
        encoded_data=False,
        fs_is_framewise=False,
        max_conditional_frames=13,
        validation_fps=5,
        time_cond_type="rf_t",
        immiscible=False,
        orig_image_size=img_size,
        orig_num_channels=num_channels,
        orig_num_timesteps=num_timesteps,
        m_norm=m_norm,
        std_norm=std_norm,
        image_pretraining_mode=False,
        only_train_temporal_layers=False,
        train_timestep_sampling="uniform",
    )

    if checkpoint_path is None:
        return model

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_compiled_prefixes(state_dict)
    model.load_state_dict(state_dict, strict=True)
    return model

@torch.inference_mode()
@torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16)
def get_pred(
    model: LatentRF3DRamvid,
    sample: Float[torch.Tensor, "B T H W C"],
    cond_indices: Float[torch.Tensor, "B T"],
    sample_steps: int = 25,
    decoder_sample_steps: int = 10,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    decode_chunk_size: int = 4,
) -> Float[torch.Tensor, "B T H W C"]:
    cond_indices = cond_indices.bool().to(device)
    sample = sample.to(device)

    if sample.shape[1] == cond_indices.shape[1]:
        # Input follows B T H W C layout.
        sample = einops.rearrange(sample, "b t h w c -> b c t h w")
    elif sample.shape[2] != cond_indices.shape[1]:
        raise ValueError(
            f"Could not align cond_indices with sample time axis: {sample.shape=}, {cond_indices.shape=}"
        )

    conditional_idx = [torch.where(cond_indices[i])[0].tolist() for i in range(cond_indices.size(0))]

    with torch.autocast(device_type=device.type, dtype=dtype):
        sample = sample.to(dtype)
        encoded = model._encode(sample)
        noise = model._get_noise(torch.ones_like(encoded))

        for i, indices in enumerate(conditional_idx):
            if len(indices) > 0:
                noise[i, :, indices, :, :] = encoded[i, :, indices, :, :]

        preds = []
        for start in range(0, noise.size(0), decode_chunk_size):
            end = min(start + decode_chunk_size, noise.size(0))
            preds.append(
                model.sample(
                    z=noise[start:end],
                    conditional_idx=conditional_idx[start:end],
                    sample_steps=sample_steps,
                    decode_sample_steps=decoder_sample_steps,
                )
            )
        pred = torch.cat(preds, dim=0)

    pred = pred.float().clamp(-1, 1)
    pred = einops.rearrange(pred, "b c t h w -> b t h w c")
    return pred
