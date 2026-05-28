import torch
import einops
import wandb
import numpy as np

from torch import nn
from tqdm import tqdm
from torch.distributed.device_mesh import DeviceMesh
from jaxtyping import Float, Bool
from properscoring import crps_ensemble
from .transformer import *

from .freud import FreudDiffusionAE, FreudEncoder, FreudDecoder, dict_to, sample_to_im, sample_to_vid, compute_grid_dims, add_padding, get_grid_image, get_grid_video, pad_vid


"""
Transformer implementation for Latent Space Forecasting model
"""

class LSMLevel(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int,
        num_heads: int,
        patch_size: tuple[int, int, int],
        d_cond_norm: int,
        activate_temporal_attention: bool,
        factorized_attention_impl: str
    ):
        super().__init__()
        self.activate_temporal_attention = activate_temporal_attention
        self.width = out_features

        self.token_merge = TokenMerge3D(
            in_features=in_features,
            out_features=out_features,
            patch_size=patch_size
        )

        self.factorized_transformer_layers = torch.nn.ModuleList([
            FactorizedAttentionLayer(
                self.width,
                d_head=self.width // num_heads,
                d_cond_norm=d_cond_norm,
                attention_impl=factorized_attention_impl
            ) for _ in range(depth)
        ])

        if not self.activate_temporal_attention:
            for layer in self.factorized_transformer_layers:
                for name, param in layer.self_attn.named_parameters():
                    if name.startswith("temporal_") or name.startswith("scale_temporal"):
                        param.requires_grad_(False)


    def forward(
        self,
        x: Float[torch.Tensor, "B T H W C"],
        pos: Float[torch.Tensor, "B T H W C"],
        cond_norm: Float[torch.Tensor, "B *COND D"]
    ):
        x, pos = self.token_merge(x, pos)
        _, T, H, W, _ = x.shape
        cond_norm = self._align_cond_norm(cond_norm, T=T, H=H, W=W)

        for layer in self.factorized_transformer_layers:
            x = layer(
                x=x,
                pos=pos,
                cond_norm=cond_norm,
                enable_temporal_attention=self.activate_temporal_attention,
            )
        return x, pos

    def _align_cond_norm(
        self,
        cond_norm: Float[torch.Tensor, "B *COND D"],
        T: int,
        H: int,
        W: int,
    ) -> Float[torch.Tensor, "B THW D"]:
        # AdaRMSNorm in jutils expects cond sequence length to match x sequence length
        # (or be singleton). For factorized layers, x is flattened to [B, T*H*W, C].
        B = cond_norm.shape[0]
        target_tokens = T * H * W

        if cond_norm.ndim == 2:
            return cond_norm[:, None, :].expand(B, target_tokens, -1)

        if cond_norm.ndim >= 4:
            assert all(
                dim == 1 for dim in cond_norm.shape[2:-1]
            ), f"Expected singleton middle dims in cond_norm, got {cond_norm.shape=}"
            cond_norm = cond_norm.reshape(B, cond_norm.shape[1], cond_norm.shape[-1])

        assert cond_norm.ndim == 3, f"Expected cond_norm with 2 or 3 dims after normalization, got {cond_norm.shape=}"

        if cond_norm.shape[1] == target_tokens:
            return cond_norm
        if cond_norm.shape[1] == T:
            cond_norm = cond_norm[:, :, None, None, :].expand(B, T, H, W, cond_norm.shape[-1])
            return cond_norm.reshape(B, target_tokens, cond_norm.shape[-1])
        if cond_norm.shape[1] == 1:
            return cond_norm.expand(B, target_tokens, cond_norm.shape[-1])

        raise AssertionError(
            f"Could not align cond_norm with x tokens: cond_norm.shape={cond_norm.shape}, expected second dim in {{1, {T}, {target_tokens}}}"
        )


class LSMSiT(nn.Module):

    def __init__(
        self,
        in_features: int=4,
        widths: list[int] = [1024],
        depths: list[int] = [24],
        heads: list[int] = [16],
        patch_sizes: list[tuple[int, int, int]] = [(1, 2, 2)],
        d_cond_norm: int = 512,
        activate_temporal_attention: bool = True,
        factorized_attention_impl: str = "parallel"
    ):
        super().__init__()

        c_features = in_features
        self.sub_modules = torch.nn.ModuleList()

        for i, (width, depth, head, patch_size) in enumerate(zip(widths, depths, heads, patch_sizes)):
            level = LSMLevel(
                in_features=c_features,
                out_features=width,
                depth=depth,
                num_heads=head,
                patch_size=patch_size,
                d_cond_norm=d_cond_norm,
                activate_temporal_attention=activate_temporal_attention,
                factorized_attention_impl=factorized_attention_impl
            )
            self.sub_modules.append(level)
            c_features = width

        # out proj unpatchifies the input
        out_patch = patch_sizes[-1]
        self.out_proj = TokenSplitLast3D(
            in_features=c_features,
            out_features=in_features,
            patch_size=out_patch,
        )

    def get_pos(self, x: Float[torch.Tensor, "B *DIM C"]) -> Float[torch.Tensor, "B *DIM C"]:
        # 3D pos
        B, T, H, W, _ = x.shape
        pos = make_axial_pos_3d(T, H, W, device=x.device).view(1, T, H, W, -1).expand(B, -1, -1, -1, -1)
        return pos

    def forward(
        self,
        x: Float[torch.Tensor, "B T Hl Wl Cl"],
        cond_norm: Float[torch.Tensor, "B D"]
    ):
        pos = self.get_pos(x)

        for sm in self.sub_modules:
            x, pos = sm(
                x=x,
                pos=pos,
                cond_norm=cond_norm,
            )
        x = self.out_proj(x)
        return x


"""
RaMViD latent space model.
"""


class RaMViDFM(nn.Module):

    def __init__(
        self,
        backbone: LSMSiT,
        freud_fs: FreudDiffusionAE,
        mapping: MappingNetwork,
        freud_ckpt_path: str,
        is_encoded_data: bool=False,
        max_conditional_frames: int=13,
        prob_unconditional: float = 0.25,
        m_norm: float=0.0,
        std_norm: float=1.0,
        val_ensemble_size: int=5,
    ):
        super().__init__()
        self.backbone = backbone
        self.is_encoded_data = is_encoded_data
        self.m_norm = m_norm
        self.std_norm = std_norm
        self.max_conditional_frames = max_conditional_frames
        self.prob_unconditional = prob_unconditional
        self.val_ensemble_size = val_ensemble_size

        self.mapping = mapping
        mapping_width = getattr(mapping, "width", None)
        if mapping_width is None:
            # Fallback for MappingNetwork implementations that do not expose `.width`
            mapping_width = int(mapping.in_norm.scale.shape[-1])
        self.time_emb = FourierFeatures(1, mapping_width)
        self.time_in_proj = nn.Linear(mapping_width, mapping_width, bias=False)

        self.first_stage = freud_fs
        if freud_ckpt_path is not None:
            first_stage_state_dict = torch.load(freud_ckpt_path, map_location="cpu")
            self.first_stage.load_state_dict(first_stage_state_dict, strict=True)
            print(f"[INFO] Loaded first stage from {freud_ckpt_path}", flush=True)
        self.first_stage.eval()
        for param in self.first_stage.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(
        self,
        x: Float[torch.Tensor, "B T H W C"],
        force_encode: bool=False
    ) -> Float[torch.Tensor, "B T Hl Wl Cl"]:
        if self.is_encoded_data and not force_encode:
            return (x - self.m_norm) / self.std_norm
        x = self.first_stage.encode(x)
        return (x - self.m_norm) / self.std_norm

    @torch.no_grad()
    def decode(
        self,
        z: Float[torch.Tensor, "B T Hl Wl Cl"],
        sample_steps: int = 10
    ) -> Float[torch.Tensor, "B T H W C"]:
        z = (z * self.std_norm) + self.m_norm
        return self.first_stage.decode(z, sample_steps=sample_steps)

    def get_conditioning(
        self,
        t: Float[torch.Tensor, "B T"],
        **data_kwargs
    ) -> dict: 
        time_emb = self.time_in_proj(self.time_emb(t[..., None]))
        cond_dict = {"cond_norm": self.mapping(time_emb)}
        return cond_dict

    def _get_zt(
        self,
        x: Float[torch.Tensor, "B T Hl Wl Cl"]
    ) -> tuple[
        Float[torch.Tensor, "B T Hl Wl Cl"],
        Float[torch.Tensor, "B T 1 1 1"],
        Float[torch.Tensor, "B T Hl Wl Cl"]
    ]:
        device = x.device
        B, *sample_dims = x.shape
        t_shape = (B, *([1] * len(sample_dims)))
        # logit sigmoid timestep sampling
        t = torch.sigmoid(torch.randn(t_shape, device=device))  # t shape: (B, 1, 1, 1, 1)

        T = x.shape[1]
        t = einops.repeat(t, "B 1 1 1 1 -> B T 1 1 1", T=T)

        # RaMViD logic
        
        # 1) with prob. `prob_unconditional`, do not use clean frames
        uncond = torch.rand(B, device=device) < self.prob_unconditional

        # 2) randomly sample [1, max_cond_frames] frame indices per batch element and set t=0 for those frames (no noise)
        max_cond_frames = min(self.max_conditional_frames, T)
        if max_cond_frames > 0:
            num_cond_frames = torch.randint(
                1,
                max_cond_frames + 1,
                (B,),
                device=device
            )
            frame_scores = torch.rand(B, T, device=device)
            perm = frame_scores.argsort(dim=1)
            ranks = torch.empty_like(perm)
            ranks.scatter_(
                1,
                perm,
                torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            )
            cond_mask = ranks < num_cond_frames.unsqueeze(1)
            cond_mask = cond_mask & (~uncond.unsqueeze(1))
            t = t.masked_fill(cond_mask[:, :, None, None, None], 0.0)

        # 3) compute zt using the forward diffusion process
        z1 = torch.randn_like(x)
        zt = (1 - t) * x + t * z1
        dtype = x.dtype
        zt, t = zt.to(dtype), t.to(dtype)
        return zt, t, z1

    def get_loss(
        self,
        x: Float[torch.Tensor, "B T Hl Wl Cl"],
        vtheta: Float[torch.Tensor, "B T Hl Wl Cl"],
        z1: Float[torch.Tensor, "B T Hl Wl Cl"],
        t: Float[torch.Tensor, "B T 1 1 1"]
    ):
        # standard FM loss
        sq_error = (z1 - x - vtheta) ** 2

        # set loss to 0 for all conditional frames (where t=0)
        valid_mask = (t != 0).to(sq_error.dtype)
        sq_error = sq_error * valid_mask

        # comparable mean across different number of valid frames
        denom = valid_mask.expand_as(sq_error).sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
        loss = sq_error.sum(dim=(1, 2, 3, 4)) / denom
        return loss

    def forward(
        self,
        x: Float[torch.Tensor, "B T H W C"] | Float[torch.Tensor, "B T Hl Wl Cl"],
        **data_kwargs
    ) -> Float[torch.Tensor, "B"]:
        x = self.encode(x)

        zt, t, z1 = self._get_zt(x)
        cond = self.get_conditioning(t, **data_kwargs)
        vtheta = self.backbone(zt, **cond)
        return self.get_loss(x, vtheta, z1, t)

    @torch.no_grad()
    def sample(
        self,
        data: Float[torch.Tensor, "B T H W C"] | Float[torch.Tensor, "B T Hl Wl Cl"],
        noise: Float[torch.Tensor, "B T Hl Wl Cl"],
        conditional_indices: Bool[torch.Tensor, "B T"],
        sample_steps: int = 50,
        decode_sample_steps: int = 10,
        data_is_latent: bool=True,
        return_decoded: bool=True,
        **kwargs
    ):
        if not data_is_latent:
            data = self.encode(data, force_encode=True)
        
        cond_idx_exp = conditional_indices[:, :, None, None, None].bool()
        noise_and_data = torch.where(
            cond_idx_exp,
            data,
            noise
        )
        
        dt = torch.tensor(1.0 / sample_steps, device=noise.device, dtype=noise.dtype)

        # initialize sample
        x = noise_and_data

        for i in range(sample_steps, 0, -1):
            t = torch.full(
                (noise.shape[0], noise.shape[1], 1, 1, 1),
                float(i) / sample_steps,
                device=noise.device, dtype=noise.dtype
            )
            t = t.masked_fill(cond_idx_exp, 0.0)
            cond = self.get_conditioning(t, **kwargs)
            vtheta = self.backbone(x, **cond)
            vtheta = vtheta.masked_fill(cond_idx_exp, 0.0)  # ensure conditional frames have vtheta=0 --> do not change
            # Forward pass trains vtheta to predict (z1 - x).
            # During sampling we integrate from t=1 -> t=0, hence "- dt * vtheta".
            x = x - vtheta * dt
            x = torch.where(cond_idx_exp, data, x)  # ensure conditional frames remain unchanged
        
        if return_decoded:
            x = self.decode(x, sample_steps=decode_sample_steps)
        return x


    def _calculate_ensemble_crps(
        self,
        ensemble_pred: Float[torch.Tensor, "B E T H W C"],
        gt: Float[torch.Tensor, "B T H W C"]
    ):
        gt_device = gt.device
        ensemble_pred = ensemble_pred.detach().cpu().to(torch.float32).numpy()
        gt = gt.detach().cpu().to(torch.float32).numpy()
        ensemble_pred = einops.rearrange(ensemble_pred, "B E T H W C -> B T H W C E")  # ensemble dim is last
        crps_b = crps_ensemble(gt, ensemble_pred)  # shape: (B, T, H, W, C)
        return torch.from_numpy(crps_b).to(gt_device).mean()


    @torch.no_grad()
    def validate(
        self,
        dataloader_val: torch.utils.data.DataLoader,
        global_rank: int,
        max_steps: int | None,
        device,
    ):
        if global_rank != 0:  # only perform validation on 0-rank GPU
            return
        
        # generate samples
        samples = []
        gts = []
        first_stage_recons = []
        first_stage_decode_steps = 10

        with tqdm(desc="Validating", disable=(global_rank != 0), total=max_steps) as pbar:
            for batch in dataloader_val:
                batch = dict_to(batch, device=device)
                batch = batch['x']
                batch = batch[0:min(batch.size(0), max_steps - len(samples))]

                # First-stage reconstruction validation (encode + decode of GT image/video)
                # to disentangle FS quality from LSM sampling.
                # This path is only available when dataloader provides decoded pixels (not pre-encoded latents).
                if not self.is_encoded_data:
                    first_stage_latent = self.first_stage.encode(batch)
                    first_stage_recon = self.first_stage.decode(
                        first_stage_latent,
                        sample_steps=first_stage_decode_steps,
                    )
                    first_stage_recons += [first_stage_recon[i].cpu() for i in range(first_stage_recon.shape[0])]

                condition = batch[:, :self.max_conditional_frames]
                condition = torch.cat([condition, torch.zeros_like(batch[:, self.max_conditional_frames:])], dim=1)
                latent_condition = self.encode(condition)
                latent_condition = einops.repeat(latent_condition, "B T Hl Wl Cl -> (B E) T Hl Wl Cl", E=self.val_ensemble_size)
                noise = torch.randn_like(latent_condition)
                cond_indices = (torch.arange(batch.size(1), device=device) < self.max_conditional_frames)[None, :]
                cond_indices = cond_indices.expand(latent_condition.size(0), -1)

                sample = self.sample(
                    data=latent_condition,
                    noise=noise,
                    conditional_indices=cond_indices,
                    sample_steps=25,
                    decode_sample_steps=10,
                    data_is_latent=True,
                    return_decoded=True,
                )

                if self.is_encoded_data:
                    batch = self.decode(batch)
                sample = einops.rearrange(sample, "(B E) T H W C -> B E T H W C", E=self.val_ensemble_size)

                samples += [sample[i].cpu() for i in range(sample.shape[0])]
                gts += [batch[i].cpu() for i in range(batch.shape[0])]

                # TODO do validation sampling

                pbar.update(batch.size(0))
                if len(samples) >= max_steps:
                    break

        samples = torch.stack(samples, dim=0)
        gts = torch.stack(gts, dim=0)

        log_dict = {}

        # Prediction quantitative metrics (RMSE + CRPS)
        samples_only_pred = samples[:, :, self.max_conditional_frames:]
        gts_only_pred = gts[:, self.max_conditional_frames:]
        
        rmse = torch.sqrt(((samples_only_pred.mean(dim=1) - gts_only_pred) ** 2).mean()).item()
        crps = self._calculate_ensemble_crps(samples_only_pred, gts_only_pred).item()

        log_dict["Val/pred_rmse"] = rmse
        log_dict["Val/pred_crps"] = crps

        if len(first_stage_recons) > 0:
            first_stage_recons = torch.stack(first_stage_recons, dim=0)
            fs_recon_rmse = torch.sqrt(((first_stage_recons - gts) ** 2).mean()).item()
            log_dict["Val/first_stage_recon_rmse"] = fs_recon_rmse

            fs_recon_images = [sample_to_im(x[-1]) for x in first_stage_recons]
            fs_gt_images = [sample_to_im(x[-1]) for x in gts]
            fs_combined_images = []
            for i, fs_recon_im in enumerate(fs_recon_images):
                fs_combined_images.append(np.vstack([fs_gt_images[i], fs_recon_im]))

            fs_num_images = len(fs_combined_images)
            fs_rows, fs_cols = compute_grid_dims(fs_num_images)
            fs_combined_images = add_padding(fs_combined_images)
            fs_grid_image = get_grid_image(fs_combined_images, fs_rows, fs_cols)
            log_dict["Val/first_stage_recon_last_img"] = wandb.Image(fs_grid_image)

        # Visualization

        # last frame as image
        image_samples = [sample_to_im(x[0, -1]) for x in samples]
        image_conditions = [sample_to_im(x[-1]) for x in gts]
        combined_images = []
        for i, im_sample in enumerate(image_samples):
            combined = np.vstack([image_conditions[i], im_sample])
            combined_images.append(combined)
        
        num_images = len(combined_images)
        row, cols = compute_grid_dims(num_images)
        combined_images = add_padding(combined_images)
        grid_image = get_grid_image(combined_images, row, cols)

        log_dict["Val/last_img"] = wandb.Image(grid_image)

        if samples.shape[2] > 1:  # only if temporal dimension exists
            # full video
            sample_videos = [sample_to_vid(x[0]) for x in samples]
            condition_videos = [sample_to_vid(x) for x in gts]

            combined_videos = []
            for i, vid_sample in enumerate(sample_videos):
                combined = np.concatenate([condition_videos[i], vid_sample], axis=1)
                combined = pad_vid(combined, 5)
                combined_videos.append(combined)
            num_videos = len(combined_videos)
            rows, cols = compute_grid_dims(num_videos)
            reel = get_grid_video(combined_videos, rows, cols)
            reel = np.repeat(reel[:, np.newaxis, :, :], 3, axis=1)  # ensure video has 3 channels for wandb
            log_dict["Val/full_video"] = wandb.Video(reel, fps=5, format="mp4")

            # only prediction video
            sample_videos = [sample_to_vid(x[0, self.max_conditional_frames:]) for x in samples]
            condition_videos = [sample_to_vid(x[self.max_conditional_frames:]) for x in gts]

            combined_videos = []
            for i, vid_sample in enumerate(sample_videos):
                combined = np.concatenate([condition_videos[i], vid_sample], axis=1)
                combined = pad_vid(combined, 5)
                combined_videos.append(combined)
            num_videos = len(combined_videos)
            rows, cols = compute_grid_dims(num_videos)
            reel = get_grid_video(combined_videos, rows, cols)
            reel = np.repeat(reel[:, np.newaxis, :, :], 3, axis=1)  # ensure video has 3 channels for wandb
            log_dict["Val/pred_video"] = wandb.Video(reel, fps=5, format="mp4")

            # mean prediction video
            sample_videos = [sample_to_vid(x[:, self.max_conditional_frames:].mean(dim=0)) for x in samples]
            condition_videos = [sample_to_vid(x[self.max_conditional_frames:]) for x in gts]

            combined_videos = []
            for i, vid_sample in enumerate(sample_videos):
                combined = np.concatenate([condition_videos[i], vid_sample], axis=1)
                combined = pad_vid(combined, 5)
                combined_videos.append(combined)
            num_videos = len(combined_videos)
            rows, cols = compute_grid_dims(num_videos)
            reel = get_grid_video(combined_videos, rows, cols)
            reel = np.repeat(reel[:, np.newaxis, :, :], 3, axis=1)  # ensure video has 3 channels for wandb
            log_dict["Val/pred_video_mean"] = wandb.Video(reel, fps=5, format="mp4")

        wandb.log(log_dict)
        
"""
Inference Helpers
"""

@torch.inference_mode()
def get_pred(
    model: RaMViDFM,
    sample: Float[torch.Tensor, "B T H W C"],
    cond_indices: Float[torch.Tensor, "B T"],
    sample_steps: int = 25,
    decoder_sample_steps: int = 10,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    decode_chunk_size: int = 4,
) -> Float[torch.Tensor, "B T H W C"]:
    cond_indices = cond_indices.bool()
    cond_idx_exp = cond_indices[:, :, None, None, None]
    condition = torch.where(cond_idx_exp, sample, torch.zeros_like(sample))
    
    with torch.autocast(device_type=device.type, dtype=dtype):
        latent_condition = model.encode(condition, force_encode=True)
        latent_noise = torch.randn_like(latent_condition)
        pred = model.sample(
            data=latent_condition,
            noise=latent_noise,
            conditional_indices=cond_indices,
            sample_steps=sample_steps,
            decode_sample_steps=decoder_sample_steps,
            data_is_latent=True,
            return_decoded=False,
        )
        pred = torch.cat(
            [
                model.decode(pred[start:start + decode_chunk_size], sample_steps=decoder_sample_steps)
                for start in range(0, pred.shape[0], decode_chunk_size)
            ],
            dim=0,
        )
    pred = pred.float().clamp(-1, 1)
    return pred

def get_lsm(
    ckpt_path: str | None = None,
    freud_ckpt_path: str | None = None,
):
    model = RaMViDFM(
        is_encoded_data=False,
        max_conditional_frames=13,
        prob_unconditional=0.25,
        m_norm=0.0452,
        std_norm=0.5391,
        val_ensemble_size=10,
        # TODO: change freud ckpt path when available
        freud_ckpt_path=freud_ckpt_path,
        mapping=MappingNetwork(
            n_layers=3,
            d_model=512,
            d_ff=512*3,
            dropout=0.0,
        ),
        backbone=LSMSiT(
            in_features=4,
            widths=[1024],
            depths=[24],
            heads=[16],
            patch_sizes=[(1, 2, 2)],
            d_cond_norm=512,
            activate_temporal_attention=True,
            factorized_attention_impl="parallel" # TODO: potentially change
        ),
        # TODO: potentially change freud impl
        freud_fs=FreudDiffusionAE(
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
                d_ff=384*3,
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
                neighborhood_sizes=[(3, 7, 7), None, (3, 7, 7) ],
                patch_sizes=[4, 2, -2, -4],
                factorized_attention_impl="parallel",
                activate_temporal_attention=True,
            )
        )
    )

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model