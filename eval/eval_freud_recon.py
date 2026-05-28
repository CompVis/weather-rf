import os
import torch
import random
import einops
import click
import csv

import torchmetrics as tm
import numpy as np

from jaxtyping import Float
from tqdm import tqdm
from tabulate import tabulate
from pathlib import Path
from datetime import datetime

from model.freud import FreudDiffusionAE, get_freud_model


# --- Data ---

class CasCastSevirDataset(torch.utils.data.Dataset):

    def __init__(self, split, input_length=13, pred_length=12, data_dir='radar:s3://weather_radar_datasets/sevir', txt_dir: str='dataset/sevir_list', channel_first: bool=True, base_freq='5min', height=384, width=384, **kwargs):
        super().__init__()
        # assert input_length == 13, pred_length==12
        self.input_length = input_length
        self.pred_length = pred_length

        self.txt_dir = txt_dir
        self.channel_first = channel_first

        self.file_list = self._init_file_list(split)
        self.data_dir = os.path.join(data_dir, f'{split}_2h')

    def _init_file_list(self, split):
        if split == 'train':
            txt_path = os.path.join(self.txt_dir, "train.txt")
        elif split == 'valid':
            txt_path = os.path.join(self.txt_dir, "val.txt")
        elif split == 'test':
            txt_path = os.path.join(self.txt_dir, "test.txt")
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)

    def _load_frames(self, file):
        file_path = os.path.join(self.data_dir, file)
        frame_data = np.load(file_path).astype(np.float32)
        tensor = torch.from_numpy(frame_data) / 255
        if len(frame_data.shape) == 3:
            tensor = tensor.unsqueeze(0)
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor

    def __getitem__(self, index):
        file = self.file_list[index]
        frame_data = self._load_frames(file)
        packed_results = dict()
        packed_results['inputs'] = frame_data[:self.input_length]
        packed_results['data_samples'] = frame_data[self.input_length:self.input_length+self.pred_length]
        packed_results['file_name'] = file

        if self.channel_first:
            packed_results['inputs'] = packed_results['inputs'].permute(1, 0, 2, 3)
            packed_results['data_samples'] = packed_results['data_samples'].permute(1, 0, 2, 3)

        return packed_results

class EvalDataset(torch.utils.data.Dataset):

    def __init__(self, path: str):
        self.pred_path = os.path.join(path, 'ens_pred/')
        self.gt_path = os.path.join(path, 'gt/')
    
    def __len__(self):
        return len([x for x in os.listdir(self.gt_path) if x.endswith('.npy')])
    
    def __getitem__(self, index):
        gt_data = np.load(os.path.join(self.gt_path, f'{index}.npy'))
        pred_data = np.load(os.path.join(self.pred_path, f'{index}.npy'))

        gt_tensor = torch.from_numpy(gt_data)
        pred_tensor = torch.from_numpy(pred_data)

        return {
            'gt': gt_tensor,
            'pred': pred_tensor
        }

def get_subset(dataset: torch.utils.data.Dataset, seed: int, num_samples: int) -> torch.utils.data.Dataset:
    random.seed(seed)
    torch.manual_seed(seed)
    total_size = len(dataset)
    indices = random.sample(range(total_size), num_samples)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

def normalize(data: torch.Tensor, data_min: float=0., data_max: float=1., exp_min: float=-1., exp_max: float=1., exp_dtype: torch.dtype=torch.bfloat16):
    data = data.clamp(data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * (exp_max - exp_min) + exp_min
    return data.to(exp_dtype)

# --- Inference ---

@torch.inference_mode()
@torch.no_grad()
def get_recon(
    model: FreudDiffusionAE,
    x: Float[torch.Tensor, "B T H W C"],
    decoding_sample_steps: int=10,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    noise: torch.Tensor | None = None
) -> Float[torch.Tensor, "B T H W C"]:
    with torch.autocast(device_type=device.type, dtype=dtype):
        latent = model.encode(x)
        recon = model.decode(latent, noise=noise, sample_steps=decoding_sample_steps)
    recon = recon.float()
    recon = recon.clamp(-1, 1)
    return recon, latent
    
# --- Evaluation ---

def eval(
    eval_pth: str,
    batch_size: int,
    num_samples: int | None = None,
    max_ensemble_size: int=100,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 42,
    verbose: int=2,
):
    eval_ds = EvalDataset(eval_pth)
    if num_samples is not None:
        eval_ds = get_subset(eval_ds, seed, num_samples)
    
    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    rmse_metric = tm.MeanSquaredError(squared=False).to(device)
    mae_metric = tm.MeanAbsoluteError().to(device)
    ssim_metric = tm.StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device)
    psnr_metric = tm.PeakSignalNoiseRatio(data_range=(0, 1)).to(device)
    dmae_metric = tm.MeanMetric().to(device)

    with tqdm(desc='Evaluating Reconstructions', disable=verbose < 0, total=len(eval_dl)) as pbar:
        for batch in eval_dl:
            pred_ens = batch['pred'][:, :max_ensemble_size].to(device) 
            gt_batch = batch['gt'].to(device)

            ens_mean = pred_ens.mean(dim=1)

            # rmse, mae
            rmse_metric.update(ens_mean, gt_batch)
            mae_metric.update(ens_mean, gt_batch)

            # SSIM
            pred_frames = einops.rearrange(ens_mean, "bs t c h w -> (bs t) c h w")
            gt_frames = einops.rearrange(gt_batch, "bs t c h w -> (bs t) c h w")
            ssim_metric.update(pred_frames, gt_frames)
            # PSNR
            psnr_metric.update(pred_frames, gt_frames)

            # dmae
            d_pred = torch.diff(ens_mean, dim=1)
            d_true = torch.diff(gt_batch, dim=1)
            batch_dmae = torch.abs(d_pred - d_true).mean()
            dmae_metric.update(batch_dmae)

            pbar.update(1)
    
    rmse = rmse_metric.compute().item()
    mae = mae_metric.compute().item()
    ssim = ssim_metric.compute().item()
    psnr = psnr_metric.compute().item()
    dmae = dmae_metric.compute().item()

    return rmse, mae, ssim, psnr, dmae

# --- Main Loop ---

@click.command()
@click.option('--model_path', type=str, required=True, help="Path to the model checkpoint.")
@click.option('--sevir_npy_path', type=str, required=True, help="Base path to the SEVIR dataset in npy format.")
@click.option('--txt_path', type=str, default="data/test_data.txt", help="Base path to the train/test split txt files.")
@click.option('--tag', type=str, default="freud", help="Tag to identify the model/run, used for saving results.")
@click.option('--save_pth', type=str, default="eval_results/freud_recon", help="Base path to save evaluation results.")
@click.option('--eval_pth', type=str, default="eval_results/freud/recon_eval_data", help="Base path to save intermediate evaluation data (reconstructions and ground truths).")
@click.option('--batch_size', type=int, default=4, help="Batch size for computing reconstructions and evaluation.")
@click.option('--num_samples', type=int, default=-1, help="Number of samples to evaluate. Set to -1 to evaluate on the entire test set.")
@click.option('--ensemble_size', type=int, default=1, help="Number of samples to use for ensemble prediction. If the model is stochastic, using multiple samples can improve reconstruction quality.")
@click.option('--input_length', type=int, default=13, help="Number of input frames provided to the model.")
@click.option('--pred_length', type=int, default=12, help="Number of frames the model is asked to reconstruct/predict.")
@click.option('--decoder_sample_steps', type=int, default=10, help="Number of sampling steps to use in the decoder when generating reconstructions. More steps may improve quality but will take longer.")
@click.option('--verbose', type=int, default=2, help="Verbosity level. Set to 0 for no progress bars, 1 for progress bars during reconstruction and evaluation, 2 for additional debug prints.")
@click.option('--seed', type=int, default=42, help="Random seed for reproducibility, used for sampling subsets and any stochasticity in the model.")
def main(
    model_path: str,
    sevir_npy_path: str,
    txt_path: str="data/test_data.txt",
    tag: str = "freud",
    save_pth: str = "eval_results/freud_recon",
    eval_pth: str = "eval_results/freud/recon_eval_data",
    batch_size: int = 4,
    num_samples: int = -1,
    ensemble_size: int = 1,
    input_length: int = 13,
    pred_length: int = 12,
    decoder_sample_steps: int = 10,
    verbose: int = 2,
    seed: int = 42,
):
    # setup
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_pth = Path(save_pth) / tag / now
    save_pth.mkdir(parents=True, exist_ok=True)
    eval_pth = Path(eval_pth) / tag / now
    eval_pth.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = get_freud_model(
        ckpt_path=model_path,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # compute all reconstructions
    dset = CasCastSevirDataset(
        split="test",
        input_length=input_length,
        pred_length=pred_length,
        data_dir=sevir_npy_path,
        txt_dir=txt_path,
    )
    if num_samples <= 0:
        num_samples = len(dset)

    c_dl = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    all_recon = []
    all_gt = []
    for batch in tqdm(
        c_dl,
        desc="Computing Reconstructions",
        disable=verbose < 0,
        total=min(int((num_samples+batch_size-1)//batch_size), len(c_dl))
    ):
        inputs = batch['inputs']
        gt = batch['data_samples']
        sample = torch.cat([inputs, gt], dim=2)
        all_gt += [g for g in sample.cpu()]

        orig_dtype = sample.dtype
        sample = sample.to(device=device)
        sample = normalize(sample, data_min=0., data_max=1., exp_min=-1., exp_max=1., exp_dtype=dtype)
        sample = einops.repeat(sample, "b c t h w -> (b e) t h w c", e=ensemble_size)
        noise = torch.randn_like(sample)
        recon, _ = get_recon(model, sample, decoding_sample_steps=decoder_sample_steps, device=device, dtype=dtype, noise=noise)
        recon = einops.rearrange(recon, "(b e) t h w c -> b e t h w c", e=ensemble_size)
        recon = normalize(recon, data_min=-1., data_max=1., exp_min=0., exp_max=1., exp_dtype=orig_dtype)

        all_recon += [r.cpu() for r in recon]

        if len(all_recon) >= num_samples:
            break
    
    all_recon = torch.stack(all_recon)[:num_samples]  # B,E,T,H,W,C
    all_gt = torch.stack(all_gt)[:num_samples]  # B,T,H,W,C

    # save samples for evaluation and further processing
    ens_pth = eval_pth / "ens_pred"
    gt_pth = eval_pth / "gt"
    ens_pth.mkdir(parents=True, exist_ok=True)
    gt_pth.mkdir(parents=True, exist_ok=True)

    for j in range(all_gt.shape[0]):
        c_pred = all_recon[j]  # E,T,H,W,C
        c_pred = einops.rearrange(c_pred, "e t h w c -> e t c h w")
        c_pred = c_pred.cpu().numpy()
        np.save(ens_pth / f"{j}.npy", c_pred)

        c_gt = all_gt[j]  # T,H,W,C
        c_gt = einops.rearrange(c_gt, "c t h w -> t c h w")
        c_gt = c_gt.cpu().numpy()
        np.save(gt_pth / f"{j}.npy", c_gt)

    # Evaluation
    rmse, mae, ssim, psnr, dmae = eval(
        eval_pth=str(eval_pth),
        batch_size=batch_size,
        num_samples=num_samples,
        max_ensemble_size=ensemble_size,
        device=device,
        seed=seed,
        verbose=verbose
    )

    # Save results
    header = ['Model', 'RMSE', 'MAE', 'SSIM', 'PSNR', 'dMAE']
    table = tabulate(
        [[tag, rmse, mae, ssim, psnr, dmae]],
        headers=header,
        tablefmt='github'
    )

    with open(save_pth / 'results.txt', 'w') as f:
        f.write(table)
    
    with open(save_pth / 'results.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([tag, rmse, mae, ssim, psnr, dmae])

if __name__ == "__main__":
    main()