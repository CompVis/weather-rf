import torch
import os
import random
import einops
import click
import math
import json
import scipy

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchvision.transforms.functional import center_crop

from model.lsm import RaMViDFM, get_lsm

# --- Metrics ---
# metrics are taken unmodified from CasCast (https://github.com/OpenEarthLab/CasCast)


@torch.no_grad()
def cal_SSIM(gt, pred, is_img=True):
    '''
    iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
    iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
    '''
    cal_ssim = StructuralSimilarityIndexMeasure(data_range=int(torch.max(gt)-torch.min(gt)) ).to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    ssim = cal_ssim(pred, gt).cpu()
    
    # print(ssim)
    # ssim = cal_ssim_2(pred, gt).cpu()
    
    return ssim.item()

@torch.no_grad()
def cal_PSNR(gt, pred, is_img=True):
    '''
    gt.shape=pred.shape=[n t c h w]
    '''
    data_range = gt.max() - gt.min()
    cal_psnr = PeakSignalNoiseRatio(data_range=data_range).to(gt.device)
    if is_img:
        pred = torch.maximum(pred, torch.min(gt))
        pred = torch.minimum(pred, torch.max(gt))
    pred = einops.rearrange(pred, 'n t c h w -> (n t) c h w')
    gt = einops.rearrange(gt, 'n t c h w -> (n t) c h w')
    psnr = 0
    for n in range(pred.shape[0]):
        psnr += cal_psnr(pred[n], gt[n]).cpu()
    return (psnr / pred.shape[0]).item()

@torch.no_grad()
def cal_CRPS(gt, pred, type='avg', scale=4, mode='mean', eps=1e-10):
    """
    gt: (b, t, c, h, w)
    pred: (b, n, t, c, h, w)
    """
    assert mode in ['mean', 'raw'], 'CRPS mode should be mean or raw'
    _normal_dist = torch.distributions.Normal(0, 1)
    _frac_sqrt_pi = 1 / np.sqrt(np.pi)

    b, n, t, _, _, _ = pred.shape
    gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')
    pred = einops.rearrange(pred, 'b n t c h w -> (b n t) c h w')
    if type == 'avg':
        pred = F.avg_pool2d(pred, scale, stride=scale)
        gt = F.avg_pool2d(gt, scale, stride=scale)
    elif type == 'max':
        pred = F.max_pool2d(pred, scale, stride=scale)
        gt = F.max_pool2d(gt, scale, stride=scale)
    else:
        gt = gt
        pred = pred
    gt = einops.rearrange(gt, '(b t) c h w -> b t c h w', b=b)
    pred = einops.rearrange(pred, '(b n t) c h w -> b n t c h w', b=b, n=n)

    pred_mean = torch.mean(pred, dim=1)
    pred_std = torch.std(pred, dim=1) if n > 1 else torch.zeros_like(pred_mean)
    normed_diff = (pred_mean - gt + eps) / (pred_std + eps)
    cdf = _normal_dist.cdf(normed_diff)
    pdf = _normal_dist.log_prob(normed_diff).exp()

    crps = (pred_std + eps) * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
    if mode == "mean":
        return torch.mean(crps).item()
    return crps.item()
    
def cal_RMSE(gt, pred, type='avg', scale=1):
    """
    gt: (b, t, c, h, w)
    pred: (b, n, t, c, h, w)
    """

    b, n, t, _, _, _ = pred.shape
    gt = einops.rearrange(gt, 'b t c h w -> (b t) c h w')
    pred = einops.rearrange(pred, 'b n t c h w -> (b n t) c h w')
    
    if scale > 1:
        if type == 'avg':
            pred = F.avg_pool2d(pred, scale, stride=scale)
            gt = F.avg_pool2d(gt, scale, stride=scale)
        elif type == 'max':
            pred = F.max_pool2d(pred, scale, stride=scale)
            gt = F.max_pool2d(gt, scale, stride=scale)
    
    gt = einops.rearrange(gt, '(b t) c h w -> b t c h w', b=b)
    pred = einops.rearrange(pred, '(b n t) c h w -> b n t c h w', b=b, n=n)
    pred = pred.mean(dim=1) # average over ensemble members

    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    return rmse.item()


def _threshold(target, pred ,T):
    t = (target >= T).float()
    p = (pred >= T).float()
    is_nan = torch.logical_or(torch.isnan(target),
                              torch.isnan(pred))
    t[is_nan] = 0
    p[is_nan] = 0
    return t, p



class SEVIRSkillScore(object):
    def __init__(self,
                 layout='NHWT',
                 mode='0',
                 seq_len=None,
                 preprocess_type='sevir',
                 threshold_list=[16, 74, 133, 160, 181, 219],
                 metrics_list=['csi', 'csi-4-avg', 'csi-16-avg',
                               'csi-4-max', 'csi-16-max', 'bias',
                                'sucr', 'pod', 'hss'], #['csi', 'bias', 'sucr', 'pod'],
                 dist_eval=False,
                #  device='cuda',
                 eps=1e-4,):
        self.layout = layout
        self.preprocess_type = preprocess_type
        self.threshold_list = threshold_list
        self.metrics_list = metrics_list
        self.eps = eps
        self.mode = mode
        self.seq_len = seq_len
        
        self.dist_eval = dist_eval
        # self.device = device
        
        if mode in ("0", ):
            self.keep_seq_len_dim = False
            state_shape = (len(self.threshold_list), )
        elif mode in ("1", "2"):
            self.keep_seq_len_dim = True
            assert isinstance(self.seq_len, int), "seq_len must be provided when we need to keep seq_len dim."
            state_shape = (len(self.threshold_list), self.seq_len)
        else:
            raise NotImplementedError(f"mode {mode} not supported!")
        
        self.hits = torch.zeros(state_shape)
        self.misses = torch.zeros(state_shape)
        self.fas = torch.zeros(state_shape)
        self.cor = torch.zeros(state_shape)

        ## pooling csi ##
        self.hits_avg_pool_4 = torch.zeros(state_shape)
        self.misses_avg_pool_4 = torch.zeros(state_shape)
        self.fas_avg_pool_4 = torch.zeros(state_shape)

        self.hits_max_pool_4 = torch.zeros(state_shape)
        self.misses_max_pool_4 = torch.zeros(state_shape)
        self.fas_max_pool_4 = torch.zeros(state_shape)

        self.hits_avg_pool_16 = torch.zeros(state_shape)
        self.misses_avg_pool_16 = torch.zeros(state_shape)
        self.fas_avg_pool_16 = torch.zeros(state_shape)

        self.hits_max_pool_16 = torch.zeros(state_shape)
        self.misses_max_pool_16 = torch.zeros(state_shape)
        self.fas_max_pool_16 = torch.zeros(state_shape)



    def pod(self, hits, misses, fas, eps):
        return hits / (hits + misses + eps)

    def sucr(self, hits, misses, fas, eps):
        return hits / (hits + fas + eps)

    def csi(self, hits, misses, fas, eps):
        return hits / (hits + misses + fas + eps)

    def bias(self, hits, misses, fas, eps):
        bias = (hits + fas) / (hits + misses + eps)
        logbias = torch.pow(bias / torch.log(torch.tensor(2.0)), 2.0)
        return logbias
    
    def hss(self, hits, misses, fas, cor, eps):
        hss = 2 * (hits * cor - misses * fas) / ((hits + misses) * (misses + cor) + (hits + fas) * (fas + cor) + eps)
        return hss

    @property
    def hits_misses_fas_reduce_dims(self):
        if not hasattr(self, "_hits_misses_fas_reduce_dims"):
            seq_dim = self.layout.find('T')
            self._hits_misses_fas_reduce_dims = list(range(len(self.layout)))
            if self.keep_seq_len_dim:
                self._hits_misses_fas_reduce_dims.pop(seq_dim)
        return self._hits_misses_fas_reduce_dims

    def preprocess(self, pred, target):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        else:
            raise NotImplementedError
        return pred, target

    def preprocess_pool(self, pred, target, pool_size=4, type='avg'):
        if self.preprocess_type == "sevir":
            pred = pred.detach() / (1. / 255.)
            target = target.detach() / (1. / 255.)
        elif self.preprocess_type == "meteonet":
            pred = pred.detach() / (1. / 70.)
            target = target.detach() / (1. / 70.)
        b, t, _, _, _ = pred.shape
        pred = einops.rearrange(pred, 'b t c h w -> (b t) c h w')
        target = einops.rearrange(target, 'b t c h w -> (b t) c h w')
        if type == 'avg':
            pred = F.avg_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.avg_pool2d(target, kernel_size=pool_size, stride=pool_size)
        elif type == 'max':
            pred = F.max_pool2d(pred, kernel_size=pool_size, stride=pool_size)
            target = F.max_pool2d(target, kernel_size=pool_size, stride=pool_size)
        pred = einops.rearrange(pred, '(b t) c h w -> b t c h w', b=b)
        target = einops.rearrange(target, '(b t) c h w -> b t c h w', b=b)
        return pred, target


    def calc_seq_hits_misses_fas(self, pred, target, threshold):
        with torch.no_grad():
            t, p = _threshold(target, pred, threshold)
            hits = torch.sum(t * p, dim=self.hits_misses_fas_reduce_dims).int()
            misses = torch.sum(t * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
            fas = torch.sum((1 - t) * p, dim=self.hits_misses_fas_reduce_dims).int()
            cor = torch.sum((1 - t) * (1 - p), dim=self.hits_misses_fas_reduce_dims).int()
        return hits, misses, fas, cor
    
    @torch.no_grad()
    def update(self, pred, target):
        ## pool 1 ##
        self.hits = self.hits.to(pred.device)
        self.misses = self.misses.to(pred.device)
        self.fas = self.fas.to(pred.device)
        self.cor = self.cor.to(pred.device)
        _pred, _target = self.preprocess(pred, target)

        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += fas
            self.cor[i] += cor
        ## max pool 4 ##
        self.hits_max_pool_4 = self.hits_max_pool_4.to(pred.device)
        self.misses_max_pool_4 = self.misses_max_pool_4.to(pred.device)
        self.fas_max_pool_4 = self.fas_max_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_4[i] += hits
            self.misses_max_pool_4[i] += misses
            self.fas_max_pool_4[i] += fas 
        ## max pool 16 ##
        self.hits_max_pool_16 = self.hits_max_pool_16.to(pred.device)
        self.misses_max_pool_16 = self.misses_max_pool_16.to(pred.device)
        self.fas_max_pool_16 = self.fas_max_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='max')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_max_pool_16[i] += hits
            self.misses_max_pool_16[i] += misses
            self.fas_max_pool_16[i] += fas 
        ## avg pool 4 ##
        self.hits_avg_pool_4 = self.hits_avg_pool_4.to(pred.device)
        self.misses_avg_pool_4 = self.misses_avg_pool_4.to(pred.device)
        self.fas_avg_pool_4 = self.fas_avg_pool_4.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=4, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_4[i] += hits
            self.misses_avg_pool_4[i] += misses
            self.fas_avg_pool_4[i] += fas 
        ## avg pool 16 ##
        self.hits_avg_pool_16 = self.hits_avg_pool_16.to(pred.device)
        self.misses_avg_pool_16 = self.misses_avg_pool_16.to(pred.device)
        self.fas_avg_pool_16 = self.fas_avg_pool_16.to(pred.device)
        _pred, _target = self.preprocess_pool(pred, target, pool_size=16, type='avg')
        for i, threshold in enumerate(self.threshold_list):
            hits, misses, fas, cor = self.calc_seq_hits_misses_fas(_pred, _target, threshold)
            self.hits_avg_pool_16[i] += hits
            self.misses_avg_pool_16[i] += misses
            self.fas_avg_pool_16[i] += fas 

    def _get_hits_misses_fas(self, metric_name):
        if metric_name.endswith('-4-avg'):
            hits = self.hits_avg_pool_4
            misses = self.misses_avg_pool_4
            fas = self.fas_avg_pool_4
        elif metric_name.endswith('-16-avg'):
            hits = self.hits_avg_pool_16
            misses = self.misses_avg_pool_16
            fas = self.fas_avg_pool_16
        elif metric_name.endswith('-4-max'):
            hits = self.hits_max_pool_4
            misses = self.misses_max_pool_4
            fas = self.fas_max_pool_4
        elif metric_name.endswith('-16-max'):
            hits = self.hits_max_pool_16
            misses = self.misses_max_pool_16
            fas = self.fas_max_pool_16
        else:
            hits = self.hits
            misses = self.misses
            fas = self.fas
        return [hits, misses, fas]
    
    def _get_correct_negtives(self):
        return self.cor
    
    @torch.no_grad()
    def compute(self):
        if self.dist_eval:
            self.synchronize_between_processes()
        
        metrics_dict = {'pod': self.pod,
                        'csi': self.csi,
                        'csi-4-avg': self.csi, 
                        'csi-16-avg': self.csi,
                        'csi-4-max': self.csi, 
                        'csi-16-max': self.csi,
                        'sucr': self.sucr,
                        'bias': self.bias,
                        'hss': self.hss}
        ret = {}
        for threshold in self.threshold_list:
            ret[threshold] = {}
        ret["avg"] = {}
        
        for metrics in self.metrics_list:
            if self.keep_seq_len_dim:
                score_avg = np.zeros((self.seq_len, ))
            else:
                score_avg = 0
            hits, misses, fas = self._get_hits_misses_fas(metrics)
            # scores = metrics_dict[metrics](self.hits, self.misses, self.fas, self.eps)
            if metrics != 'hss':
                scores = metrics_dict[metrics](hits, misses, fas, self.eps)
            else:
                cor = self._get_correct_negtives()
                scores = metrics_dict[metrics](hits, misses, fas, cor, self.eps)
            scores = scores.detach().cpu().numpy()
            for i, threshold in enumerate(self.threshold_list):
                if self.keep_seq_len_dim:
                    score = scores[i]  # shape = (seq_len, )
                else:
                    score = scores[i].item()  # shape = (1, )
                if self.mode in ("0", "1"):
                    ret[threshold][metrics] = score
                elif self.mode in ("2", ):
                    ret[threshold][metrics] = np.mean(score).item()
                else:
                    raise NotImplementedError
                score_avg += score
            score_avg /= len(self.threshold_list)
            if self.mode in ("0", "1"):
                ret["avg"][metrics] = score_avg
            elif self.mode in ("2",):
                ret["avg"][metrics] = np.mean(score_avg).item()
            else:
                raise NotImplementedError
        return ret

    @torch.no_grad()
    def get_single_frame_metrics(self, target, pred, metrics=['ssim', 'psnr', ]): #'cspr', 'cspr-4-avg', 'cspr-16-avg', 'cspr-4-max', 'cspr-16-max'
        metric_funcs = {
            'ssim': cal_SSIM,
            'psnr': cal_PSNR
        }
        metrics_dict = {}
        for metric in metrics:
            metric_fun = metric_funcs[metric]
            metrics_dict[metric] = metric_fun(gt=target*255., pred=pred*255., is_img=False)
        return metrics_dict
    
    @torch.no_grad()
    def get_crps(self, target, pred):
        """
        pred: (b, t, c, h, w)/(b, n, t, c, h, w)
        target: (b, t, c, h, w)
        """
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        crps = cal_CRPS(gt=target, pred=pred, type='none')
        crps_avg_4 = cal_CRPS(gt=target, pred=pred, type='avg', scale=4)
        crps_avg_16 = cal_CRPS(gt=target, pred=pred, type='avg', scale=16)
        crps_max_4 = cal_CRPS(gt=target, pred=pred, type='max', scale=4)
        crps_max_16 = cal_CRPS(gt=target, pred=pred, type='max', scale=16)
        crps_dict = {
            'crps': crps,
            'crps_avg_4': crps_avg_4,
            'crps_avg_16': crps_avg_16,
            'crps_max_4': crps_max_4,
            'crps_max_16': crps_max_16
        }
        return crps_dict

    @torch.no_grad()
    def get_rmse(self, target, pred):
        if len(pred.shape) == 5:
            pred = pred.unsqueeze(1)
        
        rmse = cal_RMSE(gt=target, pred=pred, type='none')
        rmse_avg_4 = cal_RMSE(gt=target, pred=pred, type='avg', scale=4)
        rmse_avg_16 = cal_RMSE(gt=target, pred=pred, type='avg', scale=16)
        rmse_max_4 = cal_RMSE(gt=target, pred=pred, type='max', scale=4)
        rmse_max_16 = cal_RMSE(gt=target, pred=pred, type='max', scale=16)
        rmse_dict = {
            'rmse': rmse,
            'rmse_avg_4': rmse_avg_4,
            'rmse_avg_16': rmse_avg_16,
            'rmse_max_4': rmse_max_4,
            'rmse_max_16': rmse_max_16
        }
        return rmse_dict


    def reset(self):
        self.hits = self.hits*0
        self.misses = self.misses*0
        self.fas = self.fas*0

        self.hits_avg_pool_4 *= 0
        self.hits_avg_pool_16 *= 0
        self.hits_max_pool_4 *= 0
        self.hits_max_pool_16 *= 0

        self.misses_avg_pool_4 *= 0
        self.misses_avg_pool_16 *= 0
        self.misses_max_pool_4 *= 0
        self.misses_max_pool_16 *= 0
 
        self.fas_avg_pool_4 *= 0
        self.fas_avg_pool_16 *= 0
        self.fas_max_pool_4  *= 0
        self.fas_max_pool_16  *= 0


@torch.no_grad()
class cal_FVD:
    def __init__(self, use_gpu=False, resize_crop=False):
        '''
        iter_cal=True, gt.shape=pred.shape=[nb b t c h w]
        iter_cal=Fasle, gt.shape=pred.shape=[n t c h w]
        '''
        
        self.use_gpu = use_gpu
        self.resize_crop = resize_crop
        # detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        self.detector = torch.jit.load("/mnt/cache/gongjunchao/workdir/radar_forecasting/utils/fvd/i3d_torchscript.pt").eval()
        if torch.cuda.is_available() and self.use_gpu:
            self.detector = self.detector.cuda()
        self.feats = []
    
    def preprocess(self, video):
        """
        video: (b, t, c, h, w) in [0, 1]
        this function transform the domain to [-1, 1] 
        """
        video = video * 2 - 1
        return video

    @torch.no_grad()
    def __call__(self, videos_real, videos_fake):
        feats_fake = []
        feats_real = []
        detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        
        videos_fake = einops.rearrange(
            self.bilinear_interpolation(videos_fake), 'n t c h w -> n c t h w'
        )
        videos_real = einops.rearrange(
            self.bilinear_interpolation(videos_real), 'n t c h w -> n c t h w'
        )
        if torch.cuda.is_available() and self.use_gpu:
            videos_fake, videos_real = videos_fake.cuda(), videos_real.cuda()
        # print(videos_fake.shape, videos_real.shape)
        # videos_fake = videos_fake.repeat(1, 1, 10, 1, 1)
        # videos_real = videos_real.repeat(1, 1, 10, 1, 1)
        feats_fake = self.detector(videos_fake, **detector_kwargs).cpu()
        feats_real = self.detector(videos_real, **detector_kwargs).cpu()
        self.feats.append(torch.stack([feats_fake, feats_real], dim=0))
        return
    
    def update(self, videos_real, videos_fake):
        self(videos_real=videos_real, videos_fake=videos_fake)
        return

    def _reset(self):
        self.feats = []

    def compute(self):
        feats = torch.cat(self.feats, dim=1)
        fake_feats = feats[0]
        real_feats = feats[1]
        fvd = self._cal_FVD(feats_fake=fake_feats, feats_real=real_feats)
        return fvd

    def bilinear_interpolation(self, image):
        N, T, C, H, W = image.shape
        def my_resize(img):
            img = img.view(-1, C, H, W)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.view(N, T, C, 224, 224)  
            return img
        def my_resize_crop(img):
            img = img.view(-1, C, H, W)
            if H<W:
                img = F.interpolate(img, size=(224, int(W*224/H)), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, 224, int(W*224/H))  
            else:   # W<=H
                img = F.interpolate(img, size=(int(H*224/W), 224), mode='bilinear', align_corners=False)
                img = img.view(N, T, C, int(H*224/W), 224)  
            return center_crop(img, (224, 224))
        if H == W and H < 224:
            return my_resize(img=image)
        elif self.resize_crop:
            return my_resize_crop(img=image)
        else: 
            return my_resize(img=image)

    def _cal_FVD(self, feats_fake, feats_real):
        def compute_fvd(feats_fake, feats_real):
            mu_gen, sigma_gen = compute_stats(feats_fake)
            mu_real, sigma_real = compute_stats(feats_real)
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            return float(fid)

        def compute_stats(feats):
            feats = feats.reshape(-1, feats.shape[-1])
            mu = feats.mean(axis=0)
            sigma = np.cov(feats, rowvar=False)
            return mu, sigma
        return compute_fvd(feats_fake, feats_real)

# --- Data Stuff ---

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

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

# --- Inference ---

@torch.inference_mode()
@torch.no_grad()
def get_pred(
    model: RaMViDFM,
    sample: torch.Tensor,
    cond_indices: torch.Tensor,
    sample_steps: int = 25,
    decoder_sample_steps: int = 10,
    device: torch.device = torch.device("cuda:0"),
    dtype: torch.dtype = torch.bfloat16,
    decode_chunk_size: int = 4,
) -> torch.Tensor:
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

# --- Evaluation ---

def eval(
    eval_pth: str,
    batch_size: int,
    pred_length: int,
    num_samples: int=-1,
    seed: int=42,
    verbose: int = 2,
):
    eval_ds = EvalDataset(eval_pth)

    num_samples = min(num_samples, len(eval_ds)) if num_samples > 0 else len(eval_ds)
    eval_ds = get_subset(eval_ds, seed=seed, num_samples=num_samples)

    eval_dl = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    # setup of metrics follows CasCast as closely as possible!
    metrics = SEVIRSkillScore(
        layout="NTCHW",
        mode="0",
        seq_len=pred_length,
        preprocess_type="sevir",
        threshold_list=[16, 74, 133, 160, 181, 219], 
        metrics_list=['csi', 'hss'],
        dist_eval=False, 
        eps=1e-4, 
    )

    final_res_single_image = {}
    i = 0
    with tqdm(desc='Calculating Metrics', disable=verbose < 0, total=num_samples) as pbar:
        for batch in eval_dl:
            gt = batch['gt']
            ens_pred = batch['pred']
            mean_pred = ens_pred.mean(dim=1)
            # HSS, CSI
            metrics.update(target=gt, pred=mean_pred)

            # SSIM / PSNR
            single_image_res = metrics.get_single_frame_metrics(target=gt, pred=mean_pred)
            for key in single_image_res:
                if key not in final_res_single_image.keys():
                    final_res_single_image[key] = [single_image_res[key]]
                else:
                    final_res_single_image[key].append(single_image_res[key])
            
            # CRPS
            crps_res = metrics.get_crps(target=gt, pred=ens_pred)
            for key in crps_res:
                if key not in final_res_single_image.keys():
                    final_res_single_image[key] = [crps_res[key]]
                else:
                    final_res_single_image[key].append(crps_res[key])
            
            # RMSE
            rmse_res = metrics.get_rmse(target=gt, pred=mean_pred)
            for key in rmse_res:
                if key not in final_res_single_image.keys():
                    final_res_single_image[key] = [rmse_res[key]]
                else:
                    final_res_single_image[key].append(rmse_res[key])

            i += gt.shape[0]
            pbar.update(gt.shape[0])
            if i >= num_samples:
                break
    
    # mean for image metrics
    for key in final_res_single_image.keys():
        clean_key = [x for x in final_res_single_image[key] if  not math.isnan(x) and x != float('-inf') and x != float('inf')]
        final_res_single_image[key] = sum(clean_key) / len(clean_key)

    result = metrics.compute()["avg"]
    return {
        "RMSE": final_res_single_image.get("rmse", None),
        "CRPS": final_res_single_image.get("crps", None),
        "SSIM": final_res_single_image.get("ssim", None),
        "PSNR": final_res_single_image.get("psnr", None),
        "HSS": result.get("hss", None),
        "CSI": result.get("csi", None),
    }

# --- Main Loop ---

@click.command()
@click.option("--model_path", type=str, required=True, help="Path to the trained LSM checkpoint.")
@click.option('--sevir_npy_path', type=str, required=True, help="Base path to the SEVIR dataset in npy format.")
@click.option('--txt_path', type=str, default="data/test_data.txt", help="Base path to the train/test split txt files.")
@click.option("--tag", type=str, default="lsm", show_default=True, help="Tag for output folder naming.")
@click.option(
    "--save_pth",
    type=str,
    default="eval_results/lsm_pred",
    show_default=True,
    help="Base path for final metrics.",
)
@click.option(
    "--eval_pth",
    type=str,
    default="eval_results/lsm/pred_eval_data",
    show_default=True,
    help="Base path for saved predictions and ground truth arrays.",
)
@click.option(
    "--batch_size",
    type=int,
    default=4,
    show_default=True,
    help="Batch size for forecasting and evaluation.",
)
@click.option(
    "--num_samples",
    type=int,
    default=-1,
    show_default=True,
    help="Number of test samples to process. Use -1 for full test split.",
)
@click.option(
    "--ensemble_size",
    type=int,
    default=10,
    show_default=True,
    help="Number of stochastic forecasts per sample.",
)
@click.option("--input_length", type=int, default=13, show_default=True, help="Number of conditioning frames.")
@click.option("--pred_length", type=int, default=12, show_default=True, help="Number of forecast frames.")
@click.option("--sample_steps", type=int, default=25, show_default=True, help="Number of LSM sampling steps.")
@click.option(
    "--decoder_sample_steps",
    type=int,
    default=10,
    show_default=True,
    help="Number of first-stage decoder sampling steps.",
)
@click.option(
    "--verbose",
    type=int,
    default=2,
    show_default=True,
    help="Verbosity level. Negative disables progress bars.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Global random seed for deterministic execution.")
def main(
    model_path: str,
    sevir_npy_path: str,
    txt_path: str="data/test_data.txt",
    tag: str = "lsm",
    save_pth: str = "eval_results/lsm_pred",
    eval_pth: str = "eval_results/lsm/pred_eval_data",
    batch_size: int = 4,
    num_samples: int = -1,
    ensemble_size: int = 10,
    input_length: int = 13,
    pred_length: int = 12,
    sample_steps: int = 25,
    decoder_sample_steps: int = 10,
    verbose: int = 2,
    seed: int = 42,
):
    seed_everything(seed)

    # setup
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_pth = Path(save_pth) / tag / now
    save_pth.mkdir(parents=True, exist_ok=True)
    eval_pth = Path(eval_pth) / tag / now
    eval_pth.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = get_lsm(
        ckpt_path=model_path,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()


    # dataset
    dset = CasCastSevirDataset(
        split="test",
        input_length=input_length,
        pred_length=pred_length,
        data_dir=sevir_npy_path,
        txt_dir=txt_path,
    )
    if num_samples <= 0:
        num_samples = len(dset)
    dl = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)
    
    # sample all predictions
    i = 0
    ens_pred_pth = eval_pth / "ens_pred"
    ens_pred_pth.mkdir(parents=True, exist_ok=True)
    gt_pth = eval_pth / "gt"
    gt_pth.mkdir(parents=True, exist_ok=True)
    with tqdm(disable=verbose < 0, total=num_samples) as pbar:
        for data in dl:
            inputs = data['inputs']
            gt = data['data_samples']
            combined = torch.cat([inputs, gt], dim=2)
            orig_dtype = combined.dtype

            combined = normalize(
                combined,
                data_min=0.,
                data_max=1.,
                exp_min=-1.,
                exp_max=1.,
                exp_dtype=dtype
            )
            combined = combined[:min(combined.shape[0], num_samples - i)]

            # get prediction
            sample = combined.to(device=device)
            sample = einops.repeat(sample, "b c t h w -> (b e) t h w c", e=ensemble_size).clone()
            noise = torch.randn_like(sample)
            sample[:, input_length:] = noise[:, input_length:]  # avoid any information leak
            cond_indices = torch.zeros((sample.shape[0], sample.shape[1]), dtype=torch.bool, device=sample.device)
            cond_indices[:, :input_length] = True
            pred = get_pred(
                model=model,
                sample=sample,
                cond_indices=cond_indices,
                sample_steps=sample_steps,
                decoder_sample_steps=decoder_sample_steps, 
                device=device,
                dtype=dtype,
            )
            pred = einops.rearrange(pred, "(b e) t h w c -> b e t h w c", e=ensemble_size)

            # save prediction and gt
            for j in range(pred.size(0)):
                c_pred = pred[j, :, input_length:]
                c_pred = einops.rearrange(c_pred, "e t h w c -> e t c h w")
                c_pred = normalize(
                    c_pred,
                    data_min=-1.,
                    data_max=1.,
                    exp_min=0.,
                    exp_max=1.,
                    exp_dtype=orig_dtype
                )
                c_pred = c_pred.cpu().numpy()
                np.save(ens_pred_pth / f"{i}.npy", c_pred)

                c_gt = gt[j]
                c_gt = einops.rearrange(c_gt, "c t h w -> t c h w")
                c_gt = c_gt.cpu().numpy()
                np.save(gt_pth / f"{i}.npy", c_gt)

                i += 1
                pbar.update()
            if i >= num_samples:
                break
    
    results = eval(
        eval_pth=eval_pth,
        batch_size=batch_size,
        pred_length=pred_length,
        num_samples=num_samples,
        seed=seed,
        verbose=verbose,
    )
    with open(save_pth / "results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
