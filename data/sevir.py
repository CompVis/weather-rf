import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import h5py
import einops
from collections import deque
import random
from copy import deepcopy

def seed_everything(seed: int):
    random.seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

class SEVIRDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        sevir_pth: str,
        seq_len: int = 25,
        normalize: bool = True,
        train: bool = True,
        val_files_index: str="data/sevir_val_files.txt",
        use_duplicate_validation_seq: bool=True,
    ) -> None:
        super().__init__()

        assert 0 < seq_len <= 49, 'SEVIR data contains 49 frames, please use seq_len between 1 and 49'

        self.channels = ['vil']
        self.seq_len = seq_len
        self.data_pth = os.path.join(sevir_pth, 'data/')
        self.norm_data = normalize
        self.train = train
        self.val_files = self._get_val_files(val_files_index)
        self.use_duplicate_validation_seq = use_duplicate_validation_seq

        self.min_max_values = {                 # determined over entire non-normalized training ds
            'ir069': (-327.67999267578125, 22.869998931884766),
            'ir107': (-327.6799926757812, 56.189998626708984),
            'vil': (0.0, 255.0),
        }
        catalog_pth = os.path.join(sevir_pth, 'CATALOG.csv')
        self.catalog = pd.read_csv(catalog_pth, parse_dates=['time_utc'], low_memory=False)

        all_val_ids = set([x[2] for x in self.val_files])
        if train:
            self.catalog.drop(self.catalog[self.catalog['id'].isin(all_val_ids)].index, inplace=True)
        else:
            self.catalog = self.catalog[self.catalog['id'].isin(all_val_ids)]
        
        self.samples = self._get_samples_list()
        self.samples.index = self.samples.index.to_flat_index()
        self.samples = self.samples.reset_index()

        self.hdf_files = self._open_files()
        self.val_files = self._clean_val_files()

        if train:
            self.length = self.samples.shape[0] * (49 - seq_len + 1)
        else:
            self.length = len(self.val_files)
    
    def _clean_val_files(self) -> list:
        val_files_clean = []
        for v in self.val_files:
            v_id = v[2]
            if v_id in self.samples[f'{self.channels[0]}_index'].values:
                if not self.use_duplicate_validation_seq:
                    if v_id not in [x[2] for x in val_files_clean]:
                        val_files_clean.append(v)
                else:
                    val_files_clean.append(v)
        return val_files_clean
    
    def _get_val_files(self, val_idx_pth: str) -> list:
        val_files = []
        with open(val_idx_pth, 'r') as file:
            for line in file:
                parts = line.strip().split(',')  # Split by commas
                # Convert to appropriate types
                val_files.append((parts[0], int(parts[1]), parts[2], int(parts[3])))
        return val_files
    
    def _open_files(self):
        hdf_filenames = []
        for t in self.channels:
            hdf_filenames += list(np.unique(self.samples[f'{t}_filename'].values))
        hdf_files = {}
        for f in hdf_filenames:
            hdf_files[f] = h5py.File(self.data_pth + '/' + f, 'r')
        return hdf_files

    def _get_samples_list(self) -> pd.DataFrame:
        img_types = set(self.channels)

        # get all rows where one of the image types is present in img_type column
        filtered_catalog = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in img_types]) ]
        # remove rows where one of the image types is missing
        filtered_catalog = filtered_catalog.groupby('id').filter(lambda x: img_types.issubset(set(x['img_type'])))
        # remove repeated ids
        filtered_catalog = filtered_catalog.groupby('id').filter(lambda x: x.shape[0]==len(img_types))
        grouped = filtered_catalog.groupby('id')
        # pandas>=2.2 deprecates passing grouping columns to apply(); pandas>=3 may drop them.
        try:
            return grouped.apply(lambda df: self._df_to_series(df, img_types), include_groups=False)
        except TypeError:
            # Compatibility with older pandas versions that do not support include_groups.
            return grouped.apply(lambda df: self._df_to_series(df, img_types))
    
    def _df_to_series(self, df, img_types):
        """Creates a new DataFrame with one row per sample"""
        group_id = getattr(df, "name", None)
        df = df.set_index('img_type')
        d = {}
        for i in img_types:
            s = df.loc[i]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]

            # Use group key when available, because `id` may be absent in newer pandas groupby.apply behavior.
            idx = group_id if group_id is not None else s['id']
            file_idx = s['file_index']

            d.update( {f'{i}_filename':[s['file_name']], 
                        f'{i}_index':[idx],
                        f'{i}_file_index': [file_idx]} )
        return pd.DataFrame(d)
    
    def normalize(self, x: np.array, img_type: str) -> np.array:
        x_norm = x

        if self.norm_data:
            orig_dtype = x_norm.dtype
            # increase precision to avoid errors
            x_norm = x_norm.astype(np.float64)
            # normalize to 0-1 range
            x_norm = (x_norm - self.min_max_values[img_type][0]) / (self.min_max_values[img_type][1] - self.min_max_values[img_type][0] + 1e-10)
            # normalize to -1-1 range
            x_norm = x_norm * 2 - 1
            x_norm = x_norm.astype(orig_dtype)
        return x_norm
    
    def denormalize(self, x: np.array, img_type: str) -> np.array:
        if self.norm_data:
            orig_dtype = x.dtype
            x = x.astype(np.float64)
            x = (x + 1) / 2   # back to 0-1 range
            x = x * (self.min_max_values[img_type][1] - self.min_max_values[img_type][0] + 1e-10) + self.min_max_values[img_type][0]
            x = x.astype(orig_dtype)
        return x
    
    def _read_data(self, row, start_frame):
        img_types = np.unique([x.split('_')[0] for x in list(row.keys()) if x != 'index'])
        if self.train:
            t_slice = slice(start_frame, start_frame+self.seq_len)
        else:
            # t_slice is predetermined in test mode
            event_id = row[f'{img_types[0]}_index']
            test_start_frame_type = [tup for tup in self.val_files if event_id in tup][0][3]
            if test_start_frame_type == 0:
                test_start_frame = 0
            if test_start_frame_type == 1:
                test_start_frame = 12
            if test_start_frame_type == 2:
                test_start_frame = 24
            t_slice = slice(test_start_frame, test_start_frame+self.seq_len)
        
        data = {}
        for t in img_types:
            fname = row[f'{t}_filename']
            idx = row[f'{t}_file_index']
            hdf_file = self.hdf_files[fname]
            c_data = hdf_file[t][idx, :, :, t_slice]
            data[t] = c_data
        return data

    def _get_metadata(self, idx):
        """
        Returns the SEVIR metadata for batch index
        """
        cols = ['id', 'event_id', 'time_utc','llcrnrlat','llcrnrlon','urcrnrlat','urcrnrlon','proj','height_m','width_m']            
        data = self.samples.iloc[idx]
        meta = self.catalog[self.catalog.id == data['index'][0]].iloc[0]
        meta = meta[cols]
        for col in meta.keys():
            meta[col] = str(meta[col]) if type(meta[col]) != str and type(meta[col]) != int and type(meta[col]) != float else meta[col]
        
        meta_dict = meta.to_dict()
        return meta_dict

    def __len__(self):
        return self.length
    

    def __getitem__(self, idx):
        if self.train:
            row_idx = idx // (49 - self.seq_len + 1)
            start_frame = idx % (49 - self.seq_len + 1)
        else:
            row_idx = self.samples.index[self.samples[f'{self.channels[0]}_index'] == self.val_files[idx][2]].tolist()[0]
            row_idx = self.samples[self.samples[f'{self.channels[0]}_index'] == self.val_files[idx][2]].index.tolist()[0]
            start_frame = -1
        
        data = self._read_data(self.samples.iloc[row_idx], start_frame)
        X = [data[t].astype(np.float32) for t in self.channels]
        X = [self.normalize(x, img_type) for x, img_type in zip(X, self.channels)]
        X = [torch.from_numpy(x) for x in X]
        X = [einops.rearrange(x, "h w t -> t h w") for x in X]
        X = X[-1].unsqueeze(-1)  # add channel dimension for VIL

        result = {}
        result['x'] = X

        result['metadata'] = self._get_metadata(row_idx)
        result['metadata']['start_frame'] = int(start_frame)

        return result

class SevirDataModule:

    def __init__(
        self,
        sevir_pth: str,
        seq_len: int = 25,
        normalize: bool = True,
        batch_size: int = 16,
        val_batch_size: int | None = None,
        prefetch_factor: int = 2,
        num_workers: int = 4,
        val_num_workers: int | None = None,
        val_prefetch_factor: int | None = None,
        val_files_index: str="data/sevir_val_files.txt",
        use_duplicate_validation_seq: bool=True,
        seed: int=42
    ) -> None:
        seed_everything(seed)

        self.train_ds = SEVIRDataset(
            sevir_pth=sevir_pth,
            seq_len=seq_len,
            normalize=normalize,
            train=True,
            val_files_index=val_files_index,
            use_duplicate_validation_seq=use_duplicate_validation_seq,
        )
        self.val_ds = SEVIRDataset(
            sevir_pth=sevir_pth,
            seq_len=seq_len,
            normalize=normalize,
            train=False,
            val_files_index=val_files_index,
            use_duplicate_validation_seq=use_duplicate_validation_seq,
        )

        self.train_batch_size = batch_size
        self.train_num_workers = num_workers
        self.train_prefetch_factor = prefetch_factor

        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.val_num_workers = val_num_workers if val_num_workers is not None else num_workers
        self.val_prefetch_factor = val_prefetch_factor if val_prefetch_factor is not None else prefetch_factor

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            prefetch_factor=self.val_prefetch_factor,
        )

    def test_dataloader(self):
        return self.val_dataloader()
