import os
from os.path import join as pjoin
import numpy as np
import torch
from torch.utils.data import Dataset

class EpitopeDataset(Dataset):
    def __init__(self, processed, data_dir, epitope_max_len=62, is_train=True):
        self.processed = processed
        self.data_dir = data_dir
        self.is_train = is_train
        self.tokenized_len = epitope_max_len + 2
        
    def __getitem__(self, idx):
        info = self.processed.iloc[idx, :]
        ## check this
        antigen_full = int(len(info['antigen_seq']) == self.tokenized_len - 2)
        # seq_len = info['epitope_seq_len']
        seq_len = info['end_pos'] - info['start_pos'] + 1
        fname = info['antigen_indices']
        antigen = np.load(pjoin(self.data_dir, f"{fname}.npy"))
        epitope = np.zeros((self.tokenized_len, antigen.shape[-1]))
        epitope[:seq_len, :] = antigen[info['start_pos']:info['end_pos']+1]
        
        mask = np.zeros(self.tokenized_len)
        mask[:seq_len] = 1
        
        epitope = torch.tensor(epitope, dtype=torch.float32)
        antigen_full = torch.tensor(np.eye(2)[antigen_full], dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        if self.is_train:
            label = info['label']
            label = torch.tensor(label, dtype=torch.float32)
            return epitope, mask, antigen_full, label
        else:
            return epitope, mask, antigen_full

    def __len__(self):
        return len(self.processed)