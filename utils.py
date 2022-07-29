import math
import random
import numpy as np
import torch
import torch.nn as nn
import esm

proteinseq_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']

def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class PositionalEncoding(nn.Module):
    # (B, L, H), batch-first
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ESM:
    def __init__(self, antigen_max_len):
        self.antigen_max_len = antigen_max_len
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.model.eval()
        
    def tokenizing(self, seq):
        return self.alphabet.encode(seq)

    def get_token_repr(self, inp):
        with torch.no_grad():
            results = self.model(inp, repr_layers=[33], return_contacts=False)
        return results["representations"][33]
        
class FocalLoss(nn.Module):
    """
    https://dacon.io/competitions/official/235585/codeshare/1796
    """

    def __init__(self, gamma=5.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # print(self.gamma)
        self.eps = eps
        # self.ce = nn.CrossEntropyLoss(reduction="none")
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        # input : logit
        logp = self.ce(input, target.float())
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()