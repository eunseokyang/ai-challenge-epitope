import os
from os.path import join as pjoin
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import esm

from utils import proteinseq_toks

# torch.multiprocessing.set_start_method('spawn')

class CustomTokenizer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.proteinseq_toks = proteinseq_toks
        self.pad_tok = '<pad>'
        self.cls_tok = '<cls>'

        all_toks = [self.pad_tok, self.cls_tok] + self.proteinseq_toks
        self.num_tokens = len(all_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(all_toks)}
        self.idx_to_tok = {i: tok for i, tok in enumerate(all_toks)}

    def tokenizing(self, seq):
        seq = list(seq[:self.max_len])
        result = [self.tok_to_idx[self.pad_tok]] * self.max_len
        # result[0] = self.tok_to_idx[self.cls_tok]
        for i, tok in enumerate(seq):
            result[i] = self.tok_to_idx[tok]
        return result

class AntigenDataset(Dataset):
    def __init__(self, antigens):
        self.antigens = antigens

    def __getitem__(self, idx):
        return self.antigens[idx]

    def __len__(self):
        return len(self.antigens)


class Tokenizer:
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def tokenize(self, seq, max_len=1024):
        assert len(seq) <= max_len - 2
        result = [self.alphabet.padding_idx] * max_len
        result[0] = self.alphabet.cls_idx
        for i, tok in enumerate(seq):
            result[i+1] = self.alphabet.tok_to_idx[tok]
        result[len(seq)+1] = self.alphabet.eos_idx
        return result

class Preprocessor:
    def __init__(self, mode='train', data_dir='./data/', save_dir='/data2/eunseok/antigen/', antigen_max_len=1022, epitope_max_len=62):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mode = mode
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.antigen_max_len = antigen_max_len
        self.epitope_max_len = epitope_max_len

        self.esm_model, self.esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        # self.antigen_batch_converter = self.esm_alphabet.get_batch_converter()
        self.esm_model.eval()
        self.esm_model.to(self.device)

        self.tokenizer = Tokenizer(self.esm_alphabet)
        self.batch_size = 32
        
        self.data = pd.read_csv(pjoin(data_dir, f'{mode}.csv'))
        os.makedirs(pjoin(self.save_dir, self.mode), exist_ok=True)

    def save_df(self, df):
        df.to_csv(pjoin(self.data_dir, f'{self.mode}_processed_{self.antigen_max_len+2}.csv'), index=False)

    def save_antigen_repr_sep(self, out, fname):
        np.save(pjoin(self.save_dir, self.mode, f'{fname}'), out)

    def save_antigen_repr(self, out, idx):
        for j, ant in enumerate(out):
            np.save(pjoin(self.save_dir, self.mode, f'{idx+j}'), ant)

    def inference(self, antigens):
        antigens = torch.tensor(antigens, dtype=torch.int32)
        for i in tqdm(range(0, antigens.shape[0], self.batch_size)):
            ant = antigens[i:i+self.batch_size].to(self.device)
            with torch.no_grad():
                results = self.esm_model(ant, repr_layers=[33], return_contacts=False)
            out = results["representations"][33]
            
            # out = out.cpu().detach().numpy()
            # with ProcessPoolExecutor(8) as exe:
            #     # submit tasks to generate files
            #     _ = [exe.submit(self.save_antigen_repr_sep, out[j], i+j) for j in range(out.shape[0])]
            
            self.save_antigen_repr(out.cpu().detach().numpy(), i)

    def preprocess(self, is_train=True):
        epitope_seq = self.data['epitope_seq'].values
        epitope_seq_len = self.data['epitope_seq'].str.len().values
        antigen_seq = self.data['antigen_seq'].values
        start_pos = self.data['start_position'].values.astype(int) # index starts from 1 !!
        end_pos = self.data['end_position'].values.astype(int)
        if is_train:
            label = self.data['label'].values
        else:
            label = None

        # custom tokenizing
        # tokenized_epitopes = []
        # for ept in epitope_seq:
        #     ept = ept[:self.epitope_max_len]
        #     tokenized_epitopes.append(self.tokenizer.tokenize(ept, self.epitope_max_len+2))
        # tokenized_epitopes = np.array(tokenized_epitopes)

        # cut antigen seq
        assert self.antigen_max_len % 2 == 0
        for i in tqdm(range(len(antigen_seq))):
            if end_pos[i] - start_pos[i] >= self.epitope_max_len:
                end_pos[i] = start_pos[i] + self.epitope_max_len - 1
            if len(antigen_seq[i]) > self.antigen_max_len:
                middle = (end_pos[i] + start_pos[i]) // 2
                left = middle - self.antigen_max_len // 2
                right = middle + self.antigen_max_len // 2 - 1
                if left < 1:
                    shift = 1 - left
                    left += shift
                    right += shift
                if right > len(antigen_seq[i]):
                    shift = right - len(antigen_seq[i])
                    left -= shift
                    right -= shift
                antigen_seq[i] = antigen_seq[i][left-1:right]
                start_pos[i] -= (left - 1)
                end_pos[i] -= (left - 1)

                assert start_pos[i] >= 1
                assert end_pos[i] <= self.antigen_max_len

                # if end_pos[i] <= self.antigen_max_len:
                #     antigen_seq[i] = antigen_seq[i][:self.antigen_max_len]
                # else:
                #     diff = end_pos[i] - self.antigen_max_len
                #     antigen_seq[i] = antigen_seq[i][diff:diff+self.antigen_max_len]
                #     start_pos[i] -= diff
                #     end_pos[i] -= diff

        # encoding antigen
        tokenized_antigens = []

        antigen_seq_unique = sorted(list(set(antigen_seq)))
        antigen_to_idx = dict(zip(antigen_seq_unique, list(range(len(antigen_seq_unique)))))
        antigen_indices = np.array([antigen_to_idx[ant] for ant in antigen_seq], dtype=int)
        
        for ant in tqdm(antigen_seq_unique):
            tokenized_antigens.append(self.tokenizer.tokenize(ant, self.antigen_max_len+2))
        tokenized_antigens = np.array(tokenized_antigens)
        
        # batch_antigen = [(str(i), ant) for i, ant in enumerate(antigen_seq_unique)]
        # _, _, tokenized_antigens = self.antigen_batch_converter(batch_antigen)

        assert tokenized_antigens.shape[1] == self.antigen_max_len + 2

        result_df = pd.DataFrame({
            'antigen_indices': antigen_indices,
            'antigen_seq': antigen_seq,
            'epitope_seq_len': epitope_seq_len,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'label': label
        })

        np.save(pjoin(self.save_dir, f'tokenized'), tokenized_antigens)
        self.save_df(result_df)
        # self.inference(tokenized_antigens)

if __name__ == '__main__':
    antigen_max_len = 1022
    preprocessor = Preprocessor(mode='validation',
                                data_dir='./data/',
                                save_dir=f'/data2/eunseok/antigen_{antigen_max_len+2}/', 
                                antigen_max_len=antigen_max_len,
                                epitope_max_len=62,
                                )
    preprocessor.preprocess()


    