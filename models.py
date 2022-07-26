from numpy import True_, true_divide
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import PositionalEncoding

class CustomLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.epitope_lstm = nn.LSTM(input_size=embedding_dim, 
                                    hidden_size=hidden_dim, 
                                    batch_first=True,
                                    num_layers=2,
                                    dropout=0.2,
                                    bidirectional=False
                                   )

        self.epitope_lstm2 = nn.LSTM(input_size=embedding_dim, 
                                    hidden_size=hidden_dim, 
                                    batch_first=True,
                                    num_layers=2,
                                    dropout=0.2,
                                    bidirectional=False
                                   )

        self.linears = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(2*hidden_dim, 1),
        
        )

    def forward(self, antigen, antigen_reversed, antigen_full, mask):
        lengths = mask.sum(axis=1).type(torch.long)
        # out = pack_padded_sequence(antigen, lengths, batch_first=True, enforce_sorted=False)
        out1, _ = self.epitope_lstm(antigen)
        out2, _ = self.epitope_lstm(antigen_reversed)
        # out, _ = self.epitope_lstm2(out)
        out1 = out1[range(out1.shape[0]), lengths-1, :]
        out2 = out2[range(out2.shape[0]), lengths-1, :]
        out = torch.concat([out1, out2], axis=1)
        out = self.linears(out)
        return out.squeeze(-1)

class CustomLSTM2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.epitope_lstm = nn.LSTM(input_size=embedding_dim, 
                                    hidden_size=hidden_dim, 
                                    batch_first=True,
                                    num_layers=2,
                                    dropout=0.2,
                                    bidirectional=True
                                   )

        # self.epitope_lstm2 = nn.LSTM(input_size=embedding_dim, 
        #                             hidden_size=hidden_dim, 
        #                             batch_first=True,
        #                             num_layers=2,
        #                             dropout=0.2,
        #                             bidirectional=False
        #                            )

        self.linears = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Linear(2*hidden_dim, 1),
        
        )

    def forward(self, antigen, antigen_reversed, antigen_full, mask, lengths):
        
        out = pack_padded_sequence(antigen, lengths, batch_first=True, enforce_sorted=False)
        out, (_, _) = self.epitope_lstm(antigen)
        # out = pad_packed_sequence(out)
        # out, _ = self.epitope_lstm2(out)
        out = out[range(out.shape[0]), lengths-1, :]

        out = self.linears(out)
        return out.squeeze(-1)

class Linear(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.linears = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

        self.linear1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )

        self.linear2 = nn.Linear(hidden_dim, 1)
        self.linear_full = nn.Linear(2, embedding_dim)

        self.fc = nn.Linear(embedding_dim, 1)
        
    def forward(self, epitope, antigen_full, mask):
        # antigen_full = self.linear_full(antigen_full)
        lengths = mask.sum(axis=1).unsqueeze(1)
        out = epitope.sum(axis=1) / lengths
        # out = out + antigen_full
        # out = self.linears(out).squeeze(-1)
        # out = epitope.mean(axis=1)
        out = self.linears(out).squeeze(axis=-1)
        return out

class Transformer(nn.Module):
    def __init__(self, n_tokens, embedding_dim, d_model, hidden_dim, n_encoder_layers):
        super().__init__()

        # self.epitope_embed = nn.Embedding(num_embeddings=n_tokens, 
        #                                   embedding_dim=embedding_dim, 
        #                                   padding_idx=0
        #                                  )
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=n_tokens)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoder_layers)
        self.linear = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, epitope, mask):
        out = self.transformer_encoder(epitope, src_key_padding_mask=mask)
        out = out.sum(axis=1)
        out = self.linear(out).squeeze(-1)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        return self.conv(x)

class CNN1D(nn.Module):
    def __init__(self, n_tokens, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.epitope_embed = nn.Embedding(num_embeddings=n_tokens, 
                                          embedding_dim=embedding_dim, 
                                          padding_idx=0
                                         )

        self.convs = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim)
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )

        self.linear2 = nn.Linear(1, 1)

    def forward(self, epitope, mask, antigen):
        embeds = self.epitope_embed(epitope) # (B, max_len, emb_dim)
        embeds = embeds.swapaxes(1, 2)       # (B, emb_dim, max_len)
        out = self.convs(embeds)             # (B, hid_dim, max_len)
        out = out.swapaxes(1, 2)             # (B, max_len, hid_dim)
        out = self.linear(out).squeeze(-1) * mask   # (B, max_len)
        # out, _ = out.max(axis=1)
        out = out.sum(axis=1)
        out = self.linear2(out.unsqueeze(-1)).squeeze(-1)
        return out