import argparse

from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import wandb

from preprocessor import Preprocessor
from dataloader import EpitopeDataset
from utils import *
from models import *

now = datetime.now().strftime('%y%m%d_%H%M%S')
checkpoint_dir = Path(f'./checkpoints/{now}/')

def evaluate(model, val_loader, device, thresholds=(0.1, 0.3, 0.5, 0.8)):
    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([8]).to(device))
    
    loss_fn = FocalLoss()

    n_eval_batches = len(val_loader)
    losses = 0

    preds = []
    labels = []

    model.eval()

    for batch, (epitope, mask, antigen_full, label) in tqdm(enumerate(val_loader)):
        epitope = epitope.to(device)
        mask = mask.to(device)
        antigen_full = antigen_full.to(device)
        label = label.to(device)

        with torch.no_grad():
            pred = model(epitope, antigen_full, mask)
            loss = loss_fn(pred, label)
            losses += loss

        preds += pred.to('cpu').tolist()
        labels += label.to('cpu').tolist()

    val_f1s = dict()
    for thr in thresholds:
        pred_labels = np.where(np.array(preds) > thr, 1, 0)
        val_f1 = f1_score(labels, pred_labels, average='macro')
        val_f1s[f'validation f1 {thr}'] = val_f1

    model.train()
    return losses / n_eval_batches, val_f1s

def load_data(data_dir):
    train_data = pd.read_csv(data_dir + './train_processed.csv')
    val_data = pd.read_csv(data_dir + './validation_processed.csv')

    # pos_data = train_data[train_data['label'] == True]
    # pos_data_concat = pd.concat([pos_data for _ in range(9)], ignore_index=True)
    # train_data = pd.concat([train_data, pos_data_concat], ignore_index=True)

    return train_data, val_data

def train(cfg, experiment):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading data
    train_processed, val_processed = load_data(cfg.data_dir)

    train_dataset = EpitopeDataset(train_processed, data_dir='/data2/eunseok/antigen_256/train/')
    val_dataset = EpitopeDataset(val_processed, data_dir='/data2/eunseok/antigen_256/validation/')

    print(f"train len: {len(train_dataset)}")
    print(f"val len: {len(val_dataset)}")

    ## TEST
    # total_len = len(val_dataset)
    # train_dataset_, val_dataset_ = torch.utils.data.random_split(val_dataset, [int(total_len*0.7), total_len-int(total_len*0.7)])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    ## TEST END

    # model = CNN1D(n_tokens)
    # model = Transformer(
    #     n_tokens=n_tokens, 
    #     embedding_dim=cfg.embedding_dim, 
    #     d_model=cfg.d_model,
    #     hidden_dim=cfg.hidden_dim,
    #     n_encoder_layers=cfg.n_encoder_layers
    # ) 
    model = Linear(embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim)
    # model = CustomLSTM(embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 10000)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), pct_start=0.1, epochs=epochs)

    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([8]).to(device))
    loss_fn = FocalLoss()
    
    global_step = 0
    for epoch in range(1, cfg.epochs+1):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch}/{cfg.epochs}', unit='epitope') as pbar:
            for batch, (epitope, mask, antigen_full, label) in enumerate(train_loader):
                epitope = epitope.to(device)
                mask = mask.to(device)
                antigen_full = antigen_full.to(device)
                label = label.to(device)

                pred = model(epitope, antigen_full, mask)
                loss = loss_fn(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step(epoch - 1 + batch/len(train_loader))

                pbar.update(epitope.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss': loss.item()})

            histograms = {}
            # for tag, value in model.named_parameters():
            #     tag = tag.replace('/', '.')
            #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            # eval
            val_loss, val_f1s = evaluate(model, val_loader, device)

            print(f"Validation F1-score: {val_f1s}")
            print(f"Validation Loss: {val_loss}")

            # scheduler.step()
            experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation loss': val_loss,
                        **val_f1s,
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

            if cfg.save_checkpoint:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
                # logging.info(f'Checkpoint {epoch} saved!')


if __name__=='__main__':
    cfg = OmegaConf.load('./config.yaml')
    experiment = wandb.init(
        project='Epitope',
        config=OmegaConf.to_container(cfg),
        # name=cfg.run_name,
    #    mode='disabled'
    )

    seed_all(seed=cfg.seed)

    train(cfg=cfg,
          experiment=experiment
          )