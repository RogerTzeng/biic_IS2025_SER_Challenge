# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import AutoModel
import importlib
# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--head_dim", type=int, default=1024)

parser.add_argument("--pooling_type", type=str, default="AttentiveStatisticsPooling")
args = parser.parse_args()

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS=args.epochs
LR=args.lr
MODEL_PATH = args.model_path
os.makedirs(MODEL_PATH, exist_ok=True)


import json
from collections import defaultdict
config_path = "configs/config_dim.json"
with open(config_path, "r") as f:
    config = json.load(f)
text_fea_path = 'path/to/roberta_features/dim'
audio_fea_path = 'path/to/wavlm_features/dim'
label_path = config["label_path"]

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_adv_emo_label(label_path, dtype)
    cur_text, cur_audio = utils.load_features(text_fea_path, audio_fea_path, cur_utts)

    cur_feature_set = utils.FeaSet(cur_text, cur_audio)
    
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.ADV_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_feature_set, cur_emo_set, cur_utts])

    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=16,
        collate_fn=utils.collate_fn_fea_lab
    )

ser_model = net.EmotionRegression_Dim_Fea(2048+1024, args.head_dim, 1, 3, dropout=0.5)
##############################################
ser_model.eval(); ser_model.cuda()

ser_opt = torch.optim.AdamW(ser_model.parameters(), LR)

scaler = GradScaler()
ser_opt.zero_grad(set_to_none=True)

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val"])
lm.alloc_stat_type_list(["dev_aro", "dev_dom", "dev_val"])

min_epoch=0
min_loss=1e10

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    lm.init_stat()
    ser_model.train()    
    batch_cnt = 0

    for xy_pair in tqdm(total_dataloader["train"]):
        x_text = xy_pair[0]; x_text=x_text.cuda(non_blocking=True).float()
        x_audio = xy_pair[1]; x_audio=x_audio.cuda(non_blocking=True).float()
        y = xy_pair[2]; y=y.cuda(non_blocking=True).float()
        utt = xy_pair[3][0]
        
        with autocast(enabled=True):
            
            emo_pred = ser_model(x_audio, x_text)
            ccc = utils.CCC_loss(emo_pred, y)
            loss = 1.0 - ccc
            total_loss = torch.sum(loss) / ACCUMULATION_STEP
        scaler.scale(total_loss).backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            scaler.step(ser_opt)
            scaler.update()
            ser_opt.zero_grad(set_to_none=True)
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_aro", ccc[0])
        lm.add_torch_stat("train_dom", ccc[1])
        lm.add_torch_stat("train_val", ccc[2])   

    ser_model.eval() 
    total_pred = [] 
    total_y = []
    for xy_pair in tqdm(total_dataloader["dev"]):
        x_text = xy_pair[0]; x_text=x_text.cuda(non_blocking=True).float()
        x_audio = xy_pair[1]; x_audio=x_audio.cuda(non_blocking=True).float()
        y = xy_pair[2]; y=y.cuda(non_blocking=True).float()
        utt = xy_pair[3][0]
        
        with torch.no_grad():
            emo_pred = ser_model(x_audio, x_text)

            total_pred.append(emo_pred)
            total_y.append(y)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    ccc = utils.CCC_loss(total_pred, total_y)
    # Logging
    lm.add_torch_stat("dev_aro", ccc[0])
    lm.add_torch_stat("dev_dom", ccc[1])
    lm.add_torch_stat("dev_val", ccc[2])

    # Save model
    lm.print_stat()
        
    dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
    if min_loss > dev_loss:
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",3.0-min_loss)
        save_model_list = ["ser", "ssl"]

        torch.save(ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "final_ser.pt"))