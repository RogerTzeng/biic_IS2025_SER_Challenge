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
import csv
from time import perf_counter
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from transformers import AutoModel

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--model_path", type=str, default="./model/cat_ser/7")
parser.add_argument("--pooling_type", type=str, default="AttentiveStatisticsPooling")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument('--audio_path', type=str, default="/path/to/MSP-PODCAST/Audios")#MSP-PODCAST-Publish-1.11/MSP-PODCAST-IS-CHALLENGE
args = parser.parse_args()

SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

import json
from collections import defaultdict
config_path = "configs/config_cat.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = args.audio_path

label_path = 'path/to/processed_labels.csv'

import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv(label_path)

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']
class_frequencies = train_df[classes].sum().to_dict()
total_samples = len(train_df)
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}
print(class_weights)

weights_list = [class_weights[cls] for cls in classes]
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)
print(class_weights_tensor)

dtype_list = ["train", "dev"]
# dtype_list = ["test1", "test2"]
# dtype_list = ["test3"]

total_dataset=dict()
total_dataloader=dict()

if "test3" not in dtype_list:
    for dtype in dtype_list:
        cur_utts, cur_labs = utils.load_cat_emo_label(label_path, dtype)
        print(len(cur_utts))

        cur_wavs = utils.load_audio(audio_path, cur_utts)
        wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
        cur_emo_set = utils.CAT_EmoSet(cur_labs)

        total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set, cur_utts])
        total_dataloader[dtype] = DataLoader(
            total_dataset[dtype], batch_size=1, shuffle=False, 
            pin_memory=True, num_workers=16,
            collate_fn=utils.collate_fn_wav_lab_mask
        )
        
else:
    files_test3 = [filename for filename in os.listdir(audio_path) if 'test3' in filename]
    for dtype in dtype_list:   
        cur_wavs = utils.load_audio(audio_path, files_test3)
        wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
        total_dataset[dtype] = utils.CombinedSet([cur_wav_set, files_test3])
        total_dataloader[dtype] = DataLoader(
            total_dataset[dtype], batch_size=1, shuffle=False, 
            pin_memory=True, num_workers=16,
            collate_fn=utils.collate_fn_wav_test3
        )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
ssl_model.freeze_feature_encoder()
ssl_model.load_state_dict(torch.load(MODEL_PATH+"/final_ssl.pt"))
ssl_model.eval(); ssl_model.cuda()
########## Implement pooling method ##########
feat_dim = 1024

pool_net = getattr(net, args.pooling_type)
attention_pool_type_list = ["AttentiveStatisticsPooling"]
if args.pooling_type in attention_pool_type_list:
    is_attentive_pooling = True
    pool_model = pool_net(feat_dim)
    pool_model.load_state_dict(torch.load(MODEL_PATH+"/final_pool.pt"))
else:
    is_attentive_pooling = False
    pool_model = pool_net()
print(pool_model)

pool_model.eval()
pool_model.cuda()
concat_pool_type_list = ["AttentiveStatisticsPooling"]
dh_input_dim = feat_dim * 2 \
    if args.pooling_type in concat_pool_type_list \
    else feat_dim

ser_model = net.EmotionRegression(dh_input_dim, args.head_dim, 1, len(classes), dropout=0.5)
##############################################
ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()

ssl_model.eval()
ser_model.eval() 

for dtype in dtype_list:
    total_pred = [] 
    total_y = []
    total_utt = []
    for xy_pair in tqdm(total_dataloader[dtype]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        mask = xy_pair[1]; mask=mask.cuda(non_blocking=True).float()
        fname = xy_pair[2][0]
        
        stime = perf_counter()
        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state # (B, T, 1024)
            ssl = pool_model(ssl, mask)
            ssl = ssl.cpu().squeeze(0)
            joblib.dump(ssl, os.path.join('/path/to/wavlm_features/cat', fname.replace('.wav', '.pkl')), compress=2)