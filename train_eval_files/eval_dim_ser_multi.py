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
parser.add_argument("--model_path", type=str, default="./model/wavlm-large")
parser.add_argument("--pooling_type", type=str, default="MeanPooling")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument('--audio_path', type=str)
parser.add_argument('--snr', type=int, default=None)
parser.add_argument('--testset', type=str, default=None)
parser.add_argument('--store_path')
args = parser.parse_args()

SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

import json
from collections import defaultdict
config_path = "configs/config_dim.json"
with open(config_path, "r") as f:
    config = json.load(f)
text_fea_path = 'path/to/roberta_features/dim'
audio_fea_path = 'path/to/wavlm_features/dim'
label_path = 'path/to/labels_consensus.csv'

import pandas as pd
import numpy as np

files_test3 = [filename for filename in os.listdir(audio_fea_path) if 'test3' in filename]

dtype = "test3"

total_dataset=dict()
total_dataloader=dict()

cur_text, cur_audio = utils.load_features(text_fea_path, audio_fea_path, files_test3)
cur_feature_set = utils.FeaSet(cur_text, cur_audio)
total_dataset[dtype] = utils.CombinedSet([cur_feature_set, files_test3])
total_dataloader[dtype] = DataLoader(
    total_dataset[dtype], batch_size=1, shuffle=False, 
    pin_memory=True, num_workers=16,
    collate_fn=utils.collate_fn_fea_test3
)

ser_model = net.EmotionRegression_Dim_Fea(2048+1024, args.head_dim, 1, 3, dropout=0.5)
##############################################
ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()

min_epoch=0
min_loss=1e10

ser_model.eval() 

lm = utils.LogManager()
for dtype in ["test3"]:
    lm.alloc_stat_type_list([f"{dtype}_aro", f"{dtype}_dom", f"{dtype}_val"])

min_epoch=0
min_loss=1e10

lm.init_stat()

ser_model.eval() 


INFERENCE_TIME=0
FRAME_SEC = 0
for dtype in ["test3"]:
    total_pred = []
    total_utt = []
    for xy_pair in tqdm(total_dataloader[dtype]):
        x_text = xy_pair[0]; x_text=x_text.cuda(non_blocking=True).float()
        x_audio = xy_pair[1]; x_audio=x_audio.cuda(non_blocking=True).float()
        utts = xy_pair[2]

        stime = perf_counter()
        with torch.no_grad():
            emo_pred = ser_model(x_audio, x_text)

            total_pred.append(emo_pred)
            total_utt.append(utts)
        etime = perf_counter()
        INFERENCE_TIME += (etime-stime)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)

    data = []
    for pred, utt in zip(total_pred, total_utt):
        # print(pred)
        pred_values =  pred.cpu().tolist()
        # print(pred_values)
        data.append([utt[0].replace('.pkl', '.wav'), min(max(1, pred_values[0] * 6 + 1), 7),min(max(1, pred_values[2] * 6 + 1), 7),min(max(1, pred_values[1] * 6 + 1), 7)])

    # print(data)
    # Writing to CSV file
    os.makedirs(MODEL_PATH + '/results', exist_ok=True) 
    csv_filename = MODEL_PATH + '/results/' + dtype + '.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["FileName", "EmoAct", "EmoVal", "EmoDom"])
        writer.writerows(data)


lm.print_stat()
print("Inference time", INFERENCE_TIME, "sec")
