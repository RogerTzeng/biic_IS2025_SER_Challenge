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
from sklearn.metrics import precision_score, recall_score, f1_score
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
parser.add_argument("--pooling_type", type=str, default="AttentiveStatisticsPooling")
parser.add_argument("--head_dim", type=int, default=1024)
parser.add_argument('--store_path')
args = parser.parse_args()

SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

import json
from collections import defaultdict
config_path = "configs/config_cat.json"
with open(config_path, "r") as f:
    config = json.load(f)
text_fea_path = 'path/to/roberta_features/cat'
audio_fea_path = 'path/to/wavlm_features/cat'
label_path = config["label_path"]
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv(label_path)

# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']

# Classes (emotions)
classes = ['Angry', 'Sad', 'Happy', 'Surprise', 'Fear', 'Disgust', 'Contempt', 'Neutral']

# Calculate class frequencies
class_frequencies = train_df[classes].sum().to_dict()

# Total number of samples
total_samples = len(train_df)

# Calculate class weights
class_weights = {cls: total_samples / (len(classes) * freq) if freq != 0 else 0 for cls, freq in class_frequencies.items()}

print(class_weights)

# Convert to list in the order of classes
weights_list = [class_weights[cls] for cls in classes]

# Convert to PyTorch tensor
class_weights_tensor = torch.tensor(weights_list, device='cuda', dtype=torch.float)


# Print or return the tensor
print(class_weights_tensor)


files_test3 = [filename for filename in os.listdir(audio_fea_path) if 'test3' in filename]

dtype = "test3"

total_dataset=dict()
total_dataloader=dict()

cur_text, cur_audio = utils.load_features(text_fea_path, audio_fea_path, files_test3)
cur_feature_set = utils.FeaSet(cur_text, cur_audio)
total_dataset[dtype] = utils.CombinedSet([cur_feature_set, files_test3])
total_dataloader[dtype] = DataLoader(
    total_dataset[dtype], batch_size=1, shuffle=False, 
    pin_memory=True, num_workers=4,
    collate_fn=utils.collate_fn_fea_test3
)

print("Loading pre-trained ", SSL_TYPE, " model...")

ser_model = net.EmotionRegression_Fea(2048+1024, args.head_dim, 1, len(classes), dropout=0.5)
##############################################
ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()


lm = utils.LogManager()
for dtype in ["test3"]:
    lm.alloc_stat_type_list([f"{dtype}_loss"])

min_epoch=0
min_loss=1e10

lm.init_stat()

ser_model.eval() 

INFERENCE_TIME=0
FRAME_SEC = 0

for dtype in ["test3"]:
    total_pred = [] 
    total_y = []
    total_utt = []
    for xy_pair in tqdm(total_dataloader[dtype]):
        x_text = xy_pair[0]; x_text=x_text.cuda(non_blocking=True).float()
        x_audio = xy_pair[1]; x_audio=x_audio.cuda(non_blocking=True).float()
        fname = xy_pair[2]
        
        stime = perf_counter()
        with torch.no_grad():
            emo_pred = ser_model(x_audio, x_text)

            total_pred.append(emo_pred)
            total_utt.append(fname)

        etime = perf_counter()
        INFERENCE_TIME += (etime-stime)

emotion_mapping = {
    0: 'A',  # Angry
    1: 'S',  # Sad
    2: 'H',  # Happy
    3: 'U',  # Surprise
    4: 'F',  # Fear
    5: 'D',  # Disgust
    6: 'C',  # Contempt
    7: 'N'   # Neutral
}

data = []
for pred, utt in zip(total_pred, total_utt):
    # Get the predicted class index (maximum logit)
    predicted_class = pred.cpu().numpy().argmax()
    # Map the index to the corresponding categorical letter
    emo_class = emotion_mapping[predicted_class]
    # Append filename and emotion class to the data
    data.append([utt[0].replace('.pkl', '.wav'), emo_class])

# Writing to CSV file
os.makedirs(MODEL_PATH + '/results', exist_ok=True)
csv_filename = MODEL_PATH + '/results/' + dtype + '.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['FileName', 'EmoClass'])
    writer.writerows(data)

print(f"Results saved to {csv_filename}")
