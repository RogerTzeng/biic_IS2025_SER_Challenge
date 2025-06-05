from transformers import AutoTokenizer, RobertaModel
import torch
from glob import glob
from tqdm import tqdm
import joblib
import os

file_path = glob('/path/to/Transcripts/*')

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
model = RobertaModel.from_pretrained("FacebookAI/roberta-large")
model.load_state_dict(torch.load("/path/to/pretrained_model/"))

model.eval()
model.cuda()
for file in tqdm(file_path):
    with open(file,"r") as f:
        string = f.read()
    filename = os.path.basename(file)
    inputs = tokenizer(string, return_tensors="pt")
    inputs = {key: value.cuda() for key, value in inputs.items()}
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    last_hidden_states = last_hidden_states.squeeze(0).cpu()
    joblib.dump(last_hidden_states, os.path.join('/path/to/roberta_features', filename.replace('.txt', '.pkl')), compress=2)