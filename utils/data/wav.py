import os
import torch
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import joblib

# Load audio
def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav
def load_audio(audio_path, utts, nj=24):
    # Audio path: directory of audio files
    # utts: list of utterance names with .wav extension
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]

    wavs = joblib.Parallel(n_jobs=nj)(
        joblib.delayed(extract_wav)(path) for path in tqdm(wav_paths, desc="Loading audio files")
    )

    return wavs

def extract_text(text_path):
    with open(text_path,"r") as f:
        string = f.read()
    return string

def load_text(text_path, utts, nj=24):
    text_paths = [os.path.join(text_path, utt.replace('.wav', '.txt')) for utt in utts]
    texts = joblib.Parallel(n_jobs=nj)(
        joblib.delayed(extract_text)(path) for path in tqdm(text_paths, desc="Loading text files")
    )

    return texts

def extract_feat(feature_path):
    fea = joblib.load(feature_path)
    if isinstance(fea, torch.Tensor):
        fea = fea.detach()
    return fea

def load_features(text_fea_path, audio_fea_path, utts, nj=16):
    text_feature_paths = [os.path.join(text_fea_path, utt.replace('.wav', '.pkl')) for utt in utts]
    audio_feature_paths = [os.path.join(audio_fea_path, utt.replace('.wav', '.pkl')) for utt in utts]

    def load_both_features(text_path, audio_path):
        text_feature = extract_feat(text_path)
        audio_feature = extract_feat(audio_path)
        return text_feature, audio_feature
    
    print("Total len:", len(utts))
    combined_features = joblib.Parallel(n_jobs=nj)(
        joblib.delayed(load_both_features)(text_path, audio_path)
        for text_path, audio_path in tqdm(zip(text_feature_paths, audio_feature_paths), desc="Loading features")
    )
    text, audio = map(list, zip(*combined_features))
    return text, audio