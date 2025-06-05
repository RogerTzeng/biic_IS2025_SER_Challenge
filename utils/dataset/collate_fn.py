import torch
import torch.nn as nn
import numpy as np
import joblib

def collate_fn_wav_lab_mask(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask, total_utt


def collate_fn_wav_test3(batch):
    total_wav = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        total_wav.append(torch.Tensor(wav))
        total_dur.append(dur)
        total_utt.append(wav_data[1])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, attention_mask, total_utt

def collate_fn_fea_lab(batch):
    total_text_fea = []
    total_audio_fea = []
    total_lab = []
    total_utt = []
    for fea_data in batch:

        text_fea, audio_fea = fea_data[0]
        lab = fea_data[1]
        total_text_fea.append(torch.mean(text_fea, dim=0))
        total_audio_fea.append(audio_fea)
        total_lab.append(lab)
        total_utt.append(fea_data[2])
    
    total_text_fea = nn.utils.rnn.pad_sequence(total_text_fea, batch_first=True)  # Pad to ensure equal lengths
    total_audio_fea = nn.utils.rnn.pad_sequence(total_audio_fea, batch_first=True)  # Pad to ensure equal lengths
    total_lab = torch.Tensor(np.array(total_lab))

    return total_text_fea, total_audio_fea, total_lab, total_utt

def collate_fn_fea_test3(batch):
    total_text_fea = []
    total_audio_fea = []
    total_utt = []
    for fea_data in batch:

        text_fea, audio_fea = fea_data[0]
        total_text_fea.append(torch.mean(text_fea, dim=0))
        total_audio_fea.append(audio_fea)
        total_utt.append(fea_data[1])
    
    total_text_fea = nn.utils.rnn.pad_sequence(total_text_fea, batch_first=True)  # Pad to ensure equal lengths
    total_audio_fea = nn.utils.rnn.pad_sequence(total_audio_fea, batch_first=True)  # Pad to ensure equal lengths

    return total_text_fea, total_audio_fea, total_utt

def collate_fn_fea_whisper_lab(batch):
    total_text_fea = []
    total_audio_fea = []
    total_whisper_fea = []
    total_lab = []
    total_utt = []
    
    for fea_data in batch:

        text_fea, audio_fea = fea_data[0]
        whisper_fea = torch.from_numpy(joblib.load(fea_data[1])).squeeze(0)
        lab = fea_data[2]
        total_text_fea.append(torch.mean(text_fea, dim=0))
        total_audio_fea.append(audio_fea)
        total_whisper_fea.append(whisper_fea)
        total_lab.append(lab)
        total_utt.append(fea_data[3])
    
    total_text_fea = nn.utils.rnn.pad_sequence(total_text_fea, batch_first=True)  # Pad to ensure equal lengths
    total_audio_fea = nn.utils.rnn.pad_sequence(total_audio_fea, batch_first=True)  # Pad to ensure equal lengths
    total_whisper_fea = nn.utils.rnn.pad_sequence(total_whisper_fea, batch_first=True)  # Pad to ensure equal lengths
    total_lab = torch.Tensor(np.array(total_lab))

    return total_text_fea, total_audio_fea, total_whisper_fea, total_lab, total_utt

def collate_fn_fea_whisper_test3(batch):
    total_text_fea = []
    total_audio_fea = []
    total_whisper_fea = []
    total_utt = []
    
    for fea_data in batch:

        text_fea, audio_fea = fea_data[0]
        whisper_fea = torch.from_numpy(joblib.load(fea_data[1])).squeeze(0)
        total_text_fea.append(torch.mean(text_fea, dim=0))
        total_audio_fea.append(audio_fea)
        total_whisper_fea.append(whisper_fea)
        total_utt.append(fea_data[2])
    
    total_text_fea = nn.utils.rnn.pad_sequence(total_text_fea, batch_first=True)  # Pad to ensure equal lengths
    total_audio_fea = nn.utils.rnn.pad_sequence(total_audio_fea, batch_first=True)  # Pad to ensure equal lengths
    total_whisper_fea = nn.utils.rnn.pad_sequence(total_whisper_fea, batch_first=True)  # Pad to ensure equal lengths

    return total_text_fea, total_audio_fea, total_whisper_fea, total_utt

def collate_fn_text_lab(batch):
    total_text = []
    total_lab = []
    total_utt = []
    for text_data in batch:

        text = text_data[0]
        lab = text_data[1]
        total_text.append(text)
        total_lab.append(lab)
        total_utt.append(text_data[2])
    
    total_lab = torch.Tensor(np.array(total_lab))

    return total_text, total_lab, total_utt

def collate_fn_wav_text_lab_mask(batch):
    total_wav = []
    total_text = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        text = wav_data[1]
        lab = wav_data[2]
        total_wav.append(torch.Tensor(wav))
        total_text.append(text)
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(wav_data[3])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_text, total_lab, attention_mask, total_utt

def collate_fn_wav_text_mask_test3(batch):
    total_wav = []
    total_text = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        text = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_text.append(text)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_text, attention_mask, total_utt