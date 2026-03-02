#!/usr/bin/env python
# coding: utf-8

# # F5-TTS Voice Cloning Pipeline (Local MPS M4 Pro Execution)
#
# Uses official F5-TTS CFM.sample() with CFG, EPSS, sway sampling.
# Trains LoRA on full 5-minute dataset split into sentences, then infers with 10s reference.

# Cell 2: Imports and MPS Device Config
import os
import io
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from peft import LoraConfig, get_peft_model
from vocos import Vocos
import IPython.display as ipd
from tqdm.auto import tqdm

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = torch.device('mps')
DTYPE = torch.float32

print(f"Device set to: {DEVICE}")
print(f"Default datatype: {DTYPE}")


# Cell 3: Phase 1 - Data Pipeline

import librosa
import soundfile as sf

def split_audio_by_silence(audio_path, sr=24000, top_db=35, min_gap=0.4):
    "Split a long audio file into sentence-level segments using silence detection."
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    intervals = librosa.effects.split(y, top_db=top_db)
    
    merged = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        gap = (start - current_end) / sr
        if gap < min_gap:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    
    segments = [torch.from_numpy(y[s:e]) for s, e in merged]
    print(f"Split audio into {len(segments)} segments")
    return segments, y

class SplitAudioDataset(Dataset):
    def __init__(self, audio_path, text_file, sr=24000):
        self.sr = sr
        texts = []
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        
        segments, self.full_audio = split_audio_by_silence(audio_path, sr=sr)
        n = min(len(segments), len(texts))
        self.segments = segments[:n]
        self.texts = texts[:n]
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return {"audio": self.segments[idx], "text": self.texts[idx]}

audio_path = "./data/harvard_sentences.m4a"
text_path = "./data/harvard_sent_text.txt"
dataset = SplitAudioDataset(audio_path, text_path)
print(f"Dataset ready: {len(dataset)} sentence-audio pairs")


# Cell 4: Phase 2 - Model Initialization & LoRA

from f5_tts.api import F5TTS
from f5_tts.model.utils import convert_char_to_pinyin

print("Initializing pretrained F5-TTS DiT Backbone...")
f5tts = F5TTS(device=str(DEVICE))

ema_model = f5tts.ema_model
ema_model.to(torch.float32)

mel_spec_module = ema_model.mel_spec
vocab_char_map = ema_model.vocab_char_map
from f5_tts.model.utils import list_str_to_idx

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.0.0", "ff.2"],
    lora_dropout=0.05, bias="none"
)

lora_model = get_peft_model(ema_model.transformer, lora_config)
lora_model.print_trainable_parameters()

vocoder = f5tts.vocoder
vocoder.to(torch.device('cpu'))
for param in vocoder.parameters():
    param.requires_grad = False
print("Vocoder frozen (CPU).")


# Cell 5: Phase 3 - LoRA Fine-Tuning

def train_flow_matching(ema_model, dataset, epochs=5, lr=7e-5, save_dir="./f5_tts_lora_adapters"):
    trainable_params = [p for p in ema_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    ema_model.train()
    
    global_step = 0
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        epoch_losses = []
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        for i in tqdm(indices, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            sample = dataset[i]
            waveform = sample['audio'].unsqueeze(0).to(DEVICE).to(DTYPE)
            text_list = convert_char_to_pinyin([sample['text']])
            
            loss, cond, pred = ema_model(inp=waveform, text=text_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            if global_step % 20 == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f}")
            global_step += 1
        
        print(f"  Epoch {epoch+1} avg loss: {np.mean(epoch_losses):.4f}")
    
    print(f"\nSaving LoRA adapters to {save_dir}")
    lora_model.save_pretrained(save_dir)

train_flow_matching(ema_model, dataset, epochs=5, lr=7e-5)


# Cell 6: Phase 4 - Inference

@torch.no_grad()
def synthesize_speech(ema_model, ref_text, gen_text, anchor_audio_path,
                      nfe_steps=32, cfg_strength=2.0, sway_coef=-1.0, speed=1.0):
    ema_model.eval()
    
    waveform_np, _ = librosa.load(anchor_audio_path, sr=24000, mono=True)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)
    
    target_rms = 0.1
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    if rms < target_rms:
        waveform = waveform * target_rms / rms
    
    cond_audio = waveform.to(DEVICE).to(DTYPE)
    
    if not ref_text.endswith(". ") and not ref_text.endswith("."):
        ref_text = ref_text + ". "
    elif ref_text.endswith("."):
        ref_text = ref_text + " "
    
    text_str = ref_text + gen_text
    text_list = convert_char_to_pinyin([text_str])
    
    hop_length = 256
    ref_audio_len = cond_audio.shape[-1] // hop_length
    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    local_speed = speed
    if gen_text_len < 10:
        local_speed = 0.3
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
    
    print(f"Generating: {nfe_steps} steps, CFG={cfg_strength}, sway={sway_coef}")
    
    generated, trajectory = ema_model.sample(
        cond=cond_audio, text=text_list, duration=duration,
        steps=nfe_steps, cfg_strength=cfg_strength, sway_sampling_coef=sway_coef,
    )
    
    generated = generated.to(torch.float32)[:, ref_audio_len:, :]
    generated_mel = generated.permute(0, 2, 1)
    
    mel_cpu = generated_mel.to(torch.device('cpu'), dtype=torch.float32)
    out_waveform = vocoder.decode(mel_cpu)
    if rms < target_rms:
        out_waveform = out_waveform * rms / target_rms
    out_waveform = out_waveform.squeeze().cpu().numpy()
    
    return out_waveform, 24000

# Execution
ref_audio_path = "./data/harvard_sentences.m4a"
text_file_path = "./data/harvard_sent_text.txt"

with open(text_file_path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
ref_text = " ".join(lines[:3])

temp_anchor = "./data/temp_anchor_slice.wav"
y, sr_load = librosa.load(ref_audio_path, sr=24000, mono=True, duration=10.0) 
sf.write(temp_anchor, y, sr_load)
print(f"Reference audio: {len(y)/24000:.1f}s")

audio_numpy, sr_out = synthesize_speech(
    ema_model=ema_model, ref_text=ref_text,
    gen_text="This is the generated continuation of the patient's voice, synthesized using flow matching.", 
    anchor_audio_path=temp_anchor,
)

print("Generation Complete!")
ipd.display(ipd.Audio(audio_numpy, rate=sr_out))
sf.write("./data/generated_output.wav", audio_numpy, sr_out)
