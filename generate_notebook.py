import nbformat
import os

def create_notebook():
    nb = nbformat.v4.new_notebook()

    intro_md = """# F5-TTS Voice Cloning Pipeline (Local MPS M4 Pro Execution)
    
This notebook implements an end-to-end few-shot voice cloning pipeline using the **F5-TTS (Flow Matching TTS)** architecture, explicitly optimized for Apple Silicon via the PyTorch `mps` (Metal Performance Shaders) backend.

**Hardware**: M4 Pro with 48GB unified memory.  
**Data Constraints**: 24000 Hz, mono channel resampling.  
**Architecture**: Uses the official F5-TTS `CFM.sample()` ODE solver with Classifier-Free Guidance (CFG), Empirically Pruned Step Sampling (EPSS), and sway sampling for maximum voice likeness.

**Training Data**: ~5 minutes of Harvard sentences audio, automatically split into ~80 sentence-level segments for LoRA fine-tuning.
"""
    nb.cells.append(nbformat.v4.new_markdown_cell(intro_md))

    install_code = """# Cell 1: Environment Setup and Installs
!pip install -q torch torchaudio transformers peft vocos tqdm IPython numpy librosa soundfile f5-tts
"""
    nb.cells.append(nbformat.v4.new_code_cell(install_code))

    imports_code = """# Cell 2: Imports and MPS Device Config
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

# Hard constraint: Use PyTorch `mps` backend for Apple Silicon and avoid `torch.float16` to prevent instability
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEVICE = torch.device('mps')
DTYPE = torch.float32 # MUST be float32, bfloat16 will crash M4 Pro during `torch._C._nn.linear` execution

print(f"Device set to: {DEVICE}")
print(f"Default datatype: {DTYPE}")
"""
    nb.cells.append(nbformat.v4.new_code_cell(imports_code))

    data_code = """# Cell 3: Phase 1 - Data Pipeline and Preprocessing
#
# Since our audio is a single long recording (~5 min), we split it into individual
# sentence segments using silence detection, then pair each with its transcript line.

import librosa
import soundfile as sf

def split_audio_by_silence(audio_path, sr=24000, top_db=35, min_gap=0.4):
    "Split a long audio file into sentence-level segments using silence detection."
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Find non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    
    # Merge segments closer than min_gap seconds (within-sentence pauses)
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
    
    # Extract audio segments
    segments = []
    for start, end in merged:
        segment = y[start:end]
        segments.append(torch.from_numpy(segment))
    
    print(f"Split audio into {len(segments)} segments")
    durations = [(end-start)/sr for start, end in merged]
    print(f"Segment durations: {min(durations):.1f}s - {max(durations):.1f}s (avg {np.mean(durations):.1f}s)")
    
    return segments, y

class SplitAudioDataset(Dataset):
    "Dataset that pairs silence-split audio segments with transcript lines."
    def __init__(self, audio_path, text_file, sr=24000):
        self.sr = sr
        
        # Load transcript lines
        texts = []
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        
        # Split audio into segments
        segments, self.full_audio = split_audio_by_silence(audio_path, sr=sr)
        
        # Align: use min(segments, texts) to handle slight mismatch
        n = min(len(segments), len(texts))
        if len(segments) != len(texts):
            print(f"Note: {len(segments)} audio segments vs {len(texts)} text lines. Using first {n}.")
        self.segments = segments[:n]
        self.texts = texts[:n]
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        return {"audio": self.segments[idx], "text": self.texts[idx]}

# Build the dataset from the full 5-minute recording
audio_path = "./data/harvard_sentences.m4a"
text_path = "./data/harvard_sent_text.txt"
dataset = SplitAudioDataset(audio_path, text_path)
print(f"\\nDataset ready: {len(dataset)} sentence-audio pairs for training")
"""
    nb.cells.append(nbformat.v4.new_code_cell(data_code))

    model_code = """# Cell 4: Phase 2 - Model Initialization & LoRA Adaptation

from f5_tts.api import F5TTS
from f5_tts.model.utils import convert_char_to_pinyin

print("Initializing pretrained F5-TTS DiT Backbone (Downloads weights on first run)...")
f5tts = F5TTS(device=str(DEVICE))

# The full CFM (Conditional Flow Matching) model contains: transformer, mel_spec, vocab, and the ODE solver.
# We access it via f5tts.ema_model - this is the official inference-ready model object.
ema_model = f5tts.ema_model

# CRITICAL Apple Silicon Fix: Explicitly cast the entire imported Safetensor weights to Float32 to prevent MPS crashes
ema_model.to(torch.float32)

# We also extract F5-TTS's internal MelSpec module to convert our raw audio into proper mel sequences
mel_spec_module = ema_model.mel_spec

# Vocab tools
vocab_char_map = ema_model.vocab_char_map
from f5_tts.model.utils import list_str_to_idx

# Configure PEFT / LoRA for speaker adaptation fine-tuning
# r=8 with 80 training sentences prevents overfitting while allowing speaker adaptation
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.0.0", "ff.2"],
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA to the transformer backbone
lora_model = get_peft_model(ema_model.transformer, lora_config)
lora_model.print_trainable_parameters()

# Vocos Vocoder (Frozen, forced to CPU for MPS compatibility)
vocoder = f5tts.vocoder
vocoder.to(torch.device('cpu'))
for param in vocoder.parameters():
    param.requires_grad = False
print("Vocoder fully frozen and loaded via f5_tts (Forced to CPU).")
"""
    nb.cells.append(nbformat.v4.new_code_cell(model_code))

    training_code = """# Cell 5: Phase 3 - LoRA Fine-Tuning on Speaker Data
#
# Fine-tune the LoRA adapters on all ~80 Harvard sentence segments.
# The official CFM.forward() handles: mel conversion, random span masking,
# CFG dropout, optimal transport path, and loss computation.

def train_flow_matching(ema_model, dataset, epochs=5, lr=7e-5, save_dir="./f5_tts_lora_adapters"):
    "Fine-tune LoRA adapters using the official CFM.forward() training path."
    trainable_params = [p for p in ema_model.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in trainable_params):,} LoRA parameters")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    ema_model.train()
    
    global_step = 0
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        epoch_losses = []
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        print(f"\\n--- Epoch {epoch+1}/{epochs} ---")
        for i in tqdm(indices, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            sample = dataset[i]
            # Raw waveform as (1, T) - CFM.forward() handles mel conversion internally
            waveform = sample['audio'].unsqueeze(0).to(DEVICE).to(DTYPE)
            
            # Text preprocessing: convert to pinyin for proper tokenization
            text_list = convert_char_to_pinyin([sample['text']])
            
            # Official CFM.forward() computes the full flow matching loss:
            # 1. waveform -> mel spectrogram
            # 2. Random span mask (70-100% masked for generation prediction)
            # 3. CFG dropout (audio_drop_prob=0.3, cond_drop_prob=0.2)
            # 4. Optimal transport: phi = (1-t)*noise + t*mel
            # 5. MSE loss on masked (predicted) region only
            loss, cond, pred = ema_model(
                inp=waveform,
                text=text_list,
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if global_step % 20 == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f}")
                
            global_step += 1
        
        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
            
    # Save LoRA adapters
    print(f"\\nSaving LoRA adapters to {save_dir}")
    lora_model.save_pretrained(save_dir)
    print("Training complete!")

# Train on the full speaker dataset
train_flow_matching(ema_model, dataset, epochs=5, lr=7e-5)
"""
    nb.cells.append(nbformat.v4.new_code_cell(training_code))

    inference_code = """# Cell 6: Phase 4 - Inference Using Official F5-TTS CFM.sample() ODE Solver
#
# Uses the fine-tuned model with the official inference pipeline:
#   - Classifier-Free Guidance (CFG=2.0)
#   - 32-step ODE with Empirically Pruned Step Sampling (EPSS)
#   - Sway sampling (coef=-1)
#   - 10-second reference audio for maximum voice priming

import librosa
import soundfile as sf

@torch.no_grad()
def synthesize_speech(ema_model, ref_text, gen_text, anchor_audio_path,
                      nfe_steps=32, cfg_strength=2.0, sway_coef=-1.0, speed=1.0):
    "Synthesize speech using the official F5-TTS CFM.sample() ODE solver."
    ema_model.eval()
    
    # 1. Load and preprocess anchor audio
    waveform_np, _ = librosa.load(anchor_audio_path, sr=24000, mono=True)
    waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # (1, T)
    
    # RMS normalization (matches official pipeline target_rms=0.1)
    target_rms = 0.1
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    if rms < target_rms:
        waveform = waveform * target_rms / rms
    
    cond_audio = waveform.to(DEVICE).to(DTYPE)
    
    # 2. Ensure ref_text ends with proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("."):
        ref_text = ref_text + ". "
    elif ref_text.endswith("."):
        ref_text = ref_text + " "
    
    # 3. Text preprocessing: ref_text + gen_text, then convert to pinyin
    text_str = ref_text + gen_text
    text_list = convert_char_to_pinyin([text_str])
    
    # 4. Compute target duration (official formula)
    hop_length = 256
    ref_audio_len = cond_audio.shape[-1] // hop_length
    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    local_speed = speed
    if gen_text_len < 10:
        local_speed = 0.3
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
    
    print(f"Reference audio: {ref_audio_len} mel frames ({ref_audio_len * hop_length / 24000:.1f}s)")
    print(f"Target duration: {duration} mel frames ({duration * hop_length / 24000:.1f}s)")
    print(f"Using {nfe_steps}-step ODE with CFG={cfg_strength}, sway={sway_coef}")
    
    # 5. Run official CFM.sample()
    generated, trajectory = ema_model.sample(
        cond=cond_audio,
        text=text_list,
        duration=duration,
        steps=nfe_steps,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_coef,
    )
    
    # 6. Extract only the newly generated portion
    generated = generated.to(torch.float32)
    generated = generated[:, ref_audio_len:, :]
    generated_mel = generated.permute(0, 2, 1)  # -> (1, D, T)
    
    # 7. Decode using frozen Vocos vocoder
    mel_cpu = generated_mel.to(torch.device('cpu'), dtype=torch.float32)
    out_waveform = vocoder.decode(mel_cpu)
    if rms < target_rms:
        out_waveform = out_waveform * rms / target_rms
    out_waveform = out_waveform.squeeze().cpu().numpy()
    
    return out_waveform, 24000

# ===== Inference =====
ref_audio_path = "./data/harvard_sentences.m4a"  
text_file_path = "./data/harvard_sent_text.txt"

# Use the first 3 sentences as reference text (~10 seconds of audio)
with open(text_file_path, 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
ref_text = " ".join(lines[:3])  # First 3 Harvard sentences
print(f"Reference text: {ref_text}")

# Extract the first ~10 seconds of the recording as voice reference
temp_anchor = "./data/temp_anchor_slice.wav"
y, sr_load = librosa.load(ref_audio_path, sr=24000, mono=True, duration=10.0) 
sf.write(temp_anchor, y, sr_load)
print(f"Reference audio slice: {len(y)/24000:.1f}s")

# Generate speech in the cloned voice
audio_numpy, sr_out = synthesize_speech(
    ema_model=ema_model, 
    ref_text=ref_text,
    gen_text="This is the generated continuation of the patient's voice, synthesized using flow matching.", 
    anchor_audio_path=temp_anchor,
    nfe_steps=32,
    cfg_strength=2.0,
    sway_coef=-1.0,
)

# Play the output audio
print("\\nGeneration Complete! Play Output:")
ipd.display(ipd.Audio(audio_numpy, rate=sr_out))

# Save to file
sf.write("./data/generated_output.wav", audio_numpy, sr_out)
print("Saved to ./data/generated_output.wav")

# Also play the reference for A/B comparison
print("\\nReference audio for comparison:")
ipd.display(ipd.Audio(y, rate=24000))
"""
    nb.cells.append(nbformat.v4.new_code_cell(inference_code))

    with open('f5_tts_voice_cloning.ipynb', 'w') as f:
        nbformat.write(nb, f)
    
    print("Notebook 'f5_tts_voice_cloning.ipynb' generated successfully.")

if __name__ == '__main__':
    create_notebook()
