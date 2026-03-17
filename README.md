# Few-Shot Voice Cloning with F5-TTS

**When does fine-tuning add value over reference audio conditioning in voice cloning?**

This project investigates whether LoRA fine-tuning improves voice cloning quality beyond what in-context anchor audio conditioning already provides in [F5-TTS](https://arxiv.org/abs/2410.06885). The motivation is preserving the voices of people with neurodegenerative diseases (ALS, Parkinson's) who are gradually losing their ability to speak.

> Course project for MIT 15.773 — Hands On Deep Learning (Spring 2025)
>
> Authors: Arthur Schoen, Max Soderlind, Lorenzo Tiraboschi

## Key Findings

We ran a **6 x 5 grid experiment** (anchor duration x fine-tuning amount), generating **675 audio samples** across 45 conditions:

- **Reference audio dominates**: Just 1.5 seconds of anchor audio improves speaker similarity by **+0.34** (on a 0–1 scale)
- **Fine-tuning adds negligible benefit**: 20 sentences of LoRA training adds at most **+0.025** — even with 16x larger adapters
- **Sweet spot**: 3–8 seconds of reference audio gives the best balance of voice similarity and speech intelligibility

## Architecture

| Component | Details |
|-----------|---------|
| **Base model** | F5-TTS DiT (339.6M params) — conditional flow matching |
| **Adapter** | PEFT LoRA (r=8, alpha=16) on attention + FFN — 2.5M trainable params (0.74%) |
| **Vocoder** | Vocos (frozen) — mel-to-waveform |
| **Training** | AdamW, cosine LR with warmup, gradient accumulation |
| **Inference** | 32-step ODE solver, CFG=2.0, sway=-1.0 |
| **Evaluation** | Resemblyzer cosine similarity (speaker embeddings) + Whisper WER |

## Experiment Design

```
Anchor durations:  [0, 0.5, 1, 2, 4, 8] seconds
Training amounts:  [0, 1, 5, 10, 20] sentences
Metric:            Speaker similarity (Resemblyzer cosine distance), averaged over 5 test sentences
```

## Repository Structure

```
.
├── README.md
├── notebook/
│   └── f5_tts_voice_cloning.ipynb   # Full experiment (runs on Google Colab)
└── report/
    ├── HODL_Project_Report.pdf       # Project report
    └── figures/                      # Diagrams and plots
```

## Getting Started

### Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKn-H-WektFrs0IUDLnMPadb-bTfwd9H?usp=sharing)

The notebook is designed to run end-to-end on Google Colab with a GPU runtime. It handles all dependency installation automatically.

### Running Locally

1. Clone this repository
2. Open `notebook/f5_tts_voice_cloning.ipynb` in Jupyter
3. Provide your own audio file (M4A format, ~20 sentences) in the config cell
4. Run cells top-to-bottom: setup → data → model → training grid → evaluation grid → results

**Requirements**: Python 3.10+, CUDA GPU (or Apple Silicon with MPS fallback), ~8GB VRAM

## Data

Training audio is not included in this repository due to size. The notebook expects an M4A file containing recordings of Harvard sentences. You can record your own or use any clear speech recording.

## License

This project is for educational and research purposes. F5-TTS is subject to its own [license](https://github.com/SWivid/F5-TTS).
