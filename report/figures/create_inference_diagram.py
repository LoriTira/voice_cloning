"""
Simplified F5-TTS inference diagram for the HODL project report.
Clean, centered vertical flow.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

fig, ax = plt.subplots(1, 1, figsize=(5.8, 7.8))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, 13)
ax.axis('off')

# ── Colors ──
C_MODEL   = '#D6E6F0'
C_LORA    = '#F5C882'
C_ODE     = '#B5D8B5'
C_VOCODER = '#D9D9D9'
C_OUTPUT  = '#D5CAE0'
C_ARROW   = '#333333'

CX = 5.0  # center x

def box(x, y, w, h, label, color, fontsize=9.5, bold=False):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle='round,pad=0.12',
                       facecolor=color, edgecolor='#444444', linewidth=1.0)
    ax.add_patch(b)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color='#222222', zorder=5)

def varrow(y_from, y_to, x=CX):
    ax.annotate('', xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.3))

def darrow(x1, y1, x2, y2, zorder=6):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=zorder,
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.1))

# ════════════════════════════════════════
# Title
# ════════════════════════════════════════
ax.text(CX, 12.5, 'F5-TTS Inference Pipeline', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#111111')

# ════════════════════════════════════════
# Output Waveform
# ════════════════════════════════════════
Y_OUT = 11.5
box(CX, Y_OUT, 4.2, 0.6, 'Output Waveform', C_OUTPUT, fontsize=10, bold=True)

# ════════════════════════════════════════
# Vocoder
# ════════════════════════════════════════
Y_VOC = 10.4
box(CX, Y_VOC, 4.2, 0.6, 'Vocos Vocoder  (frozen)', C_VOCODER, fontsize=9.5, bold=True)
varrow(Y_VOC + 0.3, Y_OUT - 0.3)

ax.text(CX + 2.65, (Y_VOC + Y_OUT) / 2, 'discard\nref. portion', ha='center', va='center',
        fontsize=6.5, color='#777777', style='italic')

# ════════════════════════════════════════
# ODE Solver
# ════════════════════════════════════════
Y_ODE = 9.1
box(CX, Y_ODE, 4.8, 0.7, 'ODE Solver  (32 steps)\nCFG = 2.0,   sway = $-$1.0',
    C_ODE, fontsize=9, bold=True)
varrow(Y_ODE + 0.35, Y_VOC - 0.3)

# ════════════════════════════════════════
# NFE loop arrow (left side, ODE ↔ DiT)
# ════════════════════════════════════════
nfe_x = CX - 2.7
ax.annotate('',
            xy=(nfe_x, 6.8),
            xytext=(nfe_x, 8.7),
            arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0,
                            connectionstyle='arc3,rad=0.45'))
ax.text(nfe_x - 0.7, 7.75, 'NFE\nloop', ha='center', va='center',
        fontsize=6.5, color='#999999', style='italic')

# ════════════════════════════════════════
# DiT Backbone + LoRA
# ════════════════════════════════════════
Y_DIT = 6.6
DIT_H = 2.6
DIT_W = 5.6

dit = FancyBboxPatch((CX - DIT_W/2, Y_DIT - DIT_H/2), DIT_W, DIT_H,
                      boxstyle='round,pad=0.15',
                      facecolor=C_MODEL, edgecolor='#444444', linewidth=1.3)
ax.add_patch(dit)

ax.text(CX, Y_DIT + 0.8, 'F5-TTS DiT', ha='center', va='center',
        fontsize=12, fontweight='bold', color='#222222')
ax.text(CX, Y_DIT + 0.25, '339.6M params,  $N$ transformer blocks',
        ha='center', va='center', fontsize=8, color='#555555')

# LoRA sub-box
lora_y = Y_DIT - 0.5
box(CX, lora_y, 4.4, 0.55, 'LoRA Adapters  (r = 8,  2.5M params,  0.74%)',
    C_LORA, fontsize=7.5)

varrow(Y_DIT + DIT_H/2, Y_ODE - 0.35)

# Timestep (right side input) — arrow points INTO DiT
ts_x = CX + 4.1
ts_y = Y_DIT + 0.25
box(ts_x, ts_y, 1.5, 0.45, 'Timestep $t$', '#E8E0F0', fontsize=8)
# Arrow from timestep box left edge → just inside DiT right edge
darrow(ts_x - 0.75 - 0.12, ts_y, CX + DIT_W/2 - 0.05, ts_y, zorder=10)

# ════════════════════════════════════════
# Inputs row
# ════════════════════════════════════════
Y_IN = 3.3
IN_W = 3.0
IN_H = 0.85

# Reference Audio (left-center)
ref_x = CX - 2.3
box(ref_x, Y_IN, IN_W, IN_H, 'Reference Audio\n(0 – 8 s anchor)', '#E8D5B7', fontsize=9, bold=True)

# Text (right-center)
txt_x = CX + 2.3
box(txt_x, Y_IN, IN_W, IN_H, 'Text\n(ref transcript + gen text)', '#E0E8C0', fontsize=9, bold=True)

# Noise (center, between inputs and DiT)
noise_y = 4.6
box(CX, noise_y, 2.6, 0.5, 'Noise  $x_0 \\sim \\mathcal{N}(0,\\,1)$',
    '#F0E0E0', fontsize=8.5)

# Arrows: inputs and noise → DiT bottom
dit_bottom = Y_DIT - DIT_H/2
darrow(ref_x + 0.3, Y_IN + IN_H/2, CX - 1.0, dit_bottom)
darrow(txt_x - 0.3, Y_IN + IN_H/2, CX + 1.0, dit_bottom)
varrow(noise_y + 0.25, dit_bottom)

plt.tight_layout()
plt.savefig('/Users/lorenzo/Desktop/MIT/code/deep_learning/HODL_project_colab/report/inference_diagram_simplified.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Done.")
