import torch
import os
import torchaudio
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import nbformat
with open('f5_tts_voice_cloning.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

code = ""
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Skip the pip installs and UI display parts because they crash in Python
        lines = cell.source.split('\n')
        for line in lines:
            if not line.startswith('!') and 'ipd.display' not in line:
                code += line + '\n'

class DummyIPython:
    def display(self, *args, **kwargs): pass
import IPython.display as ipd
ipd.display = DummyIPython().display
ipd.Audio = lambda *args, **kwargs: None

try:
    exec(code)
    print("Execution Success")
except Exception as e:
    print("Execution Failed:", e)
