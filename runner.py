import sys
class DummyIPython:
    def system(self, cmd):
        pass
    def display(self, *args, **kwargs):
        pass
import builtins
builtins.get_ipython = lambda: DummyIPython()

with open('f5_tts_voice_cloning.py') as f:
    code = f.read()
    
# IPython.display Audio is used in the notebook
import IPython.display as ipd
ipd.display = DummyIPython().display
ipd.Audio = lambda *args, **kwargs: None

exec(code)
