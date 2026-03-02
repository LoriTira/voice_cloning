from f5_tts.api import F5TTS
import inspect

f5tts = F5TTS(device='cpu')
model = f5tts.ema_model.transformer

print(inspect.signature(model.forward))
