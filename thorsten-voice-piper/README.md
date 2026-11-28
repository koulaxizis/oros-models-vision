---
license: mit
datasets:
- Thorsten-Voice/TV-44kHz-Full
language:
- de
tags:
- tts
- text-to-speech
---
**Guude 👋!**

Models and config for Thorsten-Voice model trained on Piper TTS (part of Rhasspy project).
Thanks to @synesthesiam for providing cool open source voice technology and training these (and many more models).

Please check out other available voices for Piper TTS here:
https://github.com/rhasspy/piper
or here: https://huggingface.co/rhasspy/piper-voices

# Simple usage
> echo "Willkommen in der Welt von künstlicher Sprachausgabe." |  ./piper -m de_DE-thorsten-high.onnx -c de_DE-thorsten-high.onnx.json -d /home/thorsten/ -f output.wav