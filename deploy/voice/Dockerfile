# Voice-robustness round trip (homeai/voice): Piper TTS -> acoustic conditions -> local speech-to-text engines, CPU only.
#   docker build -t homeai-voice deploy/voice
#   scripts/voice-roundtrip.sh --sets probe --stt base --conditions clean     # repo at /work, models + audio cache at /cache
# Engines: faster-whisper (what HA's Wyoming whisper add-on runs), sherpa-onnx (the add-on's Parakeet / Moonshine / Kroko
# backend). The code is mounted, not copied, so edits need no rebuild.
FROM python:3.12-slim
RUN pip install --no-cache-dir \
      piper-tts==1.8.0 faster-whisper==1.2.1 ctranslate2==4.8.2 sherpa-onnx==1.13.8 onnxruntime==1.30.0 \
      numpy==2.5.3 soundfile==0.14.0
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/work HF_HOME=/cache/hf HOMEAI_VOICE_CACHE=/cache HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /work
ENTRYPOINT ["python", "-m", "homeai.voice.roundtrip"]
