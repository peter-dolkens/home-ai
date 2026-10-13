"""Local speech-to-text engines for the round trip, all CPU, all things a Home Assistant install can run today.

  faster-whisper  tiny base small medium large-v3-turbo distil-large-v3   what HA's Wyoming "whisper" add-on runs when
                  a model is picked; int8 on CPU, language en, beam 5 (the add-on's beam_size 0 = 5 off ARM, 1 on ARM),
                  no VAD filter, segments joined with spaces; the only engines here that take an initial_prompt
  parakeet        NVIDIA Parakeet TDT 0.6B v2 int8 via sherpa-onnx: the add-on's default for English ("auto")
  moonshine-tiny  Useful Sensors Moonshine via sherpa-onnx (small, fast, English only)
  moonshine-base
  kroko           Kroko streaming zipformer via sherpa-onnx: the add-on's "sherpa_streaming" (fast) option

Model files: faster-whisper downloads through the Hugging Face cache (HF_HOME); sherpa models are the release tarballs
unpacked under <cache>/sherpa/<name> (roundtrip.py --fetch downloads them).
"""
from __future__ import annotations

import os
import time

WHISPER = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium", "large-v3-turbo": "large-v3-turbo",
           "distil-large-v3": "distil-large-v3"}
SHERPA = {"parakeet": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
          "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-int8",
          "moonshine-base": "sherpa-onnx-moonshine-base-en-int8",
          "kroko": "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"}
ENGINES = tuple(WHISPER) + tuple(SHERPA)
SHERPA_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{}.tar.bz2"
# resident memory per worker process, GB (int8, CPU), for sizing workers to the container
MEMORY_GB = {"tiny": 0.4, "base": 0.5, "small": 0.9, "medium": 1.6, "large-v3-turbo": 1.9, "distil-large-v3": 1.8,
             "parakeet": 1.4, "moonshine-tiny": 0.4, "moonshine-base": 0.7, "kroko": 0.5}


def fetch_sherpa(name: str, cache: str) -> str:
    import tarfile
    import urllib.request
    d = os.path.join(cache, "sherpa")
    path = os.path.join(d, SHERPA[name])
    if not os.path.isdir(path):
        # each pool worker loads its own engine: download and unpack under a per-process name, then rename into place,
        # so concurrent first loads cannot leave a half-written model (3 Oct: a truncated parakeet encoder.int8.onnx)
        import shutil
        os.makedirs(d, exist_ok=True)
        tmp, stage = f"{path}.{os.getpid()}.tar.bz2", f"{path}.{os.getpid()}.d"
        urllib.request.urlretrieve(SHERPA_URL.format(SHERPA[name]), tmp)
        with tarfile.open(tmp) as t:
            t.extractall(stage)
        os.remove(tmp)
        try:
            os.rename(os.path.join(stage, SHERPA[name]), path)
        except OSError:  # another worker finished first
            pass
        shutil.rmtree(stage, ignore_errors=True)
    return path


def _pick(d: str, *stems: str) -> str:
    for s in stems:
        for f in (f"{s}.int8.onnx", f"{s}.onnx"):
            if os.path.exists(os.path.join(d, f)):
                return os.path.join(d, f)
    raise FileNotFoundError(f"none of {stems} in {d}")


class Engine:
    """transcribe(int16 or float 16 kHz audio, prompt) -> text. prompts: whether an initial_prompt does anything."""

    prompts = False

    def transcribe(self, audio, prompt: str | None = None) -> str:
        raise NotImplementedError


class Whisper(Engine):
    prompts = True

    def __init__(self, size: str, threads: int, beam_size: int = 5):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(WHISPER[size], device="cpu", compute_type="int8", cpu_threads=threads)
        self.beam_size = beam_size

    def transcribe(self, audio, prompt=None):
        import numpy as np
        x = audio.astype(np.float32) / (32768.0 if audio.dtype == np.int16 else 1.0)
        segs, _ = self.model.transcribe(x, language="en", beam_size=self.beam_size, initial_prompt=prompt or None,
                                        vad_filter=False)
        return " ".join(s.text.strip() for s in segs).strip()


class Sherpa(Engine):
    def __init__(self, name: str, threads: int, cache: str):
        import sherpa_onnx
        d = fetch_sherpa(name, cache)
        tokens = os.path.join(d, "tokens.txt")
        self.online = name == "kroko"
        if name == "parakeet":
            self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=_pick(d, "encoder"), decoder=_pick(d, "decoder"), joiner=_pick(d, "joiner"), tokens=tokens,
                num_threads=threads, model_type="nemo_transducer")
        elif name.startswith("moonshine"):
            self.rec = sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=_pick(d, "preprocess"), encoder=_pick(d, "encode"), uncached_decoder=_pick(d, "uncached_decode"),
                cached_decoder=_pick(d, "cached_decode"), tokens=tokens, num_threads=threads)
        else:
            self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens, encoder=_pick(d, "encoder"), decoder=_pick(d, "decoder"), joiner=_pick(d, "joiner"),
                num_threads=threads)

    def transcribe(self, audio, prompt=None):
        import numpy as np
        x = audio.astype(np.float32) / (32768.0 if audio.dtype == np.int16 else 1.0)
        s = self.rec.create_stream()
        s.accept_waveform(16000, x)
        if self.online:
            s.accept_waveform(16000, np.zeros(24000, np.float32))  # flush the right context
            s.input_finished()
            while self.rec.is_ready(s):
                self.rec.decode_stream(s)
            return self.rec.get_result(s).strip()
        self.rec.decode_stream(s)
        return s.result.text.strip()


def load(name: str, threads: int, cache: str, beam_size: int = 5) -> Engine:
    if name in WHISPER:
        return Whisper(name, threads, beam_size)
    if name in SHERPA:
        return Sherpa(name, threads, cache)
    raise ValueError(f"unknown engine {name!r} (have {', '.join(ENGINES)})")


def whisper_token_counter():
    """Token count with Whisper's own tokenizer (the same for every size here), for the biasing-prompt budget."""
    import tokenizers
    from faster_whisper.utils import download_model
    tok = tokenizers.Tokenizer.from_file(os.path.join(download_model("tiny"), "tokenizer.json"))
    return lambda s: len(tok.encode(" " + s.strip(), add_special_tokens=False).ids)


def timed(engine: Engine, audio, prompt=None) -> tuple[str, float]:
    t = time.perf_counter()
    text = engine.transcribe(audio, prompt)
    return text, time.perf_counter() - t
