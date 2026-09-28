"""Acoustic conditions for the TTS -> STT round trip: stand-ins for where the satellite sits and what mic it has.

Every condition takes close-talk speech (16 kHz float, Piper output resampled) and returns what the satellite would
send to speech-to-text: 16-bit, 16 kHz, with the lead-in and trailing noise a wake-word + VAD pipeline captures.
All randomness comes from a seed the caller derives from (example id, turn, condition), so a rerun rebuilds exactly
the same audio.

  clean      close-talk, quiet room (noise floor ~ -75 dBFS)
  farfield   speaker ~3 m from the mic: -20 dB distance attenuation, room reverb RT60 0.3-0.6 s with the direct path
             ~11 dB under the reverberant field (critical distance ~0.8 m), ambient floor, then satellite AGC
  noisy10    ~1 m in a mildly reverberant room with a TV (dialogue + music bed) or kitchen noise (rangehood, running
  noisy5     tap, dish clatter, fridge hum) at 10 / 5 dB SNR; the noise runs through the lead-in and tail too
  lowmic     cheap mic + narrowband path: 300-3400 Hz, soft clipping, 12-bit resolution, self-noise at 30 dB SNR
"""
from __future__ import annotations

import hashlib

import numpy as np

SR = 16000
VERSION = "ac2"  # bump when any condition's audio changes, so cached transcripts are not reused
CONDITIONS = ("clean", "farfield", "noisy10", "noisy5", "lowmic")
SPEECH_DBFS = -23.0
LEAD_S, TAIL_S = 0.3, 0.6


def seed_of(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Band-limited resampling by FFT (whole utterances are short)."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    k = n_out // 2 + 1
    Y = np.zeros(k, dtype=complex)
    m = min(k, len(X))
    Y[:m] = X[:m]
    return (np.fft.irfft(Y, n_out) * (n_out / len(x))).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def db(x: float) -> float:
    return 10 ** (x / 20)


def set_level(x: np.ndarray, dbfs: float) -> np.ndarray:
    return x * (db(dbfs) / rms(x))


def shaped_noise(rng, n: int, slope_db_oct: float = 0.0, lo: float = 20.0, hi: float = 8000.0) -> np.ndarray:
    """Gaussian noise with a spectral slope (0 white, -3 pink, -6 brown) inside [lo, hi] Hz, unit RMS."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    g = np.where((f >= lo) & (f <= hi), (np.maximum(f, lo) / lo) ** (slope_db_oct / 6.02), 0.0)
    y = np.fft.irfft(X * g, n)
    return y / rms(y)


def band(x: np.ndarray, lo: float, hi: float, order: float = 4.0) -> np.ndarray:
    """Smooth band-pass by FFT magnitude (Butterworth-like roll-off of the given order)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-9
    g = 1 / np.sqrt(1 + (lo / f) ** (2 * order)) / np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(X * g, len(x)).astype(np.float32)


def room_ir(rng, rt60: float, drr_db: float) -> np.ndarray:
    """Direct path + sparse early reflections + exponentially decaying diffuse tail, direct-to-reverberant ratio
    drr_db. Higher frequencies decay faster (air and soft furnishings)."""
    n = int(SR * rt60 * 1.2)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.004 * SR)] = 0
    for _ in range(8):  # early reflections 3-30 ms
        k = int(rng.uniform(0.003, 0.03) * SR)
        tail[k] += rng.uniform(0.3, 0.8) * rng.choice([-1, 1]) * np.sqrt(np.sum(tail ** 2) / 40)
    lowpassed = band(tail.astype(np.float32), 60, 5000, 1.0)
    tail = 0.6 * lowpassed + 0.4 * tail
    ir = tail * (db(-drr_db) / np.sqrt(np.sum(tail ** 2) + 1e-20))
    ir[0] += 1.0
    return ir.astype(np.float32)


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(h, N), N)[: len(x)].astype(np.float32)


# ---------------------------------------------------------------------- noise beds
def tv_bed(dialogue: list[np.ndarray], seconds: float = 60.0, seed: int = 7) -> np.ndarray:
    """TV across the room: dialogue lines (Piper voices, supplied by the caller) over a synth music bed with a
    beat, through a TV speaker's 120 Hz-8 kHz band and a living-room reverb."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    speech = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        line = dialogue[int(rng.integers(len(dialogue)))]
        line = set_level(line, -20 + rng.uniform(-4, 3))
        end = min(n, pos + len(line))
        speech[pos:end] += line[: end - pos]
        pos = end + int(rng.uniform(0.05, 0.6) * SR)
    t = np.arange(n) / SR
    music = np.zeros(n)
    beat = 60 / rng.uniform(90, 128)
    chords = [[220, 277.2, 329.6], [196, 246.9, 293.7], [174.6, 220, 261.6], [164.8, 207.7, 246.9]]
    for i, t0 in enumerate(np.arange(0, seconds, 4 * beat)):
        a, b = int(t0 * SR), min(n, int((t0 + 4 * beat) * SR))
        tt = t[a:b] - t0
        for f0 in chords[i % 4]:
            for h in (1, 2, 3):
                music[a:b] += np.sin(2 * np.pi * f0 * h * tt) / h ** 1.5 * (0.6 + 0.4 * np.exp(-tt * 2))
    for t0 in np.arange(0, seconds, beat / 2):  # hats + kick
        a = int(t0 * SR)
        k = min(n - a, int(0.05 * SR))
        if k > 0:
            music[a:a + k] += 0.3 * rng.standard_normal(k) * np.exp(-np.arange(k) / (0.01 * SR))
            music[a:a + k] += 0.8 * np.sin(2 * np.pi * 55 * np.arange(k) / SR) * np.exp(-np.arange(k) / (0.02 * SR))
    music = set_level(music.astype(np.float32), -30)
    mix = band(speech + music, 120, 8000, 2.0)
    return convolve(mix, room_ir(rng, 0.5, -3.0))


def kitchen_bed(seconds: float = 60.0, seed: int = 11) -> np.ndarray:
    """Kitchen: rangehood (pink-ish broadband + motor hum), a tap that runs on and off, dish and cutlery clatter,
    fridge compressor hum."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    hood = shaped_noise(rng, n, -3.0, 40, 6000) * db(-6)
    hood += sum(np.sin(2 * np.pi * 120 * k * t + rng.uniform(0, 6.3)) / k for k in (1, 2, 3, 4)) * db(-18)
    tap = shaped_noise(rng, n, 0.0, 800, 7500)
    gate = np.zeros(n)
    pos = 0
    while pos < n:
        on = int(rng.uniform(1.5, 6) * SR)
        gate[pos:pos + on] = 1.0
        pos += on + int(rng.uniform(2, 8) * SR)
    gate = np.convolve(gate, np.ones(1600) / 1600, mode="same")  # 0.1 s fades
    tap *= gate * (1 + 0.2 * np.sin(2 * np.pi * 3.1 * t)) * db(-4)
    clatter = np.zeros(n)
    for t0 in np.cumsum(rng.exponential(0.9, int(seconds * 2))):
        a = int(t0 * SR)
        k = int(rng.uniform(0.02, 0.09) * SR)
        if a + k >= n:
            break
        burst = band(rng.standard_normal(k).astype(np.float32), rng.uniform(1500, 3000), 7800, 2.0)
        clatter[a:a + k] += burst * np.exp(-np.arange(k) / (k / 4)) * rng.uniform(0.5, 4.0)
    fridge = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3)) * db(-24)
    mix = (hood + tap + clatter + fridge).astype(np.float32)
    return convolve(mix, room_ir(rng, 0.4, 0.0))


def _slice(bed: np.ndarray, rng, n: int) -> np.ndarray:
    a = int(rng.integers(len(bed)))
    return np.resize(np.roll(bed, -a), n)


# ---------------------------------------------------------------------- conditions
def base_condition(condition: str) -> str:
    return condition[: -len("-biased")] if condition.endswith("-biased") else "clean" if condition == "biased" else condition


def render(speech: np.ndarray, condition: str, seed: int, beds: dict | None = None) -> tuple[np.ndarray, dict]:
    """(int16 audio at 16 kHz, metadata) for close-talk speech under a condition ("biased" conditions are clean
    or <base>-biased audio: the prompt is the only difference)."""
    cond = base_condition(condition)
    rng = np.random.default_rng(seed)
    x = set_level(speech.astype(np.float32), SPEECH_DBFS)
    x = np.concatenate([np.zeros(int(LEAD_S * SR), np.float32), x, np.zeros(int(TAIL_S * SR), np.float32)])
    meta: dict = {"condition": cond}
    if cond == "clean":
        y = x + shaped_noise(rng, len(x), -3.0) * db(-75)
    elif cond == "farfield":
        rt60 = float(rng.uniform(0.3, 0.6))
        y = convolve(x, room_ir(rng, rt60, -11.5)) * db(-20)  # 0.3 m -> 3 m
        y = y + shaped_noise(rng, len(y), -3.0, 30, 8000) * db(-55)
        y = set_level(y, SPEECH_DBFS)  # satellite AGC brings it back up, noise floor with it
        meta.update(rt60=round(rt60, 2), distance_m=3.0)
    elif cond in ("noisy10", "noisy5"):
        snr = 10.0 if cond == "noisy10" else 5.0
        kind = "tv" if rng.random() < 0.5 else "kitchen"
        y = convolve(x, room_ir(rng, float(rng.uniform(0.3, 0.45)), 3.0))
        noise = _slice(beds[kind], rng, len(y))
        sp = slice(int(LEAD_S * SR), len(y) - int(TAIL_S * SR))
        noise = noise * (rms(y[sp]) / rms(noise[sp]) * db(-snr))
        y = y + noise
        meta.update(noise=kind, snr_db=snr)
    elif cond == "lowmic":
        y = band(x, 300, 3400, 3.0)
        y = set_level(y, SPEECH_DBFS + 6)
        y = np.tanh(y * 2.5) / 2.5 * 1.2  # soft clipping on peaks
        y = y + shaped_noise(rng, len(y), 0.0, 20, 8000) * (rms(y) * db(-30))
        y = np.round(y * 2047) / 2047  # 12-bit
        meta.update(band_hz=[300, 3400], bits=12)
    else:
        raise ValueError(f"unknown condition {condition!r} (have {', '.join(CONDITIONS)}, biased, <condition>-biased)")
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.99:
        y = y * (0.99 / peak)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16), meta
