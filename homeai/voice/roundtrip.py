"""TTS -> acoustic condition -> STT round trip for every user turn of an eval set: what would the model actually hear?

Every user turn is spoken by a Piper voice (what Home Assistant uses for TTS; British, Australian and American, male
and female, three speaking rates), rendered under an acoustic condition (homeai/voice/acoustics.py: clean, farfield,
noisy10, noisy5, lowmic) and transcribed by a local STT engine (homeai/voice/stt.py: faster-whisper tiny ... large-v3-
turbo, Parakeet, Moonshine, Kroko). "biased" is clean audio with Whisper's initial_prompt built from the house's names
the way wyoming-faster-whisper does from Home Assistant; "<condition>-biased" adds the prompt to any condition.

Outputs, per (set, engine, condition), in --out (data/generated/heard):
  <set>-<engine>-<condition>.jsonl      the source rows with each user turn's content replaced by the transcript; the
                                        written text stays in messages[i]["_written"] (leaf rows: "query" / "_written")
                                        and ex["_voice"] records engine, condition, and per turn voice, rate, noise,
                                        seconds. The reference (tool calls, answers, must_mention) is untouched, so
                                        homeai.eval.e2e scores what the model did from what it heard against what the
                                        user meant.
  <set>-<engine>-<condition>.wer.json   WER, house-vocabulary WER (rooms, devices, people, numbers, command words),
                                        the share of turns whose meaning changed, latency, worst examples
  latency.json                          single-stream CPU latency per engine (--latency)

Audio (Piper output, 16 kHz) and transcripts are cached under --cache by (text, voice, rate, condition, seed, engine,
beam, prompt), so reruns and new sets that share utterances (the -b twins) cost nothing.

  # in the container (scripts/voice-roundtrip.sh wraps docker run; repo at /work, cache at /cache)
  scripts/voice-roundtrip.sh --fetch --stt base,parakeet
  scripts/voice-roundtrip.sh --sets probe --stt tiny,base,small,parakeet --conditions clean,farfield,noisy10,noisy5,lowmic
  scripts/voice-roundtrip.sh --sets test-v2,wild,wild2,bigbench-bundook --stt small --conditions clean,biased
  scripts/voice-roundtrip.sh --latency 40 --stt tiny,base,small,parakeet
  scripts/voice-roundtrip.sh --summary
  # then, like any other set:
  python3 -m homeai.eval.e2e --model homeai-4b-v2 --data data/generated/heard/wild-small-clean.jsonl --out outputs/eval/homeai-4b-v2-wild-heard-small-clean
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import statistics
import sys
import time

from ..host import host_info
from . import vocab

SETS = {  # name -> (file, row filter)
    "test-v2": ("data/generated/test-v2.jsonl", None),
    "wild": ("data/generated/wild.jsonl", None),
    "wild2": ("data/generated/wild2.jsonl", None),
    "bigbench-bundook": ("data/generated/bigbench.jsonl", "bundook"),
    "test-v2-b": ("data/generated/test-v2-b.jsonl", None),
    "wild-b": ("data/generated/wild-b.jsonl", None),
    "wild2-b": ("data/generated/wild2-b.jsonl", None),
    "bigbench-b-bundook": ("data/generated/bigbench-b.jsonl", "bundook"),
    "bigbench": ("data/generated/bigbench.jsonl", None),
    "bigbench-b": ("data/generated/bigbench-b.jsonl", None),
    "leaf-test-bundook": ("data/generated/leaf-test-bundook.jsonl", None),
}
MAIN = ("test-v2", "wild", "wild2", "bigbench-bundook")
PROBE = {"test-v2": 25, "wild": 30, "wild2": 30, "bigbench-bundook": 15}  # user turns per set in the breadth probe

# Piper voices: id -> (model, speaker, accent, gender). No Piper voice is trained as en_AU; the two VCTK speakers
# here are the corpus's Australian men (p326 Sydney, p374). No Australian woman exists in any Piper voice.
VOICES = {
    "gb-alan": ("en_GB-alan-medium", None, "gb-scottish", "m"),
    "gb-alba": ("en_GB-alba-medium", None, "gb-scottish", "f"),
    "gb-northern-m": ("en_GB-northern_english_male-medium", None, "gb-northern", "m"),
    "gb-southern-f": ("en_GB-southern_english_female-low", None, "gb-southern", "f"),
    "gb-jenny": ("en_GB-jenny_dioco-medium", None, "gb", "f"),
    "gb-cori": ("en_GB-cori-medium", None, "gb", "f"),
    "au-p326": ("en_GB-vctk-medium", "p326", "au", "m"),
    "au-p374": ("en_GB-vctk-medium", "p374", "au", "m"),
    "us-amy": ("en_US-amy-medium", None, "us", "f"),
    "us-ryan": ("en_US-ryan-medium", None, "us", "m"),
    "us-lessac": ("en_US-lessac-medium", None, "us", "f"),
}
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/{lang}/{name}/{q}/{model}.onnx"
RATES = (0.85, 1.0, 1.0, 1.15)  # Piper length_scale: quick, normal (twice as likely), slow
FEMALE = set("Amber Freya Hannah Grace Lucy Chloe Mia Jess Kate Priya Sophie Nina Tara Amelia Ella Isla Emma Ruby "
             "Elena Molly Sofia Yuki Zoe".split())
MALE = set("Peter Pete Kai Jack Josh Noah Owen Liam Tom Ben Ryan Oliver Dan Nathan Max Raj Marcus Leo Ethan Harry".split())
TV_LINES = [
    "Police are asking anyone with information to come forward.",
    "And that's the weather for the week ahead, with showers clearing by Thursday.",
    "I told you already, I don't want to talk about it.",
    "Coming up after the break, the best-selling gadgets of the year.",
    "The home side needs two runs from the final over.",
    "Where were you on the night of the fourteenth?",
    "Prices are down fifteen percent across the country.",
    "You never listen to me, that's always been the problem.",
    "Tonight's winning numbers are four, eleven, twenty-three and thirty-eight.",
    "Stay with us, there's plenty more to come.",
]
TTS_VERSION = "tts1"
HOST = host_info()  # the container is passed the Mac's name and chip (scripts/voice-roundtrip.sh): speed is per host


def h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------- sets and turns
def load_set(name: str, root: str) -> list[dict]:
    if name == "probe":
        rows = []
        for s, budget in PROBE.items():
            got = 0
            for ex in sorted(load_set(s, root), key=lambda r: h(r["id"])):
                n = len(user_turns(ex))
                if got + n > budget + 1:
                    continue
                rows.append({**ex, "_set": s})
                got += n
                if got >= budget:
                    break
        return rows
    path, where = SETS.get(name, (name, None))
    rows = [json.loads(l) for l in open(os.path.join(root, path)) if l.strip()]
    if where:
        rows = [r for r in rows if where in (r.get("big") or {}).get("kind", "")]
    return rows


def set_name(spec: str) -> str:
    return spec if spec in SETS or spec == "probe" else os.path.splitext(os.path.basename(spec))[0]


def user_turns(ex: dict) -> list[tuple]:
    """[(message index or None for a leaf "query" row, written text)]."""
    if "messages" in ex:
        return [(j, m.get("_written", m["content"])) for j, m in enumerate(ex["messages"]) if m["role"] == "user"]
    return [(None, ex.get("_written", ex["query"]))]


def voice_for(ex: dict) -> str:
    name = (ex.get("speaker") or {}).get("name")
    pool = [v for v, (_, _, _, g) in VOICES.items() if (g == "f") == (name in FEMALE)] if name in FEMALE | MALE else list(VOICES)
    return pool[int(h(ex.get("id"), "voice"), 16) % len(pool)]


def rate_for(ex: dict, k: int) -> float:
    return RATES[int(h(ex.get("id"), k, "rate"), 16) % len(RATES)]


# ---------------------------------------------------------------------- TTS
class TTS:
    def __init__(self, cache: str):
        self.cache, self.voices = cache, {}

    def path(self, key: str) -> str:
        return os.path.join(self.cache, "tts", key[:2], key + ".npy")

    def voice(self, model: str):
        if model not in self.voices:
            from piper import PiperVoice
            self.voices[model] = PiperVoice.load(fetch_voice(model, self.cache))
        return self.voices[model]

    def speak(self, text: str, voice: str, rate: float):
        """(tts key, int16 16 kHz close-talk audio), synthesised once and cached (Piper's VITS sampling is random,
        so the cache is also what makes reruns identical)."""
        import numpy as np
        from piper import SynthesisConfig
        from .acoustics import resample
        key = h(TTS_VERSION, text, voice, rate)
        p = self.path(key)
        if os.path.exists(p):
            return key, np.load(p)
        model, spk, _, _ = VOICES[voice]
        v = self.voice(model)
        sid = v.config.speaker_id_map.get(spk) if spk else None
        chunks = list(v.synthesize(text, syn_config=SynthesisConfig(speaker_id=sid, length_scale=rate)))
        sr = chunks[0].sample_rate if chunks else 16000
        x = np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(1600, np.float32)
        y = (np.clip(resample(x, sr), -1, 1) * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p + ".tmp.npy", y)
        os.replace(p + ".tmp.npy", p)
        return key, y


def fetch_voice(model: str, cache: str) -> str:
    import urllib.request
    d = os.path.join(cache, "piper")
    p = os.path.join(d, model + ".onnx")
    if not os.path.exists(p) or not os.path.exists(p + ".json"):
        os.makedirs(d, exist_ok=True)
        lang, name, q = model.split("-")
        url = PIPER_URL.format(lang=lang, name=name, q=q, model=model)
        for suffix in ("", ".json"):
            urllib.request.urlretrieve(url + suffix, p + suffix + ".tmp")
            os.replace(p + suffix + ".tmp", p + suffix)
    return p


def build_beds(tts: TTS, cache: str) -> str:
    """Noise beds (TV, kitchen) as one .npz, built once: the TV's dialogue is Piper too."""
    import numpy as np
    from . import acoustics
    p = os.path.join(cache, "beds", f"beds-{acoustics.VERSION}.npz")
    if not os.path.exists(p):
        voices = ["us-ryan", "gb-cori", "au-p326", "us-lessac", "gb-northern-m"]
        lines = [tts.speak(t, voices[i % len(voices)], 1.0)[1].astype(np.float32) / 32768 for i, t in enumerate(TV_LINES)]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez(p + ".tmp.npz", tv=acoustics.tv_bed(lines), kitchen=acoustics.kitchen_bed())
        os.replace(p + ".tmp.npz", p)
    return p


# ---------------------------------------------------------------------- STT workers
_W: dict = {}


def _init(engine: str, threads: int, cache: str, beam: int, beds: str):
    import numpy as np
    from . import stt
    _W["engine"] = stt.load(engine, threads, cache, beam)
    _W["beds"] = dict(np.load(beds))
    _W["tts"] = TTS(cache)


def _job(batch):
    import numpy as np
    from .acoustics import SR, render
    from .stt import timed
    out = []
    for key, tts_key, cond, seed, prompt in batch:
        audio, meta = render(np.load(_W["tts"].path(tts_key)).astype(np.float32) / 32768, cond, seed, _W["beds"])
        text, secs = timed(_W["engine"], audio, prompt)
        out.append((key, text, round(secs, 4), round(len(audio) / SR, 3), meta))
    return out


class Transcripts:
    """<cache>/stt/<engine>.jsonl: one line per (audio, condition, engine, prompt) key."""

    def __init__(self, cache: str, engine: str):
        self.path = os.path.join(cache, "stt", f"{engine}.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.rows = {}
        if os.path.exists(self.path):
            for l in open(self.path):
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    continue  # a line cut short by a killed run
                self.rows[r["k"]] = r
        self.fh = open(self.path, "a")

    def add(self, key, text, secs, dur, meta):
        r = {"k": key, "t": text, "s": secs, "d": dur, "m": meta}
        self.rows[key] = r
        self.fh.write(json.dumps(r) + "\n")
        self.fh.flush()


def workers_for(engine: str, mem_gb: float, cpus: int) -> tuple[int, int]:
    """(worker processes, threads each) that fit the container: at most cpus/2 workers, memory with 1 GB headroom."""
    from .stt import MEMORY_GB
    w = max(1, min(cpus // 2, int((mem_gb - 1.0) // MEMORY_GB.get(engine, 2.0))))
    return w, max(1, cpus // w)


# ---------------------------------------------------------------------- scoring
def report(set_name_: str, engine: str, cond: str, rows_out: list[dict], scored: list[dict], prompted: bool) -> dict:
    n = len(scored)
    words = sum(s["words"] for s in scored)
    errs = sum(s["errors"] for s in scored)
    vk = {k: [sum(s["vocab"][k][0] for s in scored), sum(s["vocab"][k][1] for s in scored)] for k in vocab.KINDS}
    vn, ve = sum(v[0] for v in vk.values()), sum(v[1] for v in vk.values())
    changed = [s for s in scored if s["lost"]]
    lost_kinds = collections.Counter(k for s in scored for k in {k for k, _ in s["lost"]})
    secs = sorted(s["seconds"] for s in scored)
    dur = sum(s["duration"] for s in scored)

    def by(field):
        g = collections.defaultdict(lambda: [0, 0, 0, 0])
        for s in scored:
            v = g[s[field]]
            v[0] += s["words"]; v[1] += s["errors"]; v[2] += 1; v[3] += bool(s["lost"])
        return {str(k): {"turns": v[2], "wer": round(v[1] / max(1, v[0]), 4), "meaning_changed": round(v[3] / max(1, v[2]), 4)}
                for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}

    return {
        "set": set_name_, "stt": engine, "condition": cond, "prompted": prompted, **HOST, "utterances": n, "words": words,
        "errors": errs, "wer": round(errs / max(1, words), 4), "exact": round(sum(s["exact"] for s in scored) / max(1, n), 4),
        "vocab_words": vn, "vocab_errors": ve, "vocab_wer": round(ve / max(1, vn), 4),
        "vocab": {k: {"n": v[0], "err": v[1], "wer": round(v[1] / max(1, v[0]), 4)} for k, v in vk.items()},
        "meaning_changed": round(len(changed) / max(1, n), 4), "meaning_changed_n": len(changed),
        "meaning_by_kind": {k: round(lost_kinds[k] / max(1, n), 4) for k in vocab.KINDS},
        "empty": sum(1 for s in scored if not s["heard"].strip()),
        "seconds": {"mean": round(sum(secs) / max(1, n), 4), "p50": round(secs[n // 2], 4) if n else 0,
                    "p90": round(secs[int(n * 0.9)], 4) if n else 0, "rtf": round(sum(secs) / max(1e-9, dur), 4),
                    "note": "wall time per utterance with parallel workers; see latency.json for single-stream"},
        "by_voice": by("voice"), "by_rate": by("rate"), "by_noise": by("noise"), "by_set": by("set"),
        "samples": [{k: s[k] for k in ("id", "turn", "voice", "written", "heard", "lost")}
                    for s in sorted(changed, key=lambda s: (-len(s["lost"]), s["id"]))[:40]],
    }


# ---------------------------------------------------------------------- main passes
def plan(sets: dict, tts: TTS, voice_override=None):
    """Every user turn of every set, spoken: [(set, row index, turn k, msg index, text, voice, rate, tts key)]."""
    turns, spoken = [], {}
    t0, made = time.time(), 0
    for s, rows in sets.items():
        for i, ex in enumerate(rows):
            v = voice_override or voice_for(ex)
            for k, (j, text) in enumerate(user_turns(ex)):
                r = rate_for(ex, k)
                sk = (text, v, r)
                if sk not in spoken:
                    had = os.path.exists(tts.path(h(TTS_VERSION, text, v, r)))
                    spoken[sk] = tts.speak(text, v, r)[0]
                    made += not had
                    if made and made % 100 == 0 and not had:
                        print(f"  tts {made} new utterances, {time.time() - t0:.0f}s", flush=True)
                turns.append((s, i, k, j, text, v, r, spoken[sk]))
    print(f"tts: {len(turns)} turns, {len(spoken)} distinct utterances, {made} synthesised in {time.time() - t0:.0f}s", flush=True)
    return turns


class Houses:
    def __init__(self):
        self.terms, self.prompts, self.count = {}, {}, None

    def get(self, ex):
        k = vocab.house_key(ex)
        if k not in self.terms:
            house = vocab.house_of(ex)
            self.terms[k] = vocab.Terms(house)
            if self.count is None:
                from .stt import whisper_token_counter
                self.count = whisper_token_counter()
            self.prompts[k] = vocab.bias_prompt(house, self.count)
        return self.terms[k], self.prompts[k]


def run(a, root: str):
    from .acoustics import base_condition, seed_of
    from .acoustics import VERSION as AC_VERSION
    from .stt import ENGINES, WHISPER
    sets = {set_name(s): load_set(s, root) for s in a.sets.split(",")}
    tts = TTS(a.cache)
    turns = plan(sets, tts)
    beds = build_beds(tts, a.cache)
    if a.tts_only:
        return
    houses = Houses()
    os.makedirs(a.out, exist_ok=True)
    for engine in a.stt.split(","):
        if engine not in ENGINES:
            sys.exit(f"unknown engine {engine}; have {', '.join(ENGINES)}")
        cache = Transcripts(a.cache, engine)
        conds = [c for c in a.conditions.split(",") if engine in WHISPER or not (c == "biased" or c.endswith("-biased"))]
        skipped = set(a.conditions.split(",")) - set(conds)
        if skipped:
            print(f"{engine}: no prompt support, skipping {', '.join(sorted(skipped))}")
        jobs, keyed = [], {}
        for cond in conds:
            biased = cond == "biased" or cond.endswith("-biased")
            base = base_condition(cond)
            for t in turns:
                s, i, k, j, text, v, r, tk = t
                ex = sets[s][i]
                prompt = houses.get(ex)[1] if biased else ""
                seed = seed_of(ex.get("id"), k, base)
                key = h("stt1", tk, base, AC_VERSION, seed, engine, a.beam_size if engine in WHISPER else 0, prompt)
                keyed[(cond, t)] = key
                if key not in cache.rows:
                    jobs.append((key, tk, base, seed, prompt))
        jobs = list({j_[0]: j_ for j_ in jobs}.values())
        if jobs:
            w, th = (a.workers, a.threads) if a.workers else workers_for(engine, a.mem_gb, a.cpus)
            print(f"{engine}: {len(jobs)} transcriptions to run ({len(keyed) - len(jobs)} cached), {w} workers x {th} threads",
                  flush=True)
            t0, done = time.time(), 0
            batches = [jobs[b:b + 8] for b in range(0, len(jobs), 8)]
            with cf.ProcessPoolExecutor(w, initializer=_init, initargs=(engine, th, a.cache, a.beam_size, beds)) as pool:
                for res in cf.as_completed([pool.submit(_job, b) for b in batches]):
                    for r_ in res.result():
                        cache.add(*r_)
                    done += len(res.result())
                    if done % 200 < 8 or done == len(jobs):
                        el = time.time() - t0
                        print(f"  {engine} {done}/{len(jobs)} {el:.0f}s ({el / done:.2f}s each, {len(jobs) / done * el - el:.0f}s left)",
                              flush=True)
        for cond in conds:
            for s, rows in sets.items():
                write_heard(a.out, s, engine, cond, rows, [t for t in turns if t[0] == s], keyed, cache, houses)


def write_heard(out, s, engine, cond, rows, turns, keyed, cache, houses):
    biased = cond == "biased" or cond.endswith("-biased")
    by_row = collections.defaultdict(list)
    for t in turns:
        by_row[t[1]].append(t)
    rows_out, scored = [], []
    for i, ex in enumerate(rows):
        ex2 = json.loads(json.dumps(ex))
        terms, prompt = houses.get(ex)
        meta_turns = []
        for t in by_row[i]:
            _, _, k, j, text, v, r, tk = t
            c = cache.rows[keyed[(cond, t)]]
            heard = c["t"]
            if j is None:
                ex2["_written"], ex2["query"] = text, heard
            else:
                ex2["messages"][j]["_written"] = text
                ex2["messages"][j]["content"] = heard
            sc = vocab.score(terms, text, heard)
            meta_turns.append({"turn": k, "voice": v, "rate": r, **{x: y for x, y in c["m"].items() if x != "condition"},
                               "seconds": c["s"], "audio_s": c["d"], "wer": round(sc["errors"] / max(1, sc["words"]), 3),
                               "lost": [f"{a}:{b}" for a, b in sc["lost"]]})
            scored.append({**sc, "id": ex.get("id"), "turn": k, "voice": v, "rate": r, "noise": c["m"].get("noise", "-"),
                           "set": ex.get("_set", s), "written": text, "heard": heard, "seconds": c["s"], "duration": c["d"]})
        ex2["_voice"] = {"stt": engine, "condition": cond, "prompt": prompt if biased else None, "turns": meta_turns}
        rows_out.append(ex2)
    base = os.path.join(out, f"{s}-{engine}-{cond}")
    with open(base + ".jsonl", "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    rep = report(s, engine, cond, rows_out, scored, biased)
    json.dump(rep, open(base + ".wer.json", "w"), indent=2)
    print(f"  {os.path.basename(base)}: WER {rep['wer']:.1%}  vocab {rep['vocab_wer']:.1%}  meaning changed "
          f"{rep['meaning_changed']:.1%}  ({rep['utterances']} turns)", flush=True)


def latency(a, root: str):
    """Single-stream CPU latency: one engine at a time, nothing else running, --latency-threads threads (the add-on's
    default is 4), on the first N clean probe utterances after two warm-up calls."""
    import numpy as np
    from . import stt
    from .acoustics import SR, render, seed_of
    tts = TTS(a.cache)
    sets = {"probe": load_set("probe", root)}
    turns = plan(sets, tts)[: a.latency + 2]
    beds = dict(np.load(build_beds(tts, a.cache)))
    audio = [render(np.load(tts.path(t[7])).astype(np.float32) / 32768, "clean", seed_of(sets["probe"][t[1]].get("id"), t[2], "clean"), beds)[0]
             for t in turns]
    p = os.path.join(a.out, "latency.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    for engine in a.stt.split(","):
        t0 = time.perf_counter()
        e = stt.load(engine, a.latency_threads, a.cache, a.beam_size)
        load_s = time.perf_counter() - t0
        for x in audio[:2]:
            e.transcribe(x)
        secs = sorted(stt.timed(e, x)[1] for x in audio[2:])
        dur = sum(len(x) / SR for x in audio[2:])
        n = len(secs)
        res[engine] = {"threads": a.latency_threads, "n": n, "mean": round(statistics.mean(secs), 3),
                       "p50": round(secs[n // 2], 3), "p90": round(secs[int(n * 0.9)], 3), "max": round(secs[-1], 3),
                       "rtf": round(sum(secs) / dur, 3), "audio_s_mean": round(dur / n, 2), "load_s": round(load_s, 1),
                       "beam_size": a.beam_size if engine in stt.WHISPER else None, **HOST,
                       "cpu": os.uname().machine, "cpus": os.cpu_count(),
                       "host_load_at_start": os.environ.get("HOMEAI_HOST_LOAD", "").strip("{} ")}
        print(f"{engine}: p50 {res[engine]['p50']:.2f}s  p90 {res[engine]['p90']:.2f}s  RTF {res[engine]['rtf']:.3f}", flush=True)
        del e
        json.dump(res, open(p, "w"), indent=2)


def summary(a):
    """Decision table: engine x condition -> WER / house-vocab WER / meaning changed, pooled over the chosen sets."""
    import glob
    reps = [json.load(open(f)) for f in glob.glob(os.path.join(a.out, "*.wer.json"))]
    want = a.summary_sets.split(",")
    lat = json.load(open(os.path.join(a.out, "latency.json"))) if os.path.exists(os.path.join(a.out, "latency.json")) else {}
    pooled = collections.defaultdict(lambda: collections.Counter())
    for r in reps:
        if r["set"] in want:
            c = pooled[(r["stt"], r["condition"])]
            for k in ("utterances", "words", "errors", "vocab_words", "vocab_errors", "meaning_changed_n"):
                c[k] += r[k]
            c["sets"] += 1
    from .stt import ENGINES
    conds = sorted({c for _, c in pooled}, key=lambda c: (["clean", "biased", "farfield", "noisy10", "noisy5", "lowmic"] + [c]).index(c))
    lines = [f"sets: {', '.join(want)}  (cells: WER / house-vocab WER / meaning changed; latency: single stream, CPU)", "",
             "| engine | " + " | ".join(conds) + " | p50 s | p90 s | RTF | timed on |", "|---" * (len(conds) + 5) + "|"]
    table = {}
    for e in [e for e in ENGINES if any(k[0] == e for k in pooled)]:
        cells = []
        for c in conds:
            v = pooled.get((e, c))
            if not v:
                cells.append("-")
                continue
            w, vw, m = v["errors"] / max(1, v["words"]), v["vocab_errors"] / max(1, v["vocab_words"]), v["meaning_changed_n"] / max(1, v["utterances"])
            table.setdefault(e, {})[c] = {"wer": round(w, 4), "vocab_wer": round(vw, 4), "meaning_changed": round(m, 4),
                                          "utterances": v["utterances"], "sets": v["sets"]}
            cells.append(f"{w:.1%} / {vw:.1%} / {m:.1%}")
        L = lat.get(e, {})
        lines.append(f"| {e} | " + " | ".join(cells) + f" | {L.get('p50', '-')} | {L.get('p90', '-')} | {L.get('rtf', '-')} | {L.get('host', '-')} |")
    print("\n".join(lines))
    json.dump({"sets": want, "table": table, "latency": lat}, open(os.path.join(a.out, f"summary-{'+'.join(want)}.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", default="probe", help=f"comma list: probe, {', '.join(SETS)}, or jsonl paths")
    ap.add_argument("--stt", default="base", help="comma list of engines (homeai/voice/stt.py)")
    ap.add_argument("--conditions", default="clean", help="comma list: clean farfield noisy10 noisy5 lowmic biased <cond>-biased")
    ap.add_argument("--out", default="data/generated/heard")
    ap.add_argument("--cache", default=os.environ.get("HOMEAI_VOICE_CACHE", os.path.expanduser("~/.cache/homeai-voice")))
    ap.add_argument("--beam-size", type=int, default=5, help="Whisper beam (HA add-on: 5 on x86, 1 on ARM)")
    ap.add_argument("--workers", type=int, default=0, help="STT worker processes (0: fit to --cpus / --mem-gb)")
    ap.add_argument("--threads", type=int, default=0, help="threads per worker (with --workers)")
    ap.add_argument("--cpus", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--mem-gb", type=float, default=float(os.environ.get("HOMEAI_VOICE_MEM_GB", 7)))
    ap.add_argument("--latency", type=int, default=0, help="time N clean probe utterances per engine, single stream")
    ap.add_argument("--latency-threads", type=int, default=4)
    ap.add_argument("--summary", action="store_true", help="print the engine x condition table from existing reports")
    ap.add_argument("--summary-sets", default="probe")
    ap.add_argument("--tts-only", action="store_true", help="synthesise and cache the sets' audio, then exit (run it "
                    "on one machine and copy <cache>/tts to the others: Piper sampling is random, the cache is the audio)")
    ap.add_argument("--fetch", action="store_true", help="download the Piper voices and the --stt models, then exit")
    a = ap.parse_args()
    root = os.getcwd()
    if a.summary:
        return summary(a)
    if a.fetch:
        from . import stt
        for model in {m for m, _, _, _ in VOICES.values()}:
            print("piper", fetch_voice(model, a.cache))
        for e in a.stt.split(","):
            if e in stt.WHISPER:
                from faster_whisper.utils import download_model
                print(e, download_model(stt.WHISPER[e]))
            else:
                print(e, stt.fetch_sherpa(e, a.cache))
        return
    if a.latency:
        return latency(a, root)
    run(a, root)


if __name__ == "__main__":
    main()
