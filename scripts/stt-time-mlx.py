import glob, os, statistics, sys, time
import mlx_whisper, wave, numpy as np
PROMPT = ("Bundook house. Rooms: kitchen, scullery, pantry, butler's pantry, living room, family room, dining, alfresco, "
          "master bedroom, ensuite, guest room, kids' rooms, study, mud room, laundry, garage, carport, pool house, gym. "
          "Devices: downlights, pendants, ceiling fans, ducted aircon, heat pump, solar, battery, gate, garage door, "
          "dehumidifier, robot vacuum. People: Peter, Amber.")
def load(f):
    with wave.open(f) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
clips = [load(f) for f in sorted(glob.glob(os.path.expanduser("~/home-ai-falcon/stt/clips/*.wav")))]
for name in sys.argv[1:]:
    repo = f"mlx-community/whisper-{name}-mlx"
    for c in clips[:3]:
        mlx_whisper.transcribe(c, path_or_hf_repo=repo, initial_prompt=PROMPT, language="en")
    ts = []
    for c in clips:
        t0 = time.perf_counter()
        mlx_whisper.transcribe(c, path_or_hf_repo=repo, initial_prompt=PROMPT, language="en")
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"{name:8s} warm, names prompt: mean {statistics.mean(ts):.2f} s  p50 {ts[len(ts)//2]:.2f} s  p90 {ts[int(len(ts)*.9)]:.2f} s  ({len(ts)} clips, ~2.2 s audio each)", flush=True)
