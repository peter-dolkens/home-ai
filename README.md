# home-ai

Small, local, fine-tuned LLMs (Ollama on a GPU box, MLX on a Mac mini) that act as the voice brain of an
off-grid Home Assistant house: room control, "check on the house", "where is Amber",
battery/solar/load questions, and bedtime/leaving routines.

## Where it stands (14 Oct 2026)

Full results (every design, the house heard table, Mac vs 3090 champions, the Ollama 0.40.3 re-benchmark): the
[Bundook Voice Lab results page](https://claude.ai/artifact/UVsAKngjM86oxCj9b7ZaB4), rebuilt with `scripts/report-refresh.sh`.

The target is the new Bundook build: about 11,600 Home Assistant entities, voice only, off-grid. Two serving tiers are kept
apart, each with its own champions: a **low-power Mac mini** (M2 Pro, MLX, ~5-10 W idle) as the always-on voice box, and a
**3090-equipped** GPU box (cassowary) for training, evaluation and anything heavy.

Each tier is ranked on its own hardware; the champions need not be the same model or family.

**Low-power Mac mini** (M2 Pro, 6-bit MLX, warm reply with the house prompt cached):

| | model | house heard | warm reply |
|---|---|---|---|
| Champion (best within 1 s) | LFM2.5 1.2B, design B | 81.9% (typed 89.3%) | 0.73 s, ~1.2 s end of speech to action |
| Fastest | LFM2.5 230M, design B | 76.4% | 0.24 s |
| Fast option | LFM2.5 350M, design B | 77.0% | 0.32 s |
| Too slow here | LFM2.5 2.6B, design B | 82.2% | 2.6 s |

**3090-equipped** (cassowary, Ollama 0.40.3):

| | model | score | speed |
|---|---|---|---|
| House, spoken | LFM2.5 2.6B, design B | 82.2% heard (1 seed; 1.2B 81.9%, Spark 81.2%: within seed noise) | 0.5 s / request |
| Design B, unfamiliar houses | Spark X2.5 4B (3 epochs) | 85.5% on the B sets | 0.9 s / request |
| Design A | Qwen3.5 4B v9 | 85.4% quality | 1.2 s / request |

- **Design B** (rooms-only prompt plus lookup tools, below) won or tied every comparison; designs A (whole inventory in the
  prompt), C/D (orchestrator plus per-room models) and E (fast model acts, slow model checks) are kept as baselines.
- **Training recipe of the Mac champion** (also used for every LFM size above): LoRA r256, 4 epochs, `data/generated/train-b-v8.jsonl` (random houses, hard far-field /
  TV-noise speech augmentation, 1,650 conversations on the Bundook plan, verified wild transcripts); config
  `configs/lfm25-1.2b-lora-b-v8r256e4.json`. More epochs, 8-bit, a lower learning rate or a second augmentation draw did not help.
- **Speech-to-text**: Whisper medium with the house's device and room names as its initial prompt (11-12% word error at 3 m
  and in TV noise vs Parakeet's 18-29%) is the biggest single lever. `homeai/voice/` builds the heard test sets (synthetic
  voices transcribed by each engine).
- **How sure**: seed alone moves the heard mean by 0.2-1.7 points, so picks rest on two seeds; 46% of the house test turns
  appear word for word in training (`homeai.eval.unseen_split` scores the rest). Everything is synthetic so far: real
  recordings in the house are the next real test.

Machines: cassowary (RTX 3090, the driver `scripts/overnight.sh` with `scripts/watchdog.sh`, Frigate shares its GPU and disk),
lapwing (RTX 3080, `scripts/train-leaf-qwen.sh` queues), falcon (Mac mini M2 Pro, `homeai/eval/falcon_stream.py` times MLX and
Ollama builds) and a dev Mac (`scripts/train-mlx.sh`, `scripts/falcon-mlx-build.sh`). `scripts/small-lfm-followup.sh` fans a
newly trained run out to the house heard evals, the cassowary benchmark (`outputs/bench-o040.sh` pattern) and falcon;
`scripts/report-refresh.sh` rebuilds the results page (`homeai/eval/report_v2.py`) from every host's reports.

## Training methods tried (Sep-Oct 2026)

All runs use supervised fine-tuning on the synthetic transcripts (loss on assistant turns only, tool results masked), then
outcome evals on held-out houses and, for voice, on transcribed speech. What each lever bought:

| Method / lever | Where | Result |
|---|---|---|
| **QLoRA** (4-bit base, LoRA r16) | Qwen3.5 4B / 9B, Gemma 4 E4B | Works; slightly behind bf16 LoRA on the same data (4B v5: 83.0 vs 84.0 quality). Only useful to fit bigger models. |
| **bf16 LoRA** | every current model | The workhorse. Unsloth on CUDA; `homeai/train/train_mlx.py` reproduces it with MLX on Apple silicon (the 0.8B was trained on the Mac mini). |
| **LoRA rank** r16 -> r256 | LFM2.5 1.2B | The biggest model-side lever: house heard ~74% -> ~79% (r16 to r256); the 2B stopped gaining at r32, Spark 4B at r32. |
| **Epochs** 2 / 3 / 4 / 6 | LFM2.5 1.2B r256; 350M; 230M | 4 is best for the 1.2B (heard 79.0 / 79.7 / 81.9 / 81.0); 6 no better for the small LFMs; 3 helps Spark. |
| **Learning rate** 2e-4 vs 1e-4 | LFM 1.2B | No difference. |
| **Rejection-sampling RL** (keep the model's own successful samples, retrain) | Qwen 4B (rft1, rft2) | No gain: 88% of training turns were already solved and the unsolved ones can't be learned this way; rft1 also learned to skip confirmations. |
| **Self / teacher distillation** (sequence level: the 4B's own verified transcripts of wild-style conversations) | 4B, 2B, LFM | +4 points on unfamiliar houses for the 4B and LFM; neutral or negative for the 2B. Now part of the training mix (B-v8). |
| **Speech (ASR) augmentation** (training turns round-tripped through TTS + STT, hard far-field / noise draws) | all voice models | The biggest data lever for spoken requests (LFM +4.3 at 3 m); a second draw or medium-Whisper-only augmentation added nothing. |
| **Prompt layouts** (classic, stable_prefix, user_context, clock_in_user) | 4B v10-v12, design B | Accuracy-neutral; user_context makes the Mac's prompt cache hit (6 s -> 2.6 s warm for the 4B). |
| **Designs** A-E (whole inventory, lookup tools, tiered rooms, fan-out, fast+slow pairs) | all | Design B (lookup tools) won or tied every comparison. |
| **Base models** | Qwen3.5 0.8B-27B, Gemma 4 E4B, Spark X2.5 4B, LFM2.5 230M-2.6B, Cactus Needle | LFM2.5 1.2B is the Mac champion; untuned models score 31-60%. |
| **Seeds** | picks | Two seeds per pick: seed alone moves the heard mean by 0.2-1.7 points. |
| **Full fine-tuning** (`"full_finetune": true`: every weight, lr 2e-5, 3 epochs) | LFM2.5 350M, 230M | Worse than LoRA: 350M heard 75.0 / typed 84.2 vs 77.0 / 85.5. Fully tuned weights also quantise badly (230M at Q4_K_M: typed 60.5, heard 53.5). |
| **Seed averaging** ("model soup", `scripts/soup.py`) | LFM2.5 1.2B, 350M picks | Mixed: 1.2B soup worse (heard 79.9 vs 81.9), 350M soup slightly better on heard (78.4 vs 77.0). |
| **MoE LoRA** (attention, convs and dense layers; experts frozen) | LFM2.5 8B-A1B | Too big to fine-tune on the shared 3090: the frozen bf16 weights leave ~1.2 GiB beside Frigate and Wyoming, and four runs ran out of memory (last at step ~370 of 2144). |

Not yet tried: continued pretraining, preference optimisation (DPO / ORPO / KTO on the models' own failures), logit-level
distillation from a larger model, quantisation-aware training (or DWQ) for the Mac's 6-bit builds.

## How it works

The model never sees the whole house. Its system prompt holds a compact **inventory**
(each area with the device classes it contains, plus three house-wide groups: `security`,
`energy`, `weather`) and two tools:

| tool | purpose |
|---|---|
| `get_state(areas, classes)` | pull detailed state for just the rooms / device classes a task needs |
| `call_service(domain, service, target, data)` | act, exactly like a Home Assistant service call (area_id or entity_id targets) |

So a request flows in two stages, which is what the fine-tune teaches:

1. **decide what matters**: "where is Amber" → `get_state(classes=[motion, occupancy])`;
   "check on the house" → `get_state(classes=[lock, door, window, garage, alarm, light, motion, ...])`;
   "turn on the bedroom and office lights" → no fetch at all, just `light.turn_on` with two `area_id`s.
2. **reason over the returned slice and answer or act**, e.g. elimination: two rooms show
   motion, the speaker is in one of them, one other person is home, so Amber is in the other.

The same prompt builder and tool schema are used for training data, evaluation and the live
bridge, so the model sees exactly the same format at runtime as in training.

Prompt layout: up to v9 the clock, speaker and presence lines sit near the top of the system
prompt (`classic`), so no prompt-prefix cache survives from one request to the next. From v10
the same lines sit in a final `Current context:` block (`stable_prefix`), and everything in
front of them stays the same for every request in a house; v11 puts them on a `[context]` line at
the start of each user turn (`user_context`). Both cost clarification, so v12 moves only the clock
(`clock_in_user`): the system prompt is classic without its `Now:` line and each user turn starts
with `[now] <day> <HH:MM>`, so the system message is identical for every request from one room
while nobody comes or goes. A model's training config sets it
with `"prompt_layout"`, and the train/val loader, the evals, the tiered eval and the bridge all
follow it (`homeai/prompt_layout.py`; override with `--prompt-layout`).

### Why synthetic data on random houses (plus one real profile)

Every training example is built on a freshly generated house (random rooms, devices, people,
time of day, weather, battery state). The model therefore learns *behaviours* (how to read an
inventory, what to fetch, how to reason about presence and power, how to phrase spoken answers),
not the layout of one house. Your real house is injected at runtime by the bridge.

By default 40% of examples use the **Bundook profile** ([profiles.py](homeai/house/profiles.py)),
the new build as drawn at plan.dolkens.house: Pete's and Amber's offices, the mmWave presence
sensors in nearly every room, Somfy blinds and window actuators, per-room air conditioning,
the ~52 kWh battery and ~30 kW array, hydronic heating, heat-pump hot water, and the BYD Shark
as a portable backup feed. Only the layout is fixed; time, weather, states and where people are
stay random. People at home are usually in their own office during work hours, in the dining
hall in the evening and in bed late. Update the profile as the sensor plan changes, regenerate,
retrain. Tune the mix with `--profile-share`.

## Layout

```
homeai/house/       schema, random house generator, Bundook profile/snapshot, state text rendering, HA service simulator
homeai/tools.py     design A tool schemas + execution against a House (tools_b.py: design B lookup tools)
homeai/datagen/     scenario builders, corpus/paraphrase/ASR augmentation, design-B and big-house builders
homeai/train/       Unsloth (CUDA) and MLX (Apple silicon) LoRA training, merges, Ollama import
homeai/bridge/      Home Assistant client, entity->class mapping, agent loop, CLI/HTTP server
homeai/eval/        teacher-forced and end-to-end (outcome) evals, latency / prefix-cache benches, champions, results page
homeai/tiered/      designs C/D: orchestrator plus per-room leaf models (homeai/leaf/: the room leaves)
homeai/voice/       speech round trips (TTS -> STT) for the heard test sets, confusion suite
homeai/prompt_layout.py   where the clock / speaker / presence lines sit (classic, stable_prefix, user_context, clock_in_user)
configs/            training configs, models.json (the model registry), bridge.example.json
scripts/            per-host drivers and queues (overnight.sh, watchdog.sh, train-*.sh, falcon-*.sh, archive.sh, ...)
data/generated/     training, validation and test sets (text, synthetic); data/generated/heard/: the heard (transcribed) sets
outputs/eval*/      eval reports per model and set (outputs/eval-falcon: the Mac mini stream)
tests/              offline checks (no GPU/HA/Ollama needed)
```

## Quickstart

### 1. Generate data (any machine, stdlib only)

```bash
make data            # 3000 examples -> data/generated/{train,val,test}.jsonl
make test            # offline checks of the scorer and the HA mapping
python -m homeai.datagen.build --n 20 --only presence_where --out /tmp/x   # inspect one scenario type
```

Optional token check with the real tokenizer (needs `pip install transformers jinja2`):

```bash
make dry-run
```

### 2. Train (GPU box with the 3080, Linux or WSL2)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train]"                     # unsloth pulls compatible torch/trl/transformers
make train                                    # configs/qwen25-7b.json
```

Default config is now `configs/qwen35-4b.json` (Qwen3.5-4B, March 2026): it fits a 10 GB card
comfortably, trains in well under an hour per epoch and is the faster model to serve. Once that
works end to end, run `make train CONFIG=configs/qwen35-9b.json` for the quality candidate (QLoRA on
10 GB is tight; drop `max_seq_len` to 4096 or rent a GPU for that one). `configs/qwen25-7b.json`
stays as the known-good fallback. Qwen3.5 is a hybrid thinking model; the training transcripts carry
no reasoning, so it learns to answer directly, which is what you want for voice latency. Its tool-call
format is XML-ish rather than JSON, but that is handled by the chat template on both sides.

Training details: QLoRA r=16, 2 epochs, lr 2e-4,
batch 1 × grad-accum 16, max 4608 tokens (the Bundook "check on the house" transcripts run to ~4100), loss on assistant turns only. Expect roughly
7–9 GB of VRAM and around 1–2 hours per epoch on a 3080 for 2700 examples. If you hit OOM,
switch to `configs/qwen25-3b.json` first to shake out the pipeline, or lower `max_seq_len`
to 4096 (drops ~40 of the longest Bundook away-check examples) or 3072.

If the 3080 box is awkward to reach, the same script runs on a free Colab T4 or a cheap
RunPod instance; only `make train` and `make export` need a GPU.

### 3. Export to Ollama

```bash
scripts/train-docker.sh merge configs/qwen35-4b.json                      # LoRA -> merged 16-bit safetensors (<out>/merged16)
python -m homeai.train.ollama_import --run outputs/qwen35-4b-r1 --name homeai-4b --base qwen3.5:4b
```

Why not a GGUF from llama.cpp: Ollama runs Qwen3.5 on its own engine, which expects fused
attention tensors that llama.cpp's converter does not produce, so such a GGUF fails to load
("qwen3next: layer N missing attn_qkv/attn_gate projections"). Ollama has to do the conversion
itself from safetensors (`ollama create --quantize q4_K_M`). The importer copies the merged
weights into the Ollama container, reuses the base model's `RENDERER`/`PARSER` directives, and
copies the quantised blob back to `<out>/gguf/<name>-q4_K_M.gguf` as a portable, re-importable file.

Two gotchas from the first night:

- Ollama's converter needs roughly 2.5x the model size in RAM (the 9B needed ~24 GB and was
  OOM-killed on cassowary's 31 GB). `scripts/convert-on-mac.sh` does the same conversion on the
  32 GB Mac and ships the result back.
- Use the **same Ollama version** for conversion as for serving. Newer Ollama stores Qwen3.5
  imports as per-tensor blobs that 0.21.2 cannot read and drops q4_K_M for this architecture.

### 4. Talk to it

```bash
python -m homeai.bridge.cli chat --mock --seed 7 --model homeai            # random house, no HA
python -m homeai.bridge.cli chat --mock --everyone-away --model homeai     # "check on the house" tests
python -m homeai.bridge.cli chat --mock --model qwen2.5:7b-instruct        # compare with the untuned base
```

Live house (credentials come from `.env`: `HA_HOSTNAME` plus `HA_TOKEN`, or `HA_USERNAME`/`HA_PASSWORD`):

```bash
python -m homeai.bridge.cli prompt --config configs/bridge.json       # check the inventory looks right
python -m homeai.bridge.cli chat   --config configs/bridge.json --speaker Peter --room office
python -m homeai.bridge.cli serve  --config configs/bridge.json --port 8090
# POST /chat {"text": "where is amber", "speaker": "Peter", "area": "office", "session": "office-satellite"}
```

The bridge reads HA's entity, device and area registries over the websocket API and drops
config/diagnostic, hidden and disabled entities (on the existing house that removes about half of
them). Remaining noise (camera overlay toggles, "Led indication" switches) is caught by name
patterns. Only entities with an area, or that classify into security/energy/weather, are shown.
`configs/bridge.json` currently points at the existing house on the block (grid-connected, soon
the farm outhouse); its `overrides` are where the SolarEdge/battery sensors should be pinned so
the auto-classifier doesn't have to guess. Run `prompt` and `chat` with `/state` until the
inventory matches how you think about the house.

### 5. Evaluate

```bash
make baseline        # untuned qwen3.5:4b on 60 test examples
make eval            # fine-tuned model on the full test set
```

Every assistant turn in the held-out set is scored with the reference history as context:

- **fetch** turns pass when the model's `get_state` covers every entity the reference needed
  (over-fetch is reported as a ratio, aim for close to 1.0)
- **act** turns pass when the service and the *resolved* entity set match (area_id and
  entity_id targets are treated as equivalent)
- **answer** turns pass when there is no tool call and the required phrases appear

`outputs/eval/<model>/failures.jsonl` holds every miss with the model's output.
That file is the input to the next iteration.

## Corpus, sampling and paraphrases

Rather than regenerating 3,000 examples per run, keep a large corpus and draw training sets from it:

```bash
for s in 100 101 102 103 104 105 106 107 108 109; do            # 30k examples, ten shards (seeds differ from the test set)
  python -m homeai.datagen.build --n 3000 --seed $s --single data/corpus/v1/shard-$s.jsonl
done
python -m homeai.datagen.augment --corpus data/corpus/v1 --bank data/paraphrases/v1.json --out data/corpus/v1-aug --share 0.6
python -m homeai.datagen.sample --corpus data/corpus/v1-aug --n 6000 --out data/generated/train.jsonl \
    --val-out data/generated/val.jsonl --exclude data/generated/test.jsonl --boost clarify=2 confirm_sensitive=2 check_house=1.5
```

`data/paraphrases/v1.json` maps slot-skeleton templates (`turn on the {room} lights`) to spoken paraphrases
written by Claude subagents; the augmenter fills the slots back in, rejects anything that drops a slot or
flips on/off, and then applies the usual voice-style noise. The test set is never sampled from (house
seeds are excluded), so numbers stay comparable across runs.

## The improvement loop

1. Run `make eval`, read `failures.jsonl`, group by `category` and `why`.
2. If a failure is a *data* problem (the reference answer is wrong or awkward), fix the
   scenario in `homeai/datagen/scenarios/`. If it is a *coverage* problem (a phrasing or
   situation the model has never seen), add templates or a new scenario, and raise its weight in
   `homeai/datagen/scenarios/__init__.py`.
3. Talk to the model with `chat --mock` and write down real misses; turn them into scenarios.
4. Regenerate, retrain, re-export, re-eval. Keep the previous `outputs/eval` to compare.

Knobs worth knowing:

- `--n` in `make data`: 3000 is a starting point; 8–10k helps once scenarios are diverse.
- LoRA rank 16 → 32 and 2 → 3 epochs if the model under-fits tool formats; watch val loss.
- Temperature 0.2 and `num_ctx 8192` are set in the Modelfile; 8k context leaves room for
  large `get_state` results on a 10 GB card with a q4_k_m 7B model.
- `homeai/datagen/augment.py` (to add) can paraphrase user utterances with a larger model for more natural variety.

## Deploying the bridge

The bridge has no Home Assistant library dependencies (stdlib only) and runs as its own container,
decoupled from HA's Python environment. `deploy/Dockerfile` and `deploy/compose.homeai-bridge.yaml`
follow the ai-stack conventions on cassowary (sits next to Ollama on the `proxy` network; HA is reached
over the LAN with a long-lived token in `/ai/homeai-bridge/.env`). It could equally be packaged as a
Supervisor add-on on the HA host; the compose route keeps it next to the GPU. The HA side is a thin
custom conversation agent that forwards Assist text plus the satellite's area to `POST /chat`.

## Evaluation sets

**Wild set** (`data/generated/wild.jsonl`, 303 conversations): written by Claude subagents against twelve
houses the training never saw, deliberately off-template (indirect requests, corrections, mixed intents,
pre-confirmed sensitive actions, elimination presence, energy judgement). Build more with:

```bash
python -m homeai.eval.wild_pack --out /tmp/wild --houses 12 --seed 900000      # house packs for the agents
# agents write /tmp/wild/specs/<agent>/house-<seed>.json following the spec inside each pack
python -m homeai.eval.wild_build --packs /tmp/wild --specs /tmp/wild/specs/<agent> --out /tmp/wild/<agent>.jsonl
```

The builder validates every entity id, area, service and answer phrase against the simulated house and
executes the tool calls so tool results are real. Score with `run_eval --data data/generated/wild.jsonl`.
Template-set scores (below) measure generalisation to new houses and states; the wild set measures
generalisation to how people actually talk.


`data/generated/test-v1.jsonl` is the 150-example set every run in the first results page was scored
on (no media/appliance/water/vehicle categories). `data/generated/test-v2.jsonl` (also copied to
`test.jsonl`) is the current generator's set: it adds media (generic and Music Assistant players),
appliances, water/irrigation, vehicle, lock and away-motion categories. Scores across v1 and v2 are not
directly comparable; `scripts/overnight.sh` phase 2 scores the v1 models on v2 as well so the gap is visible.

Training data v3 (`configs/*-v3.json`): the v2 corpus re-paraphrased with 80% of the bank
(`data/paraphrases/v1-train.json`), plus 451 agent-written wild-style training conversations
(`data/generated/wildtrain.jsonl`, houses 800000+, included twice). The held-out 20% of paraphrases
(`v1-held.json`) rewrites test v2 into `test-v2-para.jsonl`, a phrasing test no training run has seen.

Training data v2 (`configs/*-v2.json`) is 5,855 examples sampled from the 30k augmented corpus
(`data/corpus/v2-aug`), 60% of them with paraphrased first utterances, with clarify/confirm/check-house/
presence/media boosted.

## Design B: rooms-only prompt for very large houses

Design A (the default) lists every room's device names in the system prompt; at 10k-20k entities its whole-house
`get_state` results overflow the context. Design B (`homeai/tools_b.py`, `system_prompt_b` in `homeai/house/snapshot.py`)
lists areas with per-class device counts only, and adds `find_devices(areas, classes, name)` (entity ids, no state,
40 rows) and a bounded `get_state` (`name`, `entity_ids`, `where` such as on / open / unlocked / power>100, `sort`
power_desc / value_desc / value_asc / recent, `limit`; cut-off results say how many more and in which rooms).
Examples opt in with `"toolset": "b"`; `e2e`, `run_eval` and the bridge (`--toolset b`) pick prompt, schemas and executor
from it. The same scenario builders write B transcripts (`Conv.toolset = "b"`), with identical requests and answers:

```bash
PYTHONPATH=. python3 -m homeai.datagen.bigbench --toolset b --out data/generated/bigbench-b.jsonl
PYTHONPATH=. python3 -m homeai.datagen.train_b --mix data/generated/train-v5.jsonl --add clarify_in_room=144 act_only_one=150
PYTHONPATH=. python3 -m homeai.datagen.convert_b --data data/generated/test-v2.jsonl --out data/generated/test-v2-b.jsonl
PYTHONPATH=. python3 -m homeai.datagen.convert_b --prompt-only --data data/generated/wild.jsonl --out data/generated/wild-b.jsonl
```

## Wiring into the voice pipeline

Whisper (or HA's Assist STT) produces text; send it to `serve`'s `/chat` with the satellite's
area as `area` and the identified speaker as `speaker`; pipe `reply` to Piper/TTS. Knowing the
satellite's area is what makes "turn on the lights" and the presence elimination logic work.
A thin Home Assistant custom conversation agent that forwards to `/chat` is the natural next
step so this plugs straight into Assist pipelines.

## Roadmap

- [x] first training run + baseline vs fine-tune eval
- [x] `augment.py`: LLM paraphrasing of user utterances
- [x] design B for very large houses; tiered (C/D) and fast+slow (E) designs evaluated
- [x] base-model bake-off (Qwen3.5 0.8B-27B, Gemma 4 E4B, Spark X2.5 4B, LFM2.5 230M-2.6B; 8B-A1B too big to train beside Frigate): LFM2.5 1.2B picked
- [x] speech-to-text bake-off on the heard sets: Whisper medium with the house's names
- [x] separate champions for the Mac mini and the 3090
- [ ] real recordings in the house as a test set
- [ ] phonetic matching of device / room names in the lookup tools; Whisper fine-tuned on the house's names
- [ ] Apple SpeechAnalyzer (custom vocabulary) vs Whisper medium on the Mac
- [ ] pin the existing house's SolarEdge/battery entities in `configs/bridge.json` overrides
- [ ] smoke alarms as a class (8 in the plan) for the away-check
- [ ] Shark logistics: "how charged is the Shark", "is the Shark plugged in", charge-at-old-house planning
- [ ] history tool (`get_history(entity, hours)`) for "when did Amber get home", energy trends
- [ ] HA custom conversation agent wrapping `/chat`
- [ ] proactive mode: scheduled "battery won't last the night" notifications using the same tools
