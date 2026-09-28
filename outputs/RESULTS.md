# home-ai results

Teacher-forced eval on `data/generated/test.jsonl` (see README for scoring). Higher is better; over-fetch closest to 1.0 is best.

| run | examples | turns | pass | act | answer | fetch | over-fetch |
|---|---|---|---|---|---|---|---|
| sanity-homeai-4b (homeai-4b) | 30 | 61 | 96.7% | 93% | 97% | 100% | 1.00 |
| homeai-4b-lora (homeai-4b-lora) | 150 | 326 | 93.3% | 93% | 91% | 99% | 0.99 |
| homeai-9b (homeai-9b) | 150 | 326 | 92.9% | 90% | 92% | 98% | 1.00 |
| homeai-4b (homeai-4b) | 150 | 326 | 90.8% | 90% | 90% | 94% | 1.00 |
| baseline-qwen3.5-9b-nothink (qwen3.5:9b) | 30 | 61 | 68.9% | 60% | 85% | 33% | 0.71 |
| baseline-qwen3.5-9b-full (qwen3.5:9b) | 150 | 326 | 60.7% | 51% | 78% | 31% | 0.65 |
| baseline-qwen3.5-4b-nothink (qwen3.5:4b) | 30 | 61 | 50.8% | 0% | 74% | 50% | 0.72 |
| baseline-qwen3.5-9b (qwen3.5:9b) | 30 | 61 | 50.8% | 13% | 68% | 50% | 0.96 |
| baseline-qwen3.5-4b-full (qwen3.5:4b) | 150 | 326 | 49.7% | 11% | 67% | 48% | 0.86 |

## Per category (pass rate)

| category | sanity-homeai-4b | homeai-4b-lora | homeai-9b | homeai-4b | baseline-qwen3.5-9b-nothink | baseline-qwen3.5-9b-full | baseline-qwen3.5-4b-nothink | baseline-qwen3.5-9b | baseline-qwen3.5-4b-full |
|---|---|---|---|---|---|---|---|---|---|
| check_house | 100% (11) | 88% (41) | 95% (41) | 88% (41) | 82% (11) | 56% (41) | 64% (11) | 73% (11) | 44% (41) |
| clarify | - | 71% (14) | 71% (14) | 57% (14) | - | 50% (14) | - | - | 50% (14) |
| confirm_sensitive | - | 79% (19) | 74% (19) | 79% (19) | - | 63% (19) | - | - | 42% (19) |
| control_all_lights | 100% (2) | 100% (6) | 100% (6) | 100% (6) | 50% (2) | 67% (6) | 50% (2) | 50% (2) | 50% (6) |
| control_area | 100% (6) | 100% (32) | 97% (32) | 100% (32) | 100% (6) | 88% (32) | 50% (6) | 50% (6) | 50% (32) |
| control_brightness | 100% (6) | 100% (8) | 100% (8) | 100% (8) | 83% (6) | 88% (8) | 50% (6) | 50% (6) | 50% (8) |
| control_climate | 50% (2) | 75% (8) | 100% (8) | 75% (8) | 50% (2) | 50% (8) | 50% (2) | 50% (2) | 50% (8) |
| control_compound | 100% (2) | 100% (18) | 100% (18) | 100% (18) | 100% (2) | 89% (18) | 50% (2) | 50% (2) | 50% (18) |
| control_here | 100% (2) | 100% (10) | 100% (10) | 100% (10) | 50% (2) | 70% (10) | 50% (2) | 50% (2) | 50% (10) |
| control_named | 100% (9) | 97% (36) | 94% (36) | 94% (36) | 56% (9) | 53% (36) | 56% (9) | 78% (9) | 72% (36) |
| energy_appliance | - | 100% (12) | 92% (12) | 92% (12) | - | 8% (12) | - | - | 0% (12) |
| energy_night | 100% (3) | 100% (5) | 100% (5) | 100% (5) | 67% (3) | 40% (5) | 67% (3) | 67% (3) | 40% (5) |
| energy_status | - | 100% (12) | 100% (12) | 100% (12) | - | 50% (12) | - | - | 50% (12) |
| leaving | - | 64% (11) | 64% (11) | 64% (11) | - | 27% (11) | - | - | 9% (11) |
| missing | 100% (2) | 100% (5) | 80% (5) | 100% (5) | 50% (2) | 40% (5) | 0% (2) | 0% (2) | 0% (5) |
| presence_in_room | - | 100% (6) | 100% (6) | 100% (6) | - | 83% (6) | - | - | 83% (6) |
| presence_where | 100% (6) | 96% (25) | 92% (25) | 88% (25) | 50% (6) | 52% (25) | 17% (6) | 0% (6) | 44% (25) |
| smalltalk | 100% (1) | 100% (6) | 100% (6) | 100% (6) | 100% (1) | 100% (6) | 100% (1) | 100% (1) | 33% (6) |
| status_aggregate | 83% (6) | 94% (16) | 94% (16) | 94% (16) | 83% (6) | 69% (16) | 83% (6) | 50% (6) | 69% (16) |
| status_entity | - | 100% (24) | 100% (24) | 96% (24) | - | 79% (24) | - | - | 88% (24) |
| weather | - | 100% (6) | 100% (6) | 100% (6) | - | 50% (6) | - | - | 50% (6) |
| who_home | 100% (3) | 100% (6) | 100% (6) | 83% (6) | 0% (3) | 0% (6) | 0% (3) | 0% (3) | 0% (6) |

## Model artifacts


Download from the Mac: `rsync -avP cassowary.dolkens.net:/ai/experiments/home-ai/outputs/<run>/gguf/ ./<run>/` (or `.../lora/` for the adapter, `.../merged16/` for full 16-bit weights)
Run on cassowary: `ollama run <name>` or `python -m homeai.bridge.cli chat --mock --model <name>` from /ai/experiments/home-ai.
