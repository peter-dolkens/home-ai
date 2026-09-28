# Champions per design

| Design | Champion | Score | Speed |
|---|---|---|---|
| A: single model (quality) | homeai-4b-lora-v5 | see CHAMPIONS.md | |
| A: single model (<=1 s/turn) | homeai-4b-lora-v5 | | |
| A: single model (outcome) | homeai-4b-lora-v5 | | |
| A: large houses, 32k window | homeai-4b-lora-v7 | 78.4% bigbench | 1.85 s/turn |
| B: lookup tools | homeai-4b-lora-b | 82.4% (mean of bigbench-b, testv2-b, wild-b, wild2-b) | 1.01 s/turn |
| C/D: tiered pipeline | routed-2b | 78.6% (bigbench, testv2, wild) | p50 3.47 s/turn |
| Room leaf (component) | qwen35-2b-types | 93.9% end state, Bundook leaf test | 463 ms p50 |
| E: fast + slow | - | - final, - flicker | first action - |
