# Champions

Quality = mean pass over test-v2, wild, wild2, wild-e2e, wild2-e2e (para shown, not scored: leaked into v2 training data). Speed = seconds per model turn, idle GPU (limit 1 s).

- **Best overall:** homeai-4b-lora-v5
- **Best at <= 1 s/turn:** homeai-4b-lora-v5
- **Best on outcomes (test-v2-e2e, wild-e2e, wild2-e2e):** homeai-4b-lora-v5

| model | method | data | test-v2 | wild | para | wild2 | wild-e2e | wild2-e2e | test-v2-e2e | rules | quality | outcome | s/turn | s/request |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| homeai-4b-lora-v5 | bf16 LoRA | v5 | 94.4% | 79.2% | 92.6% | 82.5% | 81.9% | 86.5% | 93.1% | 71.9% | 84.9% | 87.2% | 0.82 | 1.70 |
| homeai-4b-lora-v7 | bf16 LoRA | v7 | 92.3% | 78.5% | 90.6% | 83.3% | 82.0% | 86.0% | 89.1% | 82.3% | 84.4% | 85.7% | 0.78 | 1.60 |
| homeai-4b-v5 | QLoRA | v5 | 91.2% | 77.0% | 89.1% | 81.6% | 82.6% | 87.8% | 89.7% | 66.9% | 84.0% | 86.7% | 0.78 | 1.61 |
| homeai-4b-lora-v6 | bf16 LoRA | v6 | 93.2% | 77.8% | 92.6% | 80.6% | 81.1% | 83.9% | 89.4% | 71.4% | 83.3% | 84.8% | 0.77 | 1.63 |
| homeai-4b-lora-v2 | bf16 LoRA | v2 | 93.5% | 77.0% | 93.8% | 81.3% | 80.4% | 81.4% | 89.1% | 73.1% | 82.7% | 83.6% | 0.77 | 1.58 |
| homeai-4b-rft2 | RFT on homeai-4b-lora-v2 (strict facts, confirm sensitive) | rft(v2) | 92.6% | 76.2% | 93.5% | 82.7% | 78.4% | 82.7% | 89.7% | 72.4% | 82.5% | 83.6% | 0.77 | 1.62 |
| homeai-9b-v2 | QLoRA | v2 | 92.2% | 76.3% | 91.7% | 81.4% | 78.9% | 83.4% | 88.8% | 76.6% | 82.4% | 83.7% | 1.00 | 2.12 |
| homeai-4b-v2 | QLoRA | v2 | 92.3% | 75.2% | 91.3% | 79.5% | 78.5% | 81.5% | 88.8% | 72.6% | 81.4% | 82.9% | 0.77 | 1.60 |
| homeai-gemma4-e4b-v5 | QLoRA (Gemma 4 E4B, tool-response mask fixed) | v5 | 85.8% | 75.8% | 84.1% | 80.5% | 80.2% | 83.7% | 87.4% | 74.5% | 81.2% | 83.8% | 0.59 | 1.15 |
| homeai-4b-v4 | QLoRA | v4 | 87.9% | 75.3% | 84.7% | 79.4% | 78.7% | 83.9% | 87.4% | 69.5% | 81.0% | 83.3% | 0.80 | 1.73 |
| homeai-4b-rft1 | RFT on homeai-4b-lora-v2 | rft(v2) | 92.3% | 73.8% | 92.6% | 80.5% | 77.8% | 80.7% | 87.6% | 73.3% | 81.0% | 82.0% | 0.76 | 1.57 |
| homeai-gemma4-e4b-v2 | QLoRA (Gemma 4 E4B) | v2 | 72.3% | 59.5% | 67.8% | 65.1% | 49.9% | 51.7% | 47.1% | 35.5% | 59.7% | 49.6% | 1.05 | 2.56 |
| homeai-lfm25-350m-lora-v5 | bf16 LoRA (LFM2.5 350m) | v5 | 44.0% | 44.4% | 43.1% | 48.0% | 39.7% | 35.5% | 35.9% | 41.4% | 42.3% | 37.0% | 0.12 | 0.25 |
| homeai-lfm25-1.2b-lora-v5 | bf16 LoRA (LFM2.5 1.2b) | v5 | 41.3% | 36.2% | 41.3% | 43.9% | 11.0% | 10.2% | 12.1% | 33.1% | 28.5% | 11.1% | 0.12 | 0.12 |
| qwen3.5:4b | untuned | - | 50.3% | - | - | - | 54.9% | - | - | - | - | - | - | - |
| qwen3.5:9b | untuned | - | 58.0% | - | - | - | - | - | - | - | - | - | - | - |
| gemma4:e4b | untuned | - | 54.6% | - | - | - | 60.6% | - | - | - | - | - | - | - |
| lfm2.5-1.2b | untuned | - | 38.9% | - | - | - | 35.5% | - | - | - | - | - | - | - |
| lfm2.5-350m | untuned | - | - | - | - | - | 31.5% | - | - | - | - | - | - | - |
| lfm2.5-230m | untuned | - | 36.3% | - | - | - | 35.5% | - | - | - | - | - | - | - |
| spark-x2.5-4b | untuned | - | 46.3% | - | - | - | 56.6% | - | - | - | - | - | - | - |
| homeai-4b | QLoRA | v1 | 85.5% | 70.2% | - | - | - | - | - | - | - | - | - | - |
| homeai-9b | QLoRA | v1 | 84.8% | 71.5% | - | - | - | - | - | - | - | - | - | - |
| homeai-4b-lora | bf16 LoRA | v1 | 87.3% | 71.2% | 84.7% | - | - | - | - | - | - | - | - | - |
| homeai-lfm25-230m-lora-v5 | bf16 LoRA (LFM2.5 230m) | v5 | - | - | - | - | 39.6% | 39.5% | 40.2% | - | - | 39.8% | - | - |
| homeai-spark-x25-4b-lora-v5 | bf16 LoRA (Spark X2.5 4B, transformers+PEFT) | v5 | - | - | - | - | - | - | - | - | - | - | - | - |
| homeai-4b-lora-v5-r32 | bf16 LoRA r32 | v5 | - | - | - | - | - | - | - | - | - | - | - | - |
| homeai-4b-lora-b | bf16 LoRA, design B tools | b | - | - | - | - | - | - | - | - | - | - | - | - |
| homeai-4b-lora-orch | bf16 LoRA, design C/D orchestrator | orch | - | - | - | - | - | - | - | - | - | - | - | - |
| homeai-2b-lora-v5 | bf16 LoRA (Qwen3.5-2B, design E fast model) | v5 | - | - | - | - | 77.8% | 81.2% | 89.7% | - | - | 82.9% | - | - |
| homeai-4b-lora-v8 | bf16 LoRA, 8k sequences | v8 | 95.0% | 79.4% | 93.8% | 82.8% | 82.8% | - | - | - | - | - | - | - |
| qwen3.8:27b | untuned (Qwen3.8 27B dense, Aug 2026) | - | - | - | - | - | - | - | - | - | - | - | - | - |
