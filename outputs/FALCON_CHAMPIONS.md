# Falcon champions (apple-m2pro)

**All timings on this page were measured on falcon and are only comparable with other falcon runs. Never rank them against cassowary (or any other host): different hardware, different serving stack.** The falcon stream is the low-power option to weigh when choosing the complete architecture.

Accuracy = mean pass over the outcome evals (testv2, wild, wild2; homeai.eval.e2e, served on falcon). Speed = mean seconds per request, cold (homeai.eval.latency --n 60: the first user request of 60 test-v2 conversations through the full tool loop), idle machine, one model loaded. Gaps under 1.5 points are ties; the speed champion is the fastest model within 2 points of the accuracy champion. Champions come from baseline rows (served as cassowary serves them: Ollama 0.21.2, Q4_K_M, same Modelfile and server settings).

- **Accuracy champion:** homeai-4b-lora-v5 (tied within 1.5 pts: homeai-4b-lora-v8)
- **Speed champion (within 2 pts):** homeai-4b-lora-v10

| model | test-v2 | wild | wild2 | accuracy | ref. accuracy¹ | s/turn | s/request | p90 s/request | warm s/request² |
|---|---|---|---|---|---|---|---|---|---|
| homeai-4b-lora-v5 **(acc)** | 92.5% | 82.6% | 86.5% | 87.2% | 87.2% | 5.39 | 11.22 | 15.12 | 11.21 |
| homeai-4b-lora-v8 | 92.2% | 82.6% | 85.3% | 86.7% | 86.8% | 5.67 | 11.56 | 17.26 | 10.83 |
| homeai-4b-lora-v10 **(speed)** | 90.8% | 80.8% | 85.3% | 85.6% | - | 5.35 | 10.94 | 14.48 | 10.97 |
| homeai-4b-lora-v7 | 88.8% | 82.2% | 85.8% | 85.6% | 85.7% | 5.42 | 11.27 | 18.99 | - |
| homeai-4b-lora-v9 | 87.4% | 81.9% | 84.6% | 84.6% | - | - | - | - | - |
| homeai-9b-v2 | 88.5% | 79.5% | 84.4% | 84.1% | 83.7% | 7.98 | 16.99 | 27.98 | - |
| homeai-gemma4-e4b-v8 | 87.9% | 80.2% | 83.6% | 83.9% | - | 2.87 | 5.72 | 8.30 | - |
| homeai-2b-lora-v5 | 87.9% | 77.3% | 80.4% | 81.9% | 82.9% | 2.34 | 4.87 | 6.81 | 5.23 |
| homeai-lfm25-1.2b-lora-v9 | 91.4% | 73.6% | 77.6% | 80.9% | - | 0.96 | 1.95 | 2.67 | - |
| homeai-lfm25-1.2b-lora-v5-fix2 | 89.7% | 75.1% | 76.0% | 80.2% | - | 0.96 | 1.96 | 3.10 | 1.67 |
| homeai-0.8b-lora-v5 | 82.2% | 72.7% | 75.6% | 76.8% | 76.6% | 1.66 | 3.63 | 5.76 | - |
| homeai-lfm25-350m-lora-v5-fix | 77.6% | 68.6% | 65.6% | 70.6% | - | 0.36 | 0.75 | 1.18 | - |

¹ The same weights' outcome score on cassowary: accuracy only, a check that falcon serves the same model. ² Warm prefix cache (homeai.eval.prefix_bench, same house and identical system prompt, later requests of a group).

## Design E pairs (falcon turn logs only)

Fast model acts at once, slow model overrules on disagreement (homeai.eval.reconcile); both halves timed on falcon.

| fast -> slow | final pass | fast pass | flicker (acting) | first action s | final state s | slow alone s |
|---|---|---|---|---|---|---|
| homeai-0.8b-lora-v5 -> homeai-4b-lora-v5 | 85.9% | 75.6% | 15.3% | 9.05 | 9.05 | 13.25 |
| homeai-2b-lora-v5 -> homeai-4b-lora-v5 | 85.9% | 80.4% | 17.6% | 9.62 | 9.65 | 13.23 |
| homeai-gemma4-e4b-v8 -> homeai-4b-lora-v5 | 85.9% | 82.9% | 13.1% | 8.62 | 8.62 | 13.46 |
| homeai-lfm25-1.2b-lora-v5-fix2 -> homeai-4b-lora-v5 | 85.9% | 78.2% | 15.1% | 8.30 | 8.30 | 13.22 |
| homeai-lfm25-1.2b-lora-v9 -> homeai-4b-lora-v5 | 85.9% | 78.4% | 15.7% | 8.35 | 8.35 | 13.30 |
| homeai-lfm25-350m-lora-v5-fix -> homeai-4b-lora-v5 | 85.9% | 69.2% | 16.9% | 9.13 | 9.13 | 13.34 |
| homeai-0.8b-lora-v5 -> homeai-4b-lora-v8 | 85.4% | 75.6% | 16.8% | 9.37 | 9.37 | 13.26 |
| homeai-2b-lora-v5 -> homeai-4b-lora-v8 | 85.4% | 80.4% | 18.1% | 9.87 | 9.90 | 13.27 |
| homeai-4b-lora-v5 -> homeai-4b-lora-v8 | 85.4% | 85.9% | 11.7% | 13.40 | 13.78 | 13.28 |
| homeai-4b-lora-v7 -> homeai-4b-lora-v8 | 85.4% | 84.8% | 12.0% | 13.39 | 13.79 | 13.35 |
| homeai-4b-lora-v9 -> homeai-4b-lora-v8 | 85.4% | 83.9% | 14.8% | 13.59 | 14.04 | 13.32 |
| homeai-gemma4-e4b-v8 -> homeai-4b-lora-v8 | 85.4% | 82.9% | 16.1% | 9.09 | 9.09 | 13.42 |
| homeai-lfm25-1.2b-lora-v5-fix2 -> homeai-4b-lora-v8 | 85.4% | 78.2% | 17.2% | 8.69 | 8.69 | 13.19 |
| homeai-lfm25-1.2b-lora-v9 -> homeai-4b-lora-v8 | 85.4% | 78.4% | 16.8% | 8.59 | 8.59 | 13.26 |
| homeai-lfm25-350m-lora-v5-fix -> homeai-4b-lora-v8 | 85.4% | 69.2% | 17.2% | 9.31 | 9.31 | 13.30 |

## Optimisations (falcon phase 2)

Each variant against its own falcon baseline (same weights unless the variant is a different quantisation or server). Deltas: accuracy in points, times in seconds (negative = faster).

| variant | setup | accuracy | Δ acc (pts) | s/request | Δ | warm s/request | Δ | s/turn | Δ |
|---|---|---|---|---|---|---|---|---|---|
| homeai-4b-lora-v10@falcon-mlx4 | mlx-lm==0.31.3 server, 4-bit g64, prompt cache on, stable_prefix prompts | 87.3% | +1.6 | 6.55 | -4.38 | 2.42 | -8.55 | 3.40 | -1.95 |
| homeai-4b-lora-v5@falcon-ctx32k | Ollama 0.21.2 stock, num_ctx 32768 (vs 8192) | - | - | 11.22 | +0.00 | 11.23 | +0.02 | 5.39 | +0.00 |
| homeai-4b-lora-v5@falcon-iq4_xs | Ollama 0.21.2 stock, IQ4_XS (llama.cpp b11243, no imatrix) | 86.5% | -0.7 | 11.00 | -0.22 | 11.53 | +0.32 | 5.35 | -0.04 |
| homeai-4b-lora-v5@falcon-kvq8 | Ollama 0.21.2 stock (flash attention on by default) + q8_0 KV cache | - | - | 11.63 | +0.42 | 14.29 | +3.08 | 5.59 | +0.20 |
| homeai-4b-lora-v5@falcon-mlx4 | mlx-lm==0.31.3 server, 4-bit g64, prompt cache on | 87.5% | +0.3 | 6.52 | -4.70 | 2.51 | -8.69 | 3.38 | -2.01 |
| homeai-4b-lora-v5@falcon-mlx4g32 | mlx-lm==0.31.3 server, 4-bit g32, prompt cache on | 86.9% | -0.4 | 6.83 | -4.38 | 2.65 | -8.56 | 3.46 | -1.93 |
| homeai-4b-lora-v5@falcon-nb1024 | Ollama 0.21.2 stock, PARAMETER num_batch 1024 | - | - | 11.20 | -0.02 | 7.99 | -3.22 | 5.47 | +0.08 |
| homeai-4b-lora-v5@falcon-nb2048 | Ollama 0.21.2 stock, PARAMETER num_batch 2048 | - | - | 14.38 | +3.17 | 14.73 | +3.52 | 6.97 | +1.58 |
| homeai-4b-lora-v5@falcon-ollama035 | Ollama 0.35.0 (llama-server backend), cassowary settings (KEEP_ALIVE=-1, NUM_PARALLEL=2), Q4_K_M, num_ctx 8192 | 87.0% | -0.2 | 5.47 | -5.74 | 4.67 | -6.53 | 2.71 | -2.68 |
| homeai-4b-lora-v5@falcon-q4_0 | Ollama 0.21.2 stock, Q4_0 (llama.cpp b11243) | 87.6% | +0.3 | 10.48 | -0.74 | 10.36 | -0.85 | 5.06 | -0.33 |
| homeai-4b-lora-v5@falcon-q4_k_m-rebuilt | Ollama 0.21.2 stock, Q4_K_M rebuilt on frogmouth from the fused weights (control for the other quants) | 87.4% | +0.2 | 11.22 | +0.00 | 11.23 | +0.03 | 5.39 | +0.00 |
| homeai-4b-lora-v5@falcon-q5_k_m | Ollama 0.21.2 stock, Q5_K_M (llama.cpp b11243) | 88.1% | +0.9 | 12.05 | +0.84 | 12.35 | +1.15 | 5.87 | +0.48 |
| homeai-4b-lora-v9@falcon-mlx4 | mlx-lm==0.31.3 server, 4-bit g64, prompt cache on, classic prompts | 87.5% | +2.9 | 6.55 | - | 2.52 | - | 3.39 | - |
| homeai-4b-lora-v9@falcon-ollama035 | Ollama 0.35.0 (llama-server backend), cassowary settings (KEEP_ALIVE=-1, NUM_PARALLEL=2), Q4_K_M, num_ctx 8192 | 84.4% | -0.2 | 5.48 | - | 4.63 | - | 2.72 | - |
| homeai-lfm25-1.2b-lora-v9@falcon-ollama035 | Ollama 0.35.0 (llama-server backend), cassowary settings (KEEP_ALIVE=-1, NUM_PARALLEL=2), Q4_K_M, num_ctx 8192 | 81.8% | +1.0 | 1.38 | -0.57 | 1.06 | - | 0.70 | -0.26 |
| homeai-2b-lora-b-v2@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, stable_prefix prompts | - | - | - | - | - | - | - | - |
| homeai-2b-lora-b-v2u-r32@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-2b-lora-b-v2u@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-2b-lora-b-v3h@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-2b-lora-b-v6n@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-4b-lora-b-v2u@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-4b-lora-b-v4w@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-4b-lora-v11@falcon-mlx4 | mlx-lm==0.31.3 server, 4-bit g64, prompt cache on, user_context prompts | 85.5% | - | 5.70 | - | 2.62 | - | 2.85 | - |
| homeai-4b-lora-v11@falcon-mlx4-bundook | mlx-lm==0.31.3 server, 4-bit g64, prompt cache on, user_context prompts, Bundook plan by house, num_ctx 32768 | - | - | - | - | - | - | - | - |
| homeai-4b-lora-v12@falcon-mlx4 | mlx-lm==0.31.3 server, 4-bit g64, prompt cache on, clock_in_user prompts | 86.9% | - | 6.53 | - | 2.32 | - | 3.38 | - |
| homeai-4b-lora-v12@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, clock_in_user prompts | 87.4% | - | 7.19 | - | 3.04 | - | 3.66 | - |
| homeai-lfm25-1.2b-lora-b-v2u@falcon-mlx8 | mlx-lm==0.31.3 server, 8-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v3h@falcon-mlx8 | mlx-lm==0.31.3 server, 8-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v7@falcon-mlx8 | mlx-lm==0.31.3 server, 8-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v7r128@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v7r256@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v7r32@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v7r32@falcon-mlx8 | mlx-lm==0.31.3 server, 8-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v7r64@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-1.2b-lora-b-v8r256e4@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-2.6b-lora-b-v8r128e4@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-230m-lora-b-v8r256e4@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-230m-lora-b-v8r256e4@falcon-mlx8 | mlx-lm==0.31.3 server, 8-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-230m-lora-b-v8r256e6@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-350m-lora-b-v8r256e4@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-350m-lora-b-v8r256e4@falcon-mlx8 | mlx-lm==0.31.3 server, 8-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-350m-lora-b-v8r256e4s2@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |
| homeai-lfm25-350m-lora-b-v8r256e6@falcon-mlx6 | mlx-lm==0.31.3 server, 6-bit g64, prompt cache on, design B tools, user_context prompts | - | - | - | - | - | - | - | - |

### Warm prefix cache

Requests grouped by house; the first of a group is cold (another house was last), the rest warm. prompt = identical system prompt; house = same house but the Now / Speaker / People lines differ (today's deployment: they sit above the inventory); header-last = those lines moved below the rules (timing only). Tokens = prompt tokens the server actually evaluated on a request's first call (a cache hit shows as a small number).

| row | mode | cold s/request | warm s/request | cold tokens evaluated | warm tokens evaluated | prompt tokens |
|---|---|---|---|---|---|---|
| homeai-2b-lora-b-v2@falcon-mlx6 | house | 3.49 | 4.36 | 2548.00 | 2553.95 | 2553.17 |
| homeai-2b-lora-b-v2u-r32@falcon-mlx6 | house | 2.65 | 1.85 | 1646.67 | 54.70 | 2552.17 |
| homeai-2b-lora-b-v2u@falcon-mlx6 | house | 2.56 | 1.77 | 1646.67 | 54.70 | 2552.17 |
| homeai-2b-lora-b-v3h@falcon-mlx6 | house | 2.58 | 1.78 | 1646.67 | 54.70 | 2552.17 |
| homeai-2b-lora-b-v6n@falcon-mlx6 | house | 2.63 | 1.97 | 1646.67 | 54.70 | 2552.17 |
| homeai-2b-lora-v5 | prompt | 4.83 | 5.23 | 1450.17 | 1449.70 | 1450.17 |
| homeai-2b-lora-v5 | house | 4.83 | 5.23 | 1450.17 | 1449.70 | 1450.17 |
| homeai-2b-lora-v5 | header-last | 4.82 | 4.91 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-b-v2u@falcon-mlx6 | house | 6.02 | 3.68 | 1646.67 | 54.70 | 2552.17 |
| homeai-4b-lora-b-v4w@falcon-mlx6 | house | 5.95 | 3.77 | 1646.67 | 54.70 | 2552.17 |
| homeai-4b-lora-v10 | prompt | 11.14 | 10.97 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v10 | house | 11.14 | 10.97 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v10 | header-last | 11.15 | 10.97 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v10@falcon-mlx4 | prompt | 5.82 | 2.42 | 1454.17 | 18.53 | 1453.78 |
| homeai-4b-lora-v10@falcon-mlx4 | house | 5.79 | 6.04 | 1454.17 | 1453.70 | 1453.78 |
| homeai-4b-lora-v10@falcon-mlx4 | header-last | 5.80 | 6.03 | 1454.17 | 1453.70 | 1453.78 |
| homeai-4b-lora-v11@falcon-mlx4 | prompt | 5.81 | 2.62 | 1453.17 | 55.37 | 1452.78 |
| homeai-4b-lora-v11@falcon-mlx4 | house | 5.81 | 2.62 | 1453.17 | 55.37 | 1452.78 |
| homeai-4b-lora-v11@falcon-mlx4 | header-last | 5.81 | 6.08 | 1454.17 | 1453.70 | 1453.78 |
| homeai-4b-lora-v11@falcon-mlx4-bundook | house | 5.75 | 7.88 | 1680.67 | 54.70 | 2665.78 |
| homeai-4b-lora-v12@falcon-mlx4 | prompt | 5.78 | 2.32 | 1450.17 | 28.53 | 1449.78 |
| homeai-4b-lora-v12@falcon-mlx4 | house | 5.78 | 2.32 | 1450.17 | 28.53 | 1449.78 |
| homeai-4b-lora-v12@falcon-mlx4 | header-last | 5.79 | 5.99 | 1454.17 | 1453.70 | 1453.78 |
| homeai-4b-lora-v12@falcon-mlx6 | prompt | 6.35 | 3.04 | 1450.17 | 28.53 | 1449.78 |
| homeai-4b-lora-v12@falcon-mlx6 | house | 6.34 | 3.10 | 1450.17 | 28.53 | 1449.78 |
| homeai-4b-lora-v12@falcon-mlx6 | header-last | 6.35 | 6.72 | 1454.17 | 1453.70 | 1453.78 |
| homeai-4b-lora-v5 | prompt | 11.11 | 11.21 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5 | house | 11.11 | 11.22 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5 | header-last | 11.13 | 11.45 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v5@falcon-ctx32k | prompt | 11.13 | 11.23 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-ctx32k | house | 11.13 | 11.24 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-ctx32k | header-last | 11.12 | 11.43 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-iq4_xs | prompt | 11.05 | 11.53 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-iq4_xs | house | 11.04 | 11.52 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-iq4_xs | header-last | 11.07 | 11.60 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v5@falcon-kvq8 | prompt | 14.11 | 14.29 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-kvq8 | house | 11.91 | 11.82 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-kvq8 | header-last | 11.43 | 11.78 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-mlx4 | prompt | 5.79 | 2.51 | 1450.17 | 18.53 | 1449.78 |
| homeai-4b-lora-v5@falcon-mlx4 | house | 5.79 | 6.13 | 1450.17 | 1449.70 | 1449.78 |
| homeai-4b-lora-v5@falcon-mlx4 | header-last | 5.79 | 6.14 | 1450.17 | 1449.70 | 1449.78 |
| homeai-4b-lora-v5@falcon-mlx4g32 | prompt | 5.98 | 2.65 | 1450.17 | 18.53 | 1449.78 |
| homeai-4b-lora-v5@falcon-mlx4g32 | house | 6.18 | 6.52 | 1450.17 | 1449.70 | 1449.78 |
| homeai-4b-lora-v5@falcon-mlx4g32 | header-last | 6.29 | 6.61 | 1450.17 | 1449.70 | 1449.78 |
| homeai-4b-lora-v5@falcon-nb1024 | prompt | 10.65 | 7.99 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-nb1024 | house | 10.65 | 10.96 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-nb1024 | header-last | 10.65 | 8.14 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-nb2048 | prompt | 13.81 | 14.73 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-nb2048 | house | 13.81 | 14.74 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-nb2048 | header-last | 13.81 | 14.99 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-ollama035 | prompt | 4.78 | 4.67 | 1449.17 | 1448.70 | 1449.17 |
| homeai-4b-lora-v5@falcon-ollama035 | house | 4.74 | 5.12 | 1449.17 | 1448.70 | 1449.17 |
| homeai-4b-lora-v5@falcon-ollama035 | header-last | 4.53 | 4.75 | 1453.17 | 1452.70 | 1453.17 |
| homeai-4b-lora-v5@falcon-q4_0 | prompt | 10.37 | 10.36 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q4_0 | house | 10.37 | 10.36 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q4_0 | header-last | 10.38 | 10.55 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q4_k_m-rebuilt | prompt | 11.12 | 11.23 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q4_k_m-rebuilt | house | 11.13 | 11.24 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q4_k_m-rebuilt | header-last | 11.12 | 11.42 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q5_k_m | prompt | 12.18 | 12.35 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q5_k_m | house | 12.17 | 12.35 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v5@falcon-q5_k_m | header-last | 12.22 | 12.29 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v8 | prompt | 11.10 | 10.83 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v8 | house | 11.10 | 10.84 | 1450.17 | 1449.70 | 1450.17 |
| homeai-4b-lora-v8 | header-last | 11.13 | 10.93 | 1454.17 | 1453.70 | 1454.17 |
| homeai-4b-lora-v9@falcon-mlx4 | prompt | 6.01 | 2.52 | 1450.17 | 18.53 | 1449.78 |
| homeai-4b-lora-v9@falcon-mlx4 | house | 6.02 | 6.13 | 1450.17 | 1449.70 | 1449.78 |
| homeai-4b-lora-v9@falcon-mlx4 | header-last | 6.01 | 6.18 | 1454.17 | 1453.70 | 1453.78 |
| homeai-4b-lora-v9@falcon-ollama035 | prompt | 4.76 | 4.63 | 1449.17 | 1448.70 | 1449.17 |
| homeai-4b-lora-v9@falcon-ollama035 | house | 4.70 | 5.05 | 1449.17 | 1448.70 | 1449.17 |
| homeai-4b-lora-v9@falcon-ollama035 | header-last | 4.50 | 4.64 | 1453.17 | 1452.70 | 1453.17 |
| homeai-lfm25-1.2b-lora-b-v2u@falcon-mlx8 | house | 1.50 | 0.77 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v3h@falcon-mlx8 | house | 1.51 | 0.80 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v7@falcon-mlx8 | house | 1.51 | 0.80 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v7r128@falcon-mlx6 | house | 1.51 | 0.74 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v7r256@falcon-mlx6 | house | 1.51 | 0.75 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v7r32@falcon-mlx6 | house | 1.51 | 0.74 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v7r32@falcon-mlx8 | house | 1.51 | 0.81 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v7r64@falcon-mlx6 | house | 1.51 | 0.74 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-b-v8r256e4@falcon-mlx6 | house | 1.52 | 0.73 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-1.2b-lora-v5-fix2 | prompt | 1.92 | 1.67 | 1324.00 | 1323.67 | 1324.00 |
| homeai-lfm25-1.2b-lora-v5-fix2 | house | 1.92 | 1.94 | 1324.00 | 1323.67 | 1324.00 |
| homeai-lfm25-1.2b-lora-v5-fix2 | header-last | 1.83 | 1.61 | 1324.00 | 1323.67 | 1324.00 |
| homeai-lfm25-1.2b-lora-v9@falcon-ollama035 | prompt | 1.16 | 1.06 | 1324.00 | 1323.67 | 1324.00 |
| homeai-lfm25-1.2b-lora-v9@falcon-ollama035 | house | 1.13 | 1.22 | 1324.00 | 1323.67 | 1324.00 |
| homeai-lfm25-1.2b-lora-v9@falcon-ollama035 | header-last | 1.04 | 1.05 | 1327.00 | 1326.67 | 1327.00 |
| homeai-lfm25-2.6b-lora-b-v8r128e4@falcon-mlx6 | house | 3.87 | 2.60 | 1561.67 | 50.20 | 2426.96 |
| homeai-lfm25-230m-lora-b-v8r256e4@falcon-mlx6 | house | 0.34 | 0.24 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-230m-lora-b-v8r256e4@falcon-mlx8 | house | 0.34 | 0.25 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-230m-lora-b-v8r256e6@falcon-mlx6 | house | 0.34 | 0.24 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-350m-lora-b-v8r256e4@falcon-mlx6 | house | 0.51 | 0.32 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-350m-lora-b-v8r256e4@falcon-mlx8 | house | 0.51 | 0.34 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-350m-lora-b-v8r256e4s2@falcon-mlx6 | house | 0.51 | 0.31 | 1592.67 | 49.20 | 2473.96 |
| homeai-lfm25-350m-lora-b-v8r256e6@falcon-mlx6 | house | 0.51 | 0.31 | 1592.67 | 49.20 | 2473.96 |

With optimisations allowed: accuracy homeai-4b-lora-v5@falcon-q5_k_m, speed homeai-4b-lora-v5@falcon-ollama035.

### Findings so far (preliminary: phase 2 still running)

- **Flash attention is already on.** Ollama 0.21.2 turns it on by default on Metal: the log says "enabling flash attention" and the load request has FlashAttention:Enabled. So OLLAMA_FLASH_ATTENTION=1 is the baseline, and the stream doesn't rerun it.
- **Ollama doesn't reuse the prompt prefix for Qwen3.5.** This is a hybrid Gated DeltaNet model, so its recurrent state can't be rolled back to a shared prefix.
  - On falcon, v5's latency bench gave warm 11.21 s/request against cold 11.22 s, even though "warm" repeats the identical request.
  - On frogmouth, the prefix bench showed the full ~1,560-token prompt evaluated on every call. That includes the second call of a request, which only extends the first.
  - As a result, every model call pays the whole system prompt.
- **Something volatile sits before the static prompt.** homeai/house/snapshot.py puts the Now (clock minute), Speaker and People lines on lines 3–5 of the system prompt, above the inventory and rules. The bridge rebuilds the prompt every round (homeai/bridge/agent.py Agent.chat).
- **mlx_lm.server (0.31.3) does reuse the prefix, but only at the end of the system message.** On frogmouth with the 0.8B, identical system prompts evaluated 19.5 of ~1,560 tokens warm, and later calls in a request 84 tokens.
  - Changing the clock minute ("house" mode) brings back the full evaluation.
  - So does moving those lines to the end of the system message, because the checkpoint sits at the system-message boundary.
  - To benefit in deployment, the volatile lines would have to leave the system message, for example as a prefix of the user turn. That needs an accuracy check, and probably a retrain.
- **MLX accuracy depends on the group size.** 0.8B v5 on 40 test-v2 conversations:

  | serving | accuracy |
  |---|---|
  | Ollama Q4_K_M | 84.0% |
  | MLX bf16 | 84.0% |
  | MLX 8-bit | 84.0% |
  | MLX 4-bit g32 | 81.9% |
  | MLX 4-bit g64 | 72.3% |
  | MLX mixed 4/6-bit g64 | 73.4% |

  The tool-call parsing and chat template match, since bf16 equals Ollama exactly. The 4B v5 at 4-bit g32 scored 91.5% on the same 40 rows (frogmouth). Falcon measures both g64 and g32 on the full sets.
- **The requantised v5 is not bit-identical to the archive.** Fusing the archived PEFT adapter on the Mac and running Ollama's Q4_K_M does not reproduce the archived GGUF bit for bit. Every quantised tensor differs slightly, including untouched ones such as token_embd, so Unsloth's merge rounds differently. The Q4_0, IQ4_XS and Q5_K_M variants are therefore compared against a rebuilt Q4_K_M control, not only the archived baseline.

