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
