# Heard-set outcome evals

**Caveat on every Piper-voiced number below: much of the clean-audio damage is Piper's pronunciation, not the recogniser.** The same turns voiced by Kokoro-82M (misaki lexicon G2P) through the same engines, clean audio, all four sets (984 turns):

| engine | TTS | WER | house-vocab WER | meaning changed |
|---|---|---|---|---|
| parakeet | piper | 7.4% | 6.7% | 9.0% |
| parakeet | kokoro | 3.4% | 2.7% | 2.9% |
| small | piper | 7.8% | 6.6% | 9.4% |
| small | kokoro | 3.4% | 2.9% | 3.5% |

Of the 39 turns every CPU engine lost with Piper, 32 keep their house terms with Kokoro; what remains is mostly "ensuite" heard as "en suit" / "end suit". Read clean-audio heard drops as an upper bound for real speech; the noise and far-field penalties on top of them are real.

Pass rate on what speech-to-text handed the model (homeai/voice/roundtrip.py), scored against the written reference, next to the same model's written pass rate on the same turns. Cells: **heard pass / drop vs written** (turns). Lost = passed written, failed heard; 'term changed' = the transcript lost a room, device, person, number or command word.

## homeai-2b-lora-v5 @ lapwing

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | 89.7 (174) | 77.8 (409) | 81.2 (344) | | | |
| medium-biased | 83.9 / -5.8 (174) | 73.8 / -3.9 (409) | 71.8 / -9.4 (344) | 75.0 / -6.3 (927) | 72 / 14 | 33% vs 3% |
| parakeet-clean | 81.6 / -8.1 (174) | 72.6 / -5.1 (409) | 73.0 / -8.3 (344) | 74.4 / -6.9 (927) | 85 / 23 | 38% vs 6% |
| small-clean | 84.5 / -5.2 (174) | 70.5 / -7.2 (409) | 72.4 / -8.9 (344) | 73.8 / -7.4 (927) | 85 / 17 | 41% vs 6% |
| small-biased | 84.5 / -5.2 (174) | 71.2 / -6.6 (409) | 70.5 / -10.8 (344) | 73.4 / -7.9 (927) | 89 / 17 | 34% vs 4% |
| tiny-clean | 83.3 / -6.3 (174) | 68.6 / -9.2 (409) | 67.2 / -14.1 (344) | 70.8 / -10.5 (927) | 114 / 19 | 42% vs 10% |
| parakeet-farfield | 76.7 / -12.9 (174) | 65.4 / -12.3 (409) | 67.2 / -14.1 (344) | 68.2 / -13.1 (927) | 145 / 27 | 53% vs 12% |
| parakeet-noisy10 | 69.0 / -20.7 (174) | 62.8 / -14.9 (409) | 61.1 / -20.2 (344) | 63.3 / -18.0 (927) | 187 / 24 | 62% vs 16% |

Confusion suite (confusions.jsonl): pass on the misheard turn; 'source' = the same request written, where it was run (moved-room rows have no written twin).

| group | n | pass | paired n | paired pass | source written pass |
|---|---|---|---|---|---|
| all | 571 | 43.4 | 517 | 42.2 | 79.0 |
| kind:homophone | 41 | 51.2 | 41 | 51.2 | 81.7 |
| kind:number | 28 | 46.4 | 24 | 45.8 | 66.7 |
| kind:observed | 400 | 37.0 | 375 | 36.3 | 80.3 |
| kind:person | 34 | 54.4 | 31 | 56.5 | 66.1 |
| kind:room | 68 | 69.8 | 46 | 70.7 | 81.5 |
| recoverable:False | 10 | 10.0 | 10 | 10.0 | 60.0 |
| recoverable:True | 561 | 44.0 | 507 | 42.8 | 79.4 |

## homeai-4b-lora-b @ lapwing

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | - | - | - | | | |

Confusion suite (confusions.jsonl): pass on the misheard turn; 'source' = the same request written, where it was run (moved-room rows have no written twin).

| group | n | pass | paired n | paired pass | source written pass |
|---|---|---|---|---|---|
| all | 571 | 42.6 | 0 | - | - |
| kind:homophone | 41 | 52.4 | 0 | - | - |
| kind:number | 28 | 50.0 | 0 | - | - |
| kind:observed | 400 | 37.1 | 0 | - | - |
| kind:person | 34 | 54.4 | 0 | - | - |
| kind:room | 68 | 60.3 | 0 | - | - |
| recoverable:False | 10 | 20.0 | 0 | - | - |
| recoverable:True | 561 | 43.0 | 0 | - | - |

## homeai-4b-lora-v5 @ lapwing

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | 93.1 (174) | 82.5 (409) | 86.8 (344) | | | |
| medium-biased | 87.4 / -5.8 (174) | 77.3 / -5.3 (409) | 76.9 / -9.9 (344) | 79.0 / -7.1 (927) | 72 / 5 | 29% vs 3% |
| small-biased | 87.1 / -6.0 (174) | 76.8 / -5.8 (409) | 76.4 / -10.3 (344) | 78.6 / -7.5 (927) | 81 / 10 | 26% vs 5% |
| small-clean | 86.5 / -6.6 (174) | 75.9 / -6.6 (409) | 77.2 / -9.6 (344) | 78.4 / -7.7 (927) | 83 / 10 | 35% vs 7% |
| parakeet-clean | 83.6 / -9.5 (174) | 75.3 / -7.2 (409) | 76.3 / -10.5 (344) | 77.2 / -8.8 (927) | 91 / 8 | 33% vs 6% |
| tiny-clean | 84.5 / -8.6 (174) | 73.1 / -9.4 (409) | 74.0 / -12.8 (344) | 75.6 / -10.5 (927) | 111 / 14 | 41% vs 11% |
| parakeet-farfield | 78.7 / -14.4 (174) | 68.8 / -13.7 (409) | 72.7 / -14.1 (344) | 72.1 / -14.0 (927) | 145 / 17 | 52% vs 12% |
| parakeet-noisy10 | 70.7 / -22.4 (174) | 67.2 / -15.3 (409) | 65.7 / -21.1 (344) | 67.3 / -18.8 (927) | 190 / 17 | 62% vs 16% |

Confusion suite (confusions.jsonl): pass on the misheard turn; 'source' = the same request written, where it was run (moved-room rows have no written twin).

| group | n | pass | paired n | paired pass | source written pass |
|---|---|---|---|---|---|
| all | 571 | 45.6 | 517 | 45.4 | 83.1 |
| kind:homophone | 41 | 54.9 | 41 | 54.9 | 84.2 |
| kind:number | 28 | 60.7 | 24 | 58.3 | 75.0 |
| kind:observed | 400 | 38.8 | 375 | 39.2 | 84.8 |
| kind:person | 34 | 54.4 | 31 | 56.5 | 69.3 |
| kind:room | 68 | 69.8 | 46 | 72.8 | 81.5 |
| recoverable:False | 10 | 20.0 | 10 | 20.0 | 70.0 |
| recoverable:True | 561 | 46.1 | 507 | 45.9 | 83.3 |

## homeai-4b-lora-v5 @ magpie

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | 90.0 (35) | - | - | | | |
| small-biased | 87.1 / -2.9 (35) | - | - | 87.1 / -2.9 (35) | 1 / 0 | 0% vs 9% |
| medium-biased | 84.3 / -5.7 (35) | - | - | 84.3 / -5.7 (35) | 2 / 0 | 50% vs 0% |
| parakeet-farfield | 84.3 / -5.7 (35) | - | - | 84.3 / -5.7 (35) | 3 / 1 | 100% vs 16% |
| parakeet-clean | 81.4 / -8.6 (35) | - | - | 81.4 / -8.6 (35) | 3 / 0 | 67% vs 12% |
| tiny-clean | 81.4 / -8.6 (35) | - | - | 81.4 / -8.6 (35) | 4 / 1 | 75% vs 19% |
| small-clean | 78.6 / -11.4 (35) | - | - | 78.6 / -11.4 (35) | 4 / 0 | 75% vs 10% |
| parakeet-noisy10 | 70.0 / -20.0 (35) | - | - | 70.0 / -20.0 (35) | 7 / 0 | 86% vs 21% |

## homeai-4b-lora-v8 @ lapwing

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | 92.0 (174) | 83.6 (409) | 84.7 (344) | | | |
| medium-biased | 87.4 / -4.6 (174) | 80.4 / -3.2 (409) | 76.4 / -8.3 (344) | 80.3 / -5.3 (927) | 60 / 9 | 35% vs 3% |
| small-biased | 84.8 / -7.2 (174) | 78.4 / -5.3 (409) | 78.5 / -6.2 (344) | 79.6 / -6.0 (927) | 69 / 13 | 29% vs 5% |
| small-clean | 84.2 / -7.8 (174) | 77.9 / -5.8 (409) | 77.0 / -7.7 (344) | 78.8 / -6.9 (927) | 79 / 15 | 38% vs 7% |
| parakeet-clean | 83.3 / -8.6 (174) | 77.9 / -5.8 (409) | 76.4 / -8.3 (344) | 78.4 / -7.2 (927) | 80 / 13 | 37% vs 6% |
| tiny-clean | 83.9 / -8.1 (174) | 76.4 / -7.2 (409) | 71.5 / -13.2 (344) | 76.0 / -9.6 (927) | 103 / 14 | 47% vs 10% |
| parakeet-farfield | 77.0 / -14.9 (174) | 70.7 / -13.0 (409) | 70.8 / -14.0 (344) | 71.9 / -13.7 (927) | 147 / 21 | 55% vs 11% |
| parakeet-noisy10 | 72.7 / -19.2 (174) | 70.0 / -13.6 (409) | 66.1 / -18.6 (344) | 69.1 / -16.5 (927) | 175 / 23 | 66% vs 16% |

Confusion suite (confusions.jsonl): pass on the misheard turn; 'source' = the same request written, where it was run (moved-room rows have no written twin).

| group | n | pass | paired n | paired pass | source written pass |
|---|---|---|---|---|---|
| all | 571 | 44.8 | 517 | 44.9 | 83.5 |
| kind:homophone | 41 | 58.5 | 41 | 58.5 | 78.0 |
| kind:number | 28 | 53.6 | 24 | 50.0 | 79.2 |
| kind:observed | 400 | 38.4 | 375 | 38.5 | 83.2 |
| kind:person | 34 | 55.9 | 31 | 58.1 | 90.3 |
| kind:room | 68 | 65.4 | 46 | 72.8 | 88.0 |
| recoverable:False | 10 | 10.0 | 10 | 10.0 | 80.0 |
| recoverable:True | 561 | 45.5 | 507 | 45.6 | 83.5 |

## homeai-4b-lora-v9 @ lapwing

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | 90.2 (174) | 82.2 (409) | 87.2 (344) | | | |
| medium-biased | 87.9 / -2.3 (174) | 81.0 / -1.1 (409) | 77.9 / -9.3 (344) | 81.2 / -4.4 (927) | 55 / 13 | 33% vs 3% |
| small-biased | 87.9 / -2.3 (174) | 79.1 / -3.1 (409) | 80.1 / -7.1 (344) | 81.1 / -4.4 (927) | 58 / 16 | 33% vs 5% |
| parakeet-clean | 86.2 / -4.0 (174) | 79.6 / -2.6 (409) | 79.7 / -7.6 (344) | 80.8 / -4.7 (927) | 61 / 17 | 39% vs 7% |
| small-clean | 87.4 / -2.9 (174) | 79.2 / -2.9 (409) | 78.5 / -8.7 (344) | 80.5 / -5.1 (927) | 60 / 12 | 40% vs 7% |
| tiny-clean | 86.2 / -4.0 (174) | 77.4 / -4.8 (409) | 76.7 / -10.5 (344) | 78.8 / -6.7 (927) | 82 / 20 | 46% vs 11% |
| parakeet-farfield | 80.5 / -9.8 (174) | 73.7 / -8.4 (409) | 76.0 / -11.2 (344) | 75.8 / -9.7 (927) | 114 / 24 | 59% vs 12% |
| parakeet-noisy10 | 73.3 / -17.0 (174) | 71.6 / -10.5 (409) | 71.4 / -15.8 (344) | 71.9 / -13.7 (927) | 146 / 19 | 66% vs 18% |

Confusion suite (confusions.jsonl): pass on the misheard turn; 'source' = the same request written, where it was run (moved-room rows have no written twin).

| group | n | pass | paired n | paired pass | source written pass |
|---|---|---|---|---|---|
| all | 571 | 57.3 | 517 | 56.3 | 84.5 |
| kind:homophone | 41 | 65.8 | 41 | 65.8 | 82.9 |
| kind:number | 28 | 64.3 | 24 | 62.5 | 91.7 |
| kind:observed | 400 | 50.6 | 375 | 50.5 | 84.4 |
| kind:person | 34 | 79.4 | 31 | 80.7 | 87.1 |
| kind:room | 68 | 77.2 | 46 | 75.0 | 81.5 |
| recoverable:False | 10 | 20.0 | 10 | 20.0 | 80.0 |
| recoverable:True | 561 | 57.9 | 507 | 57.0 | 84.6 |

## homeai-lfm25-1.2b-lora-v5-fix2 @ lapwing

| condition | testv2 | wild | wild2 | all | lost / regained | term changed: lost turns vs rest |
|---|---|---|---|---|---|---|
| written | 89.7 (174) | 74.6 (409) | 76.0 (344) | | | |
| medium-biased | 84.8 / -4.9 (174) | 70.4 / -4.2 (409) | 69.3 / -6.7 (344) | 72.7 / -5.2 (927) | 68 / 16 | 28% vs 3% |
| small-clean | 84.8 / -4.9 (174) | 70.3 / -4.3 (409) | 68.3 / -7.7 (344) | 72.3 / -5.7 (927) | 78 / 24 | 41% vs 6% |
| small-biased | 81.6 / -8.1 (174) | 69.2 / -5.4 (409) | 69.5 / -6.5 (344) | 71.6 / -6.3 (927) | 83 / 23 | 31% vs 4% |
| parakeet-clean | 83.9 / -5.8 (174) | 68.2 / -6.4 (409) | 66.6 / -9.4 (344) | 70.5 / -7.4 (927) | 90 / 20 | 38% vs 6% |
| tiny-clean | 78.2 / -11.5 (174) | 68.0 / -6.6 (409) | 66.6 / -9.4 (344) | 69.4 / -8.6 (927) | 103 / 24 | 35% vs 12% |
| parakeet-farfield | 77.6 / -12.1 (174) | 62.5 / -12.1 (409) | 63.8 / -12.2 (344) | 65.8 / -12.1 (927) | 137 / 24 | 53% vs 12% |
| parakeet-noisy10 | 71.3 / -18.4 (174) | 62.8 / -11.7 (409) | 57.3 / -18.8 (344) | 62.4 / -15.6 (927) | 169 / 23 | 62% vs 17% |

Confusion suite (confusions.jsonl): pass on the misheard turn; 'source' = the same request written, where it was run (moved-room rows have no written twin).

| group | n | pass | paired n | paired pass | source written pass |
|---|---|---|---|---|---|
| all | 571 | 40.9 | 517 | 40.9 | 72.0 |
| kind:homophone | 41 | 46.3 | 41 | 46.3 | 70.7 |
| kind:number | 28 | 53.6 | 24 | 54.2 | 87.5 |
| kind:observed | 400 | 38.8 | 375 | 38.1 | 72.5 |
| kind:person | 34 | 42.6 | 31 | 43.5 | 72.6 |
| kind:room | 68 | 44.1 | 46 | 50.0 | 60.9 |
| recoverable:False | 10 | 20.0 | 10 | 20.0 | 100.0 |
| recoverable:True | 561 | 41.3 | 507 | 41.3 | 71.5 |

## Speed (not comparable across hosts; accuracy runs, not a latency benchmark)

| run | host | hardware | s/turn | s/call |
|---|---|---|---|---|
| homeai-2b-lora-v5-testv2-heard-medium-biased | lapwing | rtx-3080 | 0.99 | 0.47 |
| homeai-2b-lora-v5-testv2-heard-parakeet-clean | lapwing | rtx-3080 | 0.98 | 0.46 |
| homeai-2b-lora-v5-testv2-heard-parakeet-farfield | lapwing | rtx-3080 | 1.01 | 0.46 |
| homeai-2b-lora-v5-testv2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 0.96 | 0.46 |
| homeai-2b-lora-v5-testv2-heard-small-biased | lapwing | rtx-3080 | 0.99 | 0.47 |
| homeai-2b-lora-v5-testv2-heard-small-clean | lapwing | rtx-3080 | 0.99 | 0.47 |
| homeai-2b-lora-v5-testv2-heard-tiny-clean | lapwing | rtx-3080 | 0.99 | 0.46 |
| homeai-2b-lora-v5-wild-heard-medium-biased | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild-heard-parakeet-clean | lapwing | rtx-3080 | 1.0 | 0.45 |
| homeai-2b-lora-v5-wild-heard-parakeet-farfield | lapwing | rtx-3080 | 1.0 | 0.45 |
| homeai-2b-lora-v5-wild-heard-parakeet-noisy10 | lapwing | rtx-3080 | 0.98 | 0.45 |
| homeai-2b-lora-v5-wild-heard-small-biased | lapwing | rtx-3080 | 1.0 | 0.45 |
| homeai-2b-lora-v5-wild-heard-small-clean | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild-heard-tiny-clean | lapwing | rtx-3080 | 1.02 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-medium-biased | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-parakeet-clean | lapwing | rtx-3080 | 0.98 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-parakeet-farfield | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-small-biased | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-small-clean | lapwing | rtx-3080 | 0.98 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-tiny-clean | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-testv2-heard-written | lapwing | rtx-3080 | 0.99 | 0.47 |
| homeai-2b-lora-v5-wild-heard-written | lapwing | rtx-3080 | 0.99 | 0.45 |
| homeai-2b-lora-v5-wild2-heard-written | lapwing | rtx-3080 | 0.97 | 0.45 |
| homeai-4b-lora-v5-testv2-heard-medium-biased | lapwing | rtx-3080 | 1.66 | 0.8 |
| homeai-4b-lora-v5-testv2-heard-parakeet-clean | lapwing | rtx-3080 | 1.65 | 0.8 |
| homeai-4b-lora-v5-testv2-heard-parakeet-farfield | lapwing | rtx-3080 | 1.73 | 0.8 |
| homeai-4b-lora-v5-testv2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.59 | 0.78 |
| homeai-4b-lora-v5-testv2-heard-small-biased | lapwing | rtx-3080 | 1.69 | 0.81 |
| homeai-4b-lora-v5-testv2-heard-small-clean | lapwing | rtx-3080 | 1.66 | 0.8 |
| homeai-4b-lora-v5-testv2-heard-tiny-clean | lapwing | rtx-3080 | 1.64 | 0.79 |
| homeai-4b-lora-v5-wild-heard-medium-biased | lapwing | rtx-3080 | 1.62 | 0.76 |
| homeai-4b-lora-v5-wild-heard-parakeet-clean | lapwing | rtx-3080 | 1.64 | 0.76 |
| homeai-4b-lora-v5-wild-heard-parakeet-farfield | lapwing | rtx-3080 | 1.64 | 0.76 |
| homeai-4b-lora-v5-wild-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.67 | 0.76 |
| homeai-4b-lora-v5-wild-heard-small-biased | lapwing | rtx-3080 | 1.63 | 0.76 |
| homeai-4b-lora-v5-wild-heard-small-clean | lapwing | rtx-3080 | 1.64 | 0.76 |
| homeai-4b-lora-v5-wild-heard-tiny-clean | lapwing | rtx-3080 | 1.66 | 0.76 |
| homeai-4b-lora-v5-wild2-heard-medium-biased | lapwing | rtx-3080 | 1.63 | 0.77 |
| homeai-4b-lora-v5-wild2-heard-parakeet-clean | lapwing | rtx-3080 | 1.65 | 0.77 |
| homeai-4b-lora-v5-wild2-heard-parakeet-farfield | lapwing | rtx-3080 | 1.66 | 0.77 |
| homeai-4b-lora-v5-wild2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.63 | 0.77 |
| homeai-4b-lora-v5-wild2-heard-small-biased | lapwing | rtx-3080 | 1.65 | 0.77 |
| homeai-4b-lora-v5-wild2-heard-small-clean | lapwing | rtx-3080 | 1.63 | 0.77 |
| homeai-4b-lora-v5-wild2-heard-tiny-clean | lapwing | rtx-3080 | 1.68 | 0.78 |
| homeai-4b-lora-v5-testv2-heard-written | lapwing | rtx-3080 | 1.63 | 0.8 |
| homeai-4b-lora-v5-wild-heard-written | lapwing | rtx-3080 | 1.63 | 0.76 |
| homeai-4b-lora-v5-wild2-heard-written | lapwing | rtx-3080 | 1.58 | 0.76 |
| homeai-4b-lora-v5-testv2-heard-medium-biased-s30 | magpie | apple-m2 | 31.89 | 17.17 |
| homeai-4b-lora-v5-testv2-heard-parakeet-clean-s30 | magpie | apple-m2 | 35.17 | 17.34 |
| homeai-4b-lora-v5-testv2-heard-parakeet-farfield-s30 | magpie | apple-m2 | 35.02 | 17.27 |
| homeai-4b-lora-v5-testv2-heard-parakeet-noisy10-s30 | magpie | apple-m2 | 32.13 | 16.79 |
| homeai-4b-lora-v5-testv2-heard-small-biased-s30 | magpie | apple-m2 | 34.0 | 17.5 |
| homeai-4b-lora-v5-testv2-heard-small-clean-s30 | magpie | apple-m2 | 32.02 | 16.73 |
| homeai-4b-lora-v5-testv2-heard-tiny-clean-s30 | magpie | apple-m2 | 33.66 | 17.07 |
| homeai-4b-lora-v5-testv2-heard-written-s30 | magpie | apple-m2 | 42.3 | 22.1 |
| homeai-4b-lora-v8-testv2-heard-medium-biased | lapwing | rtx-3080 | 2.1 | 1.02 |
| homeai-4b-lora-v8-testv2-heard-parakeet-clean | lapwing | rtx-3080 | 2.12 | 1.02 |
| homeai-4b-lora-v8-testv2-heard-parakeet-farfield | lapwing | rtx-3080 | 2.15 | 1.01 |
| homeai-4b-lora-v8-testv2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.57 | 0.78 |
| homeai-4b-lora-v8-testv2-heard-small-biased | lapwing | rtx-3080 | 2.16 | 1.03 |
| homeai-4b-lora-v8-testv2-heard-small-clean | lapwing | rtx-3080 | 2.14 | 1.02 |
| homeai-4b-lora-v8-testv2-heard-tiny-clean | lapwing | rtx-3080 | 2.13 | 1.01 |
| homeai-4b-lora-v8-wild-heard-medium-biased | lapwing | rtx-3080 | 1.58 | 0.75 |
| homeai-4b-lora-v8-wild-heard-parakeet-clean | lapwing | rtx-3080 | 1.59 | 0.75 |
| homeai-4b-lora-v8-wild-heard-parakeet-farfield | lapwing | rtx-3080 | 1.62 | 0.76 |
| homeai-4b-lora-v8-wild-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.61 | 0.76 |
| homeai-4b-lora-v8-wild-heard-small-biased | lapwing | rtx-3080 | 1.59 | 0.76 |
| homeai-4b-lora-v8-wild-heard-small-clean | lapwing | rtx-3080 | 1.6 | 0.76 |
| homeai-4b-lora-v8-wild-heard-tiny-clean | lapwing | rtx-3080 | 1.6 | 0.76 |
| homeai-4b-lora-v8-wild2-heard-medium-biased | lapwing | rtx-3080 | 1.66 | 0.78 |
| homeai-4b-lora-v8-wild2-heard-parakeet-clean | lapwing | rtx-3080 | 1.62 | 0.77 |
| homeai-4b-lora-v8-wild2-heard-parakeet-farfield | lapwing | rtx-3080 | 1.65 | 0.77 |
| homeai-4b-lora-v8-wild2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.61 | 0.77 |
| homeai-4b-lora-v8-wild2-heard-small-biased | lapwing | rtx-3080 | 1.64 | 0.77 |
| homeai-4b-lora-v8-wild2-heard-small-clean | lapwing | rtx-3080 | 1.64 | 0.78 |
| homeai-4b-lora-v8-wild2-heard-tiny-clean | lapwing | rtx-3080 | 1.65 | 0.77 |
| homeai-4b-lora-v8-testv2-heard-written | lapwing | rtx-3080 | 2.1 | 1.03 |
| homeai-4b-lora-v8-wild-heard-written | lapwing | rtx-3080 | 1.77 | 0.85 |
| homeai-4b-lora-v8-wild2-heard-written | lapwing | rtx-3080 | 1.58 | 0.76 |
| homeai-4b-lora-v9-testv2-heard-medium-biased | lapwing | rtx-3080 | 1.62 | 0.8 |
| homeai-4b-lora-v9-testv2-heard-parakeet-clean | lapwing | rtx-3080 | 1.62 | 0.79 |
| homeai-4b-lora-v9-testv2-heard-parakeet-farfield | lapwing | rtx-3080 | 1.64 | 0.79 |
| homeai-4b-lora-v9-testv2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.62 | 0.8 |
| homeai-4b-lora-v9-testv2-heard-small-biased | lapwing | rtx-3080 | 1.64 | 0.8 |
| homeai-4b-lora-v9-testv2-heard-small-clean | lapwing | rtx-3080 | 1.63 | 0.8 |
| homeai-4b-lora-v9-testv2-heard-tiny-clean | lapwing | rtx-3080 | 1.62 | 0.79 |
| homeai-4b-lora-v9-wild-heard-medium-biased | lapwing | rtx-3080 | 1.61 | 0.77 |
| homeai-4b-lora-v9-wild-heard-parakeet-clean | lapwing | rtx-3080 | 1.6 | 0.77 |
| homeai-4b-lora-v9-wild-heard-parakeet-farfield | lapwing | rtx-3080 | 1.59 | 0.76 |
| homeai-4b-lora-v9-wild-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.62 | 0.77 |
| homeai-4b-lora-v9-wild-heard-small-biased | lapwing | rtx-3080 | 1.58 | 0.76 |
| homeai-4b-lora-v9-wild-heard-small-clean | lapwing | rtx-3080 | 1.61 | 0.77 |
| homeai-4b-lora-v9-wild-heard-tiny-clean | lapwing | rtx-3080 | 1.58 | 0.76 |
| homeai-4b-lora-v9-wild2-heard-medium-biased | lapwing | rtx-3080 | 1.57 | 0.76 |
| homeai-4b-lora-v9-wild2-heard-parakeet-clean | lapwing | rtx-3080 | 1.57 | 0.76 |
| homeai-4b-lora-v9-wild2-heard-parakeet-farfield | lapwing | rtx-3080 | 1.59 | 0.77 |
| homeai-4b-lora-v9-wild2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 1.58 | 0.76 |
| homeai-4b-lora-v9-wild2-heard-small-biased | lapwing | rtx-3080 | 1.6 | 0.77 |
| homeai-4b-lora-v9-wild2-heard-small-clean | lapwing | rtx-3080 | 1.58 | 0.77 |
| homeai-4b-lora-v9-wild2-heard-tiny-clean | lapwing | rtx-3080 | 1.6 | 0.77 |
| homeai-4b-lora-v9-testv2-heard-written | lapwing | rtx-3080 | 1.63 | 0.81 |
| homeai-4b-lora-v9-wild-heard-written | lapwing | rtx-3080 | 1.57 | 0.76 |
| homeai-4b-lora-v9-wild2-heard-written | lapwing | rtx-3080 | 1.56 | 0.76 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-medium-biased | lapwing | rtx-3080 | 0.32 | 0.16 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-parakeet-clean | lapwing | rtx-3080 | 0.32 | 0.16 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-parakeet-farfield | lapwing | rtx-3080 | 0.32 | 0.16 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 0.31 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-small-biased | lapwing | rtx-3080 | 0.32 | 0.16 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-small-clean | lapwing | rtx-3080 | 0.32 | 0.16 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-tiny-clean | lapwing | rtx-3080 | 0.32 | 0.16 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-medium-biased | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-parakeet-clean | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-parakeet-farfield | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-parakeet-noisy10 | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-small-biased | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-small-clean | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-tiny-clean | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-medium-biased | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-parakeet-clean | lapwing | rtx-3080 | 0.31 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-parakeet-farfield | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-parakeet-noisy10 | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-small-biased | lapwing | rtx-3080 | 0.31 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-small-clean | lapwing | rtx-3080 | 0.31 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-tiny-clean | lapwing | rtx-3080 | 0.31 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-testv2-heard-written | lapwing | rtx-3080 | 0.34 | 0.17 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild-heard-written | lapwing | rtx-3080 | 0.32 | 0.15 |
| homeai-lfm25-1.2b-lora-v5-fix2-wild2-heard-written | lapwing | rtx-3080 | 0.31 | 0.15 |
