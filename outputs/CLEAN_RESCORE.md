# Outcome evals on uncontaminated turns

Outcome sets test-v2-e2e, wild-e2e and wild2-e2e, rescored per model on the turns whose normalised user text is *not* a user turn of that model's own training data (homeai.datagen.leakage normaliser). **strict**: a turn also counts as contaminated when any earlier turn of its conversation is. Test sets unchanged, no model re-run; built by `python3 -m homeai.eval.clean_rescore` from the read-only copy in `outputs/eval-cassowary`. Champions in bold.

## Summary

- test-v2-e2e: 52%-56% of turns contaminated depending on the training set (52%-56% strict)
- wild-e2e: 11%-16% of turns contaminated depending on the training set (14%-18% strict)
- wild2-e2e: 13%-17% of turns contaminated depending on the training set (16%-21% strict)
- leader (mean of the three sets): all turns homeai-4b-lora-v5, clean homeai-4b-v5, strict homeai-4b-lora-v8
- clean vs all: Spearman 0.97 over 21 models; top 10 in: homeai-4b-v4, out: homeai-9b-v2
- strict-clean vs all: Spearman 0.97 over 21 models; top 10 in: homeai-4b-v4, out: homeai-9b-v2
- clean - all, top 10 on all turns: -0.0 to +2.0 pts (median +0.9)

## Ranking: all turns vs clean turns (mean of the three sets)

| rank all | model | data | all | clean | strict-clean | clean - all (pts) | rank clean | move | rank strict | move |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **homeai-4b-lora-v5** | v5 | 87.2% | 87.7% | 87.5% | +0.6 | 3 | down 2 | 3 | down 2 |
| 2 | **homeai-4b-lora-v8** | v8 | 86.8% | 88.5% | 88.4% | +1.8 | 2 | = | 1 | up 1 |
| 3 | homeai-4b-v5 | v5 | 86.7% | 88.5% | 88.4% | +1.9 | 1 | up 2 | 2 | up 1 |
| 4 | homeai-spark-x25-4b-lora-v5 | v5 | 86.0% | 86.7% | 86.5% | +0.6 | 5 | down 1 | 5 | down 1 |
| 5 | **homeai-4b-lora-v7** | v7 | 85.7% | 87.7% | 87.5% | +2.0 | 4 | up 1 | 4 | up 1 |
| 6 | homeai-4b-lora-v6 | v6 | 84.8% | 86.4% | 86.1% | +1.6 | 6 | = | 6 | = |
| 7 | **homeai-gemma4-e4b-v5** | v5 | 83.8% | 84.6% | 84.5% | +0.9 | 8 | down 1 | 8 | down 1 |
| 8 | **homeai-9b-v2** | v2 | 83.7% | 83.7% | 83.5% | -0.0 | 11 | down 3 | 11 | down 3 |
| 9 | homeai-4b-lora-v2 | v2 | 83.6% | 84.1% | 83.9% | +0.5 | 9 | = | 9 | = |
| 10 | homeai-4b-rft2 | rft(v2) | 83.6% | 83.7% | 83.6% | +0.2 | 10 | = | 10 | = |
| 11 | homeai-4b-v4 | v4 | 83.3% | 84.9% | 84.9% | +1.6 | 7 | up 4 | 7 | up 4 |
| 12 | homeai-4b-v2 | v2 | 82.9% | 82.8% | 82.6% | -0.1 | 14 | down 2 | 14 | down 2 |
| 13 | **homeai-2b-lora-v5** | v5 | 82.9% | 83.1% | 82.9% | +0.2 | 12 | up 1 | 12 | up 1 |
| 14 | homeai-4b-rft1 | rft(v2) | 82.0% | 83.0% | 82.8% | +1.0 | 13 | up 1 | 13 | up 1 |
| 15 | **homeai-lfm25-1.2b-lora-v5-fix2** | v5 | 80.1% | 80.4% | 80.2% | +0.3 | 15 | = | 15 | = |
| 16 | homeai-lfm25-350m-lora-v5-fix | v5 | 70.6% | 67.7% | 67.5% | -2.9 | 16 | = | 16 | = |
| 17 | homeai-gemma4-e4b-v2 | v2 | 49.6% | 45.1% | 45.2% | -4.5 | 17 | = | 17 | = |
| 18 | homeai-lfm25-230m-lora-v5 | v5 | 39.8% | 38.4% | 38.3% | -1.4 | 18 | = | 18 | = |
| 19 | homeai-lfm25-350m-lora-v5 | v5 | 37.0% | 34.3% | 34.1% | -2.8 | 19 | = | 19 | = |
| 20 | homeai-lfm25-1.2b-lora-v5 | v5 | 11.1% | 8.2% | 8.3% | -2.9 | 20 | = | 20 | = |
| 21 | homeai-lfm25-1.2b-lora-v5-fix | v5 | 10.8% | 6.9% | 7.0% | -3.9 | 21 | = | 21 | = |

## Paired check against the all-turns leader (homeai-4b-lora-v5)

Turns the model passes where the leader fails (better) and the reverse (worse); clean = turns clean for both. z = (better - worse) / sqrt(better + worse): |z| < 2 is within noise.

| model | all: better / worse | z | clean: better / worse | z |
|---|---|---|---|---|
| **homeai-4b-lora-v8** | 41 / 45 | -0.43 | 35 / 27 | +1.02 |
| homeai-4b-v5 | 37 / 40 | -0.34 | 33 / 24 | +1.19 |
| homeai-spark-x25-4b-lora-v5 | 43 / 50 | -0.73 | 28 / 36 | -1.00 |
| **homeai-4b-lora-v7** | 27 / 38 | -1.36 | 26 / 25 | +0.14 |
| **homeai-gemma4-e4b-v5** | 60 / 80 | -1.69 | 46 / 65 | -1.80 |
| **homeai-9b-v2** | 30 / 64 | -3.51 | 28 / 51 | -2.59 |
| **homeai-2b-lora-v5** | 27 / 74 | -4.68 | 23 / 58 | -3.89 |
| **homeai-lfm25-1.2b-lora-v5-fix2** | 45 / 115 | -5.53 | 32 / 98 | -5.79 |

## Per set: pass on all / clean / contaminated / strict-clean turns (turn counts in brackets)

| model | set | all | clean | contaminated | strict-clean | clean - all (pts) | source |
|---|---|---|---|---|---|---|---|
| **homeai-4b-lora-v5** | test-v2-e2e | 93.1% (174) | 96.2% (80) | 90.4% (94) | 96.2% (80) | +3.1 | failures.jsonl |
| **homeai-4b-lora-v5** | wild-e2e | 81.9% (409) | 80.8% (352) | 88.6% (57) | 80.6% (343) | -1.1 | failures.jsonl |
| **homeai-4b-lora-v5** | wild2-e2e | 86.5% (344) | 86.1% (299) | 88.9% (45) | 85.7% (290) | -0.4 | failures.jsonl |
| **homeai-4b-lora-v8** | test-v2-e2e | 92.0% (174) | 96.8% (77) | 88.1% (97) | 96.8% (77) | +4.8 | turns.jsonl |
| **homeai-4b-lora-v8** | wild-e2e | 82.8% (409) | 83.2% (351) | 80.2% (58) | 83.0% (341) | +0.4 | turns.jsonl |
| **homeai-4b-lora-v8** | wild2-e2e | 85.6% (344) | 85.7% (297) | 85.1% (47) | 85.5% (286) | +0.1 | turns.jsonl |
| homeai-4b-v5 | test-v2-e2e | 89.7% (174) | 95.6% (80) | 84.6% (94) | 95.6% (80) | +6.0 | failures.jsonl |
| homeai-4b-v5 | wild-e2e | 82.6% (409) | 82.4% (352) | 84.2% (57) | 82.2% (343) | -0.3 | failures.jsonl |
| homeai-4b-v5 | wild2-e2e | 87.8% (344) | 87.6% (299) | 88.9% (45) | 87.2% (290) | -0.2 | failures.jsonl |
| homeai-spark-x25-4b-lora-v5 | test-v2-e2e | 92.5% (174) | 96.2% (80) | 89.4% (94) | 96.2% (80) | +3.7 | turns.jsonl |
| homeai-spark-x25-4b-lora-v5 | wild-e2e | 82.8% (409) | 81.1% (352) | 93.0% (57) | 80.9% (343) | -1.7 | turns.jsonl |
| homeai-spark-x25-4b-lora-v5 | wild2-e2e | 82.8% (344) | 82.6% (299) | 84.4% (45) | 82.4% (290) | -0.2 | turns.jsonl |
| **homeai-4b-lora-v7** | test-v2-e2e | 89.1% (174) | 95.6% (80) | 83.5% (94) | 95.6% (80) | +6.5 | failures.jsonl |
| **homeai-4b-lora-v7** | wild-e2e | 82.0% (409) | 81.4% (352) | 86.0% (57) | 81.2% (343) | -0.6 | failures.jsonl |
| **homeai-4b-lora-v7** | wild2-e2e | 86.0% (344) | 86.1% (299) | 85.6% (45) | 85.7% (290) | +0.1 | failures.jsonl |
| homeai-4b-lora-v6 | test-v2-e2e | 89.4% (174) | 94.4% (80) | 85.1% (94) | 94.4% (80) | +5.0 | failures.jsonl |
| homeai-4b-lora-v6 | wild-e2e | 81.1% (409) | 81.3% (350) | 79.7% (59) | 81.0% (339) | +0.2 | failures.jsonl |
| homeai-4b-lora-v6 | wild2-e2e | 83.9% (344) | 83.6% (298) | 85.9% (46) | 83.0% (288) | -0.3 | failures.jsonl |
| **homeai-gemma4-e4b-v5** | test-v2-e2e | 87.4% (174) | 93.8% (80) | 81.9% (94) | 93.8% (80) | +6.4 | failures.jsonl |
| **homeai-gemma4-e4b-v5** | wild-e2e | 80.2% (409) | 77.8% (352) | 94.7% (57) | 77.6% (343) | -2.4 | failures.jsonl |
| **homeai-gemma4-e4b-v5** | wild2-e2e | 83.7% (344) | 82.3% (299) | 93.3% (45) | 82.1% (290) | -1.4 | failures.jsonl |
| **homeai-9b-v2** | test-v2-e2e | 88.8% (174) | 91.0% (83) | 86.8% (91) | 91.0% (83) | +2.2 | failures.jsonl |
| **homeai-9b-v2** | wild-e2e | 78.9% (409) | 77.8% (362) | 87.2% (47) | 77.5% (353) | -1.1 | failures.jsonl |
| **homeai-9b-v2** | wild2-e2e | 83.4% (344) | 82.3% (300) | 90.9% (44) | 82.0% (289) | -1.1 | failures.jsonl |
| homeai-4b-lora-v2 | test-v2-e2e | 89.1% (174) | 92.2% (83) | 86.3% (91) | 92.2% (83) | +3.1 | failures.jsonl |
| homeai-4b-lora-v2 | wild-e2e | 80.4% (409) | 79.8% (362) | 85.1% (47) | 79.6% (353) | -0.6 | failures.jsonl |
| homeai-4b-lora-v2 | wild2-e2e | 81.4% (344) | 80.3% (300) | 88.6% (44) | 79.9% (289) | -1.1 | failures.jsonl |
| homeai-4b-rft2 | test-v2-e2e | 89.7% (174) | 92.2% (83) | 87.4% (91) | 92.2% (83) | +2.5 | failures.jsonl |
| homeai-4b-rft2 | wild-e2e | 78.4% (409) | 77.8% (358) | 82.4% (51) | 77.6% (351) | -0.6 | failures.jsonl |
| homeai-4b-rft2 | wild2-e2e | 82.7% (344) | 81.2% (296) | 91.7% (48) | 80.9% (286) | -1.5 | failures.jsonl |
| homeai-4b-v4 | test-v2-e2e | 87.4% (174) | 93.2% (81) | 82.3% (93) | 93.2% (81) | +5.9 | failures.jsonl |
| homeai-4b-v4 | wild-e2e | 78.7% (409) | 78.6% (345) | 79.7% (64) | 78.5% (335) | -0.2 | failures.jsonl |
| homeai-4b-v4 | wild2-e2e | 83.9% (344) | 82.9% (284) | 88.3% (60) | 82.9% (272) | -0.9 | failures.jsonl |
| homeai-4b-v2 | test-v2-e2e | 88.8% (174) | 91.6% (83) | 86.3% (91) | 91.6% (83) | +2.8 | failures.jsonl |
| homeai-4b-v2 | wild-e2e | 78.5% (409) | 76.8% (362) | 91.5% (47) | 76.5% (353) | -1.7 | failures.jsonl |
| homeai-4b-v2 | wild2-e2e | 81.5% (344) | 80.2% (300) | 90.9% (44) | 79.8% (289) | -1.4 | failures.jsonl |
| **homeai-2b-lora-v5** | test-v2-e2e | 89.7% (174) | 91.9% (80) | 87.8% (94) | 91.9% (80) | +2.2 | turns.jsonl |
| **homeai-2b-lora-v5** | wild-e2e | 77.8% (409) | 76.7% (352) | 84.2% (57) | 76.4% (343) | -1.0 | turns.jsonl |
| **homeai-2b-lora-v5** | wild2-e2e | 81.2% (344) | 80.8% (299) | 84.4% (45) | 80.5% (290) | -0.5 | turns.jsonl |
| homeai-4b-rft1 | test-v2-e2e | 87.6% (174) | 92.2% (83) | 83.5% (91) | 92.2% (83) | +4.5 | failures.jsonl |
| homeai-4b-rft1 | wild-e2e | 77.8% (409) | 77.4% (358) | 80.4% (51) | 77.2% (351) | -0.4 | failures.jsonl |
| homeai-4b-rft1 | wild2-e2e | 80.7% (344) | 79.4% (296) | 88.5% (48) | 79.0% (286) | -1.3 | failures.jsonl |
| **homeai-lfm25-1.2b-lora-v5-fix2** | test-v2-e2e | 89.7% (174) | 95.0% (80) | 85.1% (94) | 95.0% (80) | +5.3 | turns.jsonl |
| **homeai-lfm25-1.2b-lora-v5-fix2** | wild-e2e | 74.6% (409) | 71.9% (352) | 91.2% (57) | 71.4% (343) | -2.7 | turns.jsonl |
| **homeai-lfm25-1.2b-lora-v5-fix2** | wild2-e2e | 76.0% (344) | 74.2% (299) | 87.8% (45) | 74.1% (290) | -1.8 | turns.jsonl |
| homeai-lfm25-350m-lora-v5-fix | test-v2-e2e | 78.4% (174) | 75.6% (80) | 80.9% (94) | 75.6% (80) | -2.8 | turns.jsonl |
| homeai-lfm25-350m-lora-v5-fix | wild-e2e | 68.1% (409) | 64.8% (352) | 88.6% (57) | 64.1% (343) | -3.3 | turns.jsonl |
| homeai-lfm25-350m-lora-v5-fix | wild2-e2e | 65.1% (344) | 62.7% (299) | 81.1% (45) | 62.6% (290) | -2.4 | turns.jsonl |
| homeai-gemma4-e4b-v2 | test-v2-e2e | 47.1% (174) | 39.8% (83) | 53.8% (91) | 39.8% (83) | -7.4 | failures.jsonl |
| homeai-gemma4-e4b-v2 | wild-e2e | 49.9% (409) | 47.8% (362) | 66.0% (47) | 48.4% (353) | -2.1 | failures.jsonl |
| homeai-gemma4-e4b-v2 | wild2-e2e | 51.7% (344) | 47.7% (300) | 79.5% (44) | 47.4% (289) | -4.1 | failures.jsonl |
| homeai-lfm25-230m-lora-v5 | test-v2-e2e | 40.2% (174) | 42.5% (80) | 38.3% (94) | 42.5% (80) | +2.3 | failures.jsonl |
| homeai-lfm25-230m-lora-v5 | wild-e2e | 39.6% (409) | 35.5% (352) | 64.9% (57) | 35.9% (343) | -4.1 | failures.jsonl |
| homeai-lfm25-230m-lora-v5 | wild2-e2e | 39.5% (344) | 37.1% (299) | 55.6% (45) | 36.6% (290) | -2.4 | failures.jsonl |
| homeai-lfm25-350m-lora-v5 | test-v2-e2e | 35.9% (174) | 33.1% (80) | 38.3% (94) | 33.1% (80) | -2.8 | failures.jsonl |
| homeai-lfm25-350m-lora-v5 | wild-e2e | 39.7% (409) | 36.9% (352) | 57.0% (57) | 36.7% (343) | -2.8 | failures.jsonl |
| homeai-lfm25-350m-lora-v5 | wild2-e2e | 35.5% (344) | 32.8% (299) | 53.3% (45) | 32.4% (290) | -2.7 | failures.jsonl |
| homeai-lfm25-1.2b-lora-v5 | test-v2-e2e | 12.1% (174) | 8.8% (80) | 14.9% (94) | 8.8% (80) | -3.3 | failures.jsonl |
| homeai-lfm25-1.2b-lora-v5 | wild-e2e | 11.0% (409) | 8.8% (352) | 24.6% (57) | 9.0% (343) | -2.2 | failures.jsonl |
| homeai-lfm25-1.2b-lora-v5 | wild2-e2e | 10.2% (344) | 7.0% (299) | 31.1% (45) | 7.2% (290) | -3.2 | failures.jsonl |
| homeai-lfm25-1.2b-lora-v5-fix | test-v2-e2e | 13.8% (174) | 8.8% (80) | 18.1% (94) | 8.8% (80) | -5.0 | turns.jsonl |
| homeai-lfm25-1.2b-lora-v5-fix | wild-e2e | 10.8% (409) | 7.7% (352) | 29.8% (57) | 7.9% (343) | -3.1 | turns.jsonl |
| homeai-lfm25-1.2b-lora-v5-fix | wild2-e2e | 7.8% (344) | 4.3% (299) | 31.1% (45) | 4.5% (290) | -3.5 | turns.jsonl |
| qwen3.5:4b | wild-e2e | 54.9% (409) | 54.9% (409) | - (0) | 54.9% (409) | +0.0 | failures.jsonl |
| gemma4:e4b | wild-e2e | 60.6% (409) | 60.6% (409) | - (0) | 60.6% (409) | +0.0 | failures.jsonl |
| lfm2.5-1.2b | wild-e2e | 35.5% (409) | 35.5% (409) | - (0) | 35.5% (409) | +0.0 | failures.jsonl |
| lfm2.5-350m | wild-e2e | 31.5% (409) | 31.5% (409) | - (0) | 31.5% (409) | +0.0 | failures.jsonl |
| lfm2.5-230m | wild-e2e | 35.5% (409) | 35.5% (409) | - (0) | 35.5% (409) | +0.0 | failures.jsonl |
| spark-x2.5-4b | wild-e2e | 56.6% (409) | 56.6% (409) | - (0) | 56.6% (409) | +0.0 | failures.jsonl |

## Contaminated test turns per training set

Turns whose user text is a training user turn; strict adds every later turn of those conversations.

| training data | test-v2-e2e contaminated | test-v2-e2e strict | wild-e2e contaminated | wild-e2e strict | wild2-e2e contaminated | wild2-e2e strict |
|---|---|---|---|---|---|---|
| train-rft1.jsonl + train.jsonl | 91/174 (52.3%) | 91/174 (52.3%) | 51/409 (12.5%) | 58/409 (14.2%) | 48/344 (14.0%) | 58/344 (16.9%) |
| train-rft2.jsonl + train.jsonl | 91/174 (52.3%) | 91/174 (52.3%) | 51/409 (12.5%) | 58/409 (14.2%) | 48/344 (14.0%) | 58/344 (16.9%) |
| train-v4.jsonl | 93/174 (53.4%) | 93/174 (53.4%) | 64/409 (15.6%) | 74/409 (18.1%) | 60/344 (17.4%) | 72/344 (20.9%) |
| train-v5.jsonl | 94/174 (54.0%) | 94/174 (54.0%) | 57/409 (13.9%) | 66/409 (16.1%) | 45/344 (13.1%) | 54/344 (15.7%) |
| train-v6.jsonl | 94/174 (54.0%) | 94/174 (54.0%) | 59/409 (14.4%) | 70/409 (17.1%) | 46/344 (13.4%) | 56/344 (16.3%) |
| train-v7.jsonl | 94/174 (54.0%) | 94/174 (54.0%) | 57/409 (13.9%) | 66/409 (16.1%) | 45/344 (13.1%) | 54/344 (15.7%) |
| train-v8.jsonl | 97/174 (55.7%) | 97/174 (55.7%) | 58/409 (14.2%) | 68/409 (16.6%) | 47/344 (13.7%) | 58/344 (16.9%) |
| train.jsonl | 91/174 (52.3%) | 91/174 (52.3%) | 47/409 (11.5%) | 56/409 (13.7%) | 44/344 (12.8%) | 55/344 (16.0%) |
