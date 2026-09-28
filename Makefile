PYTHON ?= python3
N ?= 3000
CONFIG ?= configs/qwen35-4b.json
MODEL ?= homeai
BASELINE ?= qwen3.5:4b

data:            ## generate the synthetic dataset into data/generated
	$(PYTHON) -m homeai.datagen.build --n $(N) --out data/generated

dry-run:         ## render with the HF tokenizer and print token stats (no GPU)
	$(PYTHON) -m homeai.train.train --config $(CONFIG) --dry-run

train:           ## QLoRA fine-tune on the GPU box
	$(PYTHON) -m homeai.train.train --config $(CONFIG)

export:          ## merge LoRA -> 16-bit safetensors -> Ollama-native import (run on cassowary)
	scripts/train-docker.sh merge $(CONFIG) && $(PYTHON) -m homeai.train.ollama_import --run $$($(PYTHON) -c "import json;print(json.load(open('$(CONFIG)'))['out'])") --name $(MODEL) --base qwen3.5:4b

eval:            ## teacher-forced eval of the fine-tuned model
	$(PYTHON) -m homeai.eval.run_eval --model $(MODEL) --data data/generated/test.jsonl

baseline:        ## same eval against the untuned base model
	$(PYTHON) -m homeai.eval.run_eval --model $(BASELINE) --data data/generated/test.jsonl --limit 60

chat:            ## talk to the model against a mock house
	$(PYTHON) -m homeai.bridge.cli chat --mock --model $(MODEL)

test:            ## offline checks (no GPU, no HA, no Ollama)
	$(PYTHON) tests/test_eval_scoring.py && $(PYTHON) tests/test_bridge.py && $(PYTHON) tests/test_house_stability.py && $(PYTHON) tests/test_tools_b.py && $(PYTHON) tests/test_leaf.py && $(PYTHON) tests/test_tiered.py && $(PYTHON) tests/test_voice.py && $(PYTHON) tests/test_train_mlx.py

voice:           ## TTS -> STT round trip in Docker, e.g. make voice ARGS="--sets probe --stt base,parakeet --conditions clean,noisy10"
	scripts/voice-roundtrip.sh $(ARGS)

.PHONY: data dry-run train export eval baseline chat test voice
