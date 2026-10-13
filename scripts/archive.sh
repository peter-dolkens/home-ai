#!/usr/bin/env bash
# Off-box model archive on Piper (the Synology NAS, ~7.7 TB free), so cassowary's root disk -- which also holds Frigate's
# database -- never has to hold more than the models in use (Peter, 29 Sep: "move models off ... between runs").
# Everything needed to put a model back into Ollama without retraining: the portable GGUF, the LoRA adapter, the config.
#   scripts/archive.sh push outputs/<run>...        copy to Piper, verify by checksum, mark <run>/.archived
#   scripts/archive.sh drop-gguf outputs/<run>...   delete the local GGUF copies of archived runs (Ollama keeps its own)
#   scripts/archive.sh restore <run> <model> <base-tag> [temperature]   fetch the GGUF back and import it into Ollama
#   scripts/archive.sh list
# Works from cassowary and lapwing (both have key access to piper.dolkens.net). ARCHIVE overrides the destination.
set -uo pipefail
cd "$(dirname "$0")/.."
ARCHIVE=${ARCHIVE:-piper.dolkens.net:home-ai-archive/runs}
HOST=${ARCHIVE%%:*}; ROOT=${ARCHIVE#*:}
ts() { date '+%F %T'; }

push() {  # run dir
  local run=${1%/} name items=()
  name=$(basename "$run")
  for f in gguf lora config.used.json Modelfile; do [ -e "$run/$f" ] && items+=("$run/$f"); done
  [ ${#items[@]} -gt 0 ] || { echo "$(ts) archive $name: nothing to archive"; return 0; }
  # Piper's DSM refuses rsync over ssh for this account, so stream a tar and compare sha256 sums on both ends
  local rel=() i; for i in "${items[@]}"; do rel+=("$(basename "$i")"); done
  ssh -n "$HOST" "mkdir -p $ROOT/$name" || return 1
  tar -C "$run" -cf - "${rel[@]}" | ssh "$HOST" "tar -C $ROOT/$name -xf -" || { echo "$(ts) archive $name: copy FAILED"; return 1; }
  local here there try
  # 6 Oct: both lists are re-sorted in the C locale here. The "checksum mismatches" of 4-6 Oct were only file ORDER: under
  # the driver (no locale) sort put README.md before adapter_*, Piper's shell (a UTF-8 locale) the other way round; checks
  # from a login shell matched because both sides then sorted alike. The re-reads stay as a guard against slow writes.
  here=$(cd "$run" && find "${rel[@]}" -type f -print0 | xargs -0 sha256sum | LC_ALL=C sort)
  for try in 1 2 3 4 5 6 7; do
    there=$(ssh -n "$HOST" "cd $ROOT/$name && find ${rel[*]} -type f -print0 | xargs -0 sha256sum" | LC_ALL=C sort)
    [ "$here" = "$there" ] && break
    echo "$(ts) archive $name: sums differ (try $try): $(diff <(echo "$here") <(echo "$there") | grep '^[<>]' | cut -c1-90 | tr '\n' ' ')"
    [ $try -lt 7 ] && sleep 90
  done
  if [ -n "$here" ] && [ "$here" = "$there" ]; then
    date '+%F %T' > "$run/.archived"; echo "$(ts) archive $name: verified on $ARCHIVE/$name ($(du -sh "$run/gguf" 2>/dev/null | cut -f1) gguf)"
  else
    echo "$(ts) archive $name: checksum mismatch, not marked"; return 1
  fi
}

drop_gguf() {  # run dir -- only once archived; the model itself stays in Ollama
  local run=${1%/}
  [ -f "$run/.archived" ] || { echo "$(ts) drop-gguf $(basename "$run"): not archived, keeping"; return 1; }
  [ -d "$run/gguf" ] || return 0
  local sz; sz=$(du -sh "$run/gguf" | cut -f1)
  rm -rf "$run/gguf" && echo "$(ts) drop-gguf $(basename "$run"): removed local copy ($sz; archived on $HOST)"
}

restore() {  # run model base [temperature]
  local run=outputs/$1 model=$2 base=$3 temp=${4:-0.2}
  mkdir -p "$run/gguf"
  ssh "$HOST" "tar -C $ROOT/$1 -cf - gguf" | tar -C "$run" -xf - || return 1
  python3 -m homeai.train.ollama_import --run "$run" --name "$model" --base "$base" --temperature "$temp" --gguf \
    && rm -f "$run/.pruned" && echo "$(ts) restore $model: imported from $ARCHIVE/$1"
}

case ${1:-} in
  push) shift; for r in "$@"; do push "$r"; done ;;
  drop-gguf) shift; for r in "$@"; do drop_gguf "$r"; done ;;
  restore) shift; restore "$@" ;;
  list) ssh "$HOST" "cd $ROOT 2>/dev/null && du -sh * | sort -k2" ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
