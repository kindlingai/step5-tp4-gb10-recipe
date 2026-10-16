#!/usr/bin/env bash
# Source in an existing Bash boot wrapper. This function prints shell-quoted
# environment assignments; it does not copy files, contact nodes, or boot them.
fp8kv_remote_env() {
  local key
  for key in STEP5_FP8KV FP8KV_DIR FP8KV_BLOCKS FP8KV_EXTRA_PYTHONPATH              KV_DTYPE MAXLEN GMU BATCH_TOKENS MAXSEQ STEP5_DENSE LMONLY SPEC_TOKENS; do
    if [[ -v $key ]]; then printf '%s=%q ' "$key" "${!key}"; fi
  done
}
