#!/usr/bin/env bash
# Source after the base launcher configuration, before docker run.
FP8KV_ARGS=()
FP8KV_VLLM_ARGS=()
if [[ ${STEP5_FP8KV:-0} == 1 ]]; then
  : "${FP8KV_DIR:?Set FP8KV_DIR to the absolute overlay directory on this node}"
  : "${FP8KV_BLOCKS:?Set the profiled block cap, including scratch headroom}"
  [[ $FP8KV_DIR == /* && -d $FP8KV_DIR/step5_fp8kv ]] || {
    echo 'The absolute overlay directory is missing' >&2; exit 2;
  }
  [[ $FP8KV_BLOCKS =~ ^[1-9][0-9]*$ ]] || {
    echo 'FP8KV_BLOCKS must be a positive integer' >&2; exit 2;
  }
  [[ ${KV_DTYPE:-auto} == fp8_e4m3 && ${STEP5_DENSE:-1} == 1 ]] || {
    echo 'This integration requires KV_DTYPE=fp8_e4m3 and dense attention' >&2; exit 2;
  }
  (cd -- "$FP8KV_DIR/.." && sha256sum -c overlay.sha256) || exit 2
  FP8KV_ARGS=(
    --mount "type=bind,source=$FP8KV_DIR,target=/opt/step5-fp8kv,readonly"
    -e STEP5_FP8KV=1 -e "PYTHONPATH=/opt/step5-fp8kv${FP8KV_EXTRA_PYTHONPATH:+:$FP8KV_EXTRA_PYTHONPATH}"
  )
  FP8KV_VLLM_ARGS=(--block-size 16 --num-gpu-blocks-override "$FP8KV_BLOCKS")
fi
