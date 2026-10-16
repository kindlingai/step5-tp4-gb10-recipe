#!/usr/bin/env bash
# Run on each compute node; additional vLLM arguments can follow the rank.
source "$(dirname -- "$0")/common.sh"
R=${1:?Usage: launch.sh rank [extra vLLM arguments]}; shift
rank_check "$R"
STEP5_FP8KV=${STEP5_FP8KV:-1}
if [[ $STEP5_FP8KV == 1 ]]; then
  KV_DTYPE=${KV_DTYPE:-fp8_e4m3}
  FP8KV_DIR=${FP8KV_DIR:-$RECIPE_DIR/fp8-kv/overlay}
  : "${MAXLEN:?Set MAXLEN for the selected FP8 profile}"
  : "${FP8KV_BLOCKS:?Set FP8KV_BLOCKS for the selected FP8 profile}"
  : "${GMU:?Set GMU for the selected FP8 profile}"
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}
else
  KV_DTYPE=${KV_DTYPE:-auto}
  MAXLEN=${MAXLEN:-196608}
  GMU=${GMU:-0.85}
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}
fi
source "$RECIPE_DIR/fp8-kv/integration/launch.fragment.sh"
[[ -s $MODEL_DIR/config.json && -s $MODEL_DIR/model.safetensors.index.json ]] || {
  echo 'Model view is missing; complete storage setup first' >&2; exit 2;
}
IMAGE=${IMAGE:-vllm/vllm-openai:step-5-preview}
NAME=${CONTAINER_PREFIX:-step5}-r$R
# Do not remove an existing container automatically.
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists; inspect and remove it explicitly before another launch" >&2; exit 2
fi
# Dense attention: the DSA kernels in this image require SM90/SM103.
DENSE_ARGS=(); [[ ${STEP5_DENSE:-1} != 1 ]] || DENSE_ARGS=(--hf-overrides '{"text_config":{"sparse_config":{"enabled":false}}}')
LM_ARGS=(); [[ ${LMONLY:-1} != 1 ]] || LM_ARGS=(--language-model-only)
# The boot workaround runs on the host, not inside the inference container.
if [[ ${DROP_PAGE_CACHE:-1} == 1 && ! -w /proc/sys/vm/drop_caches ]]; then
  echo 'Run the launcher with elevated host privileges for the page-cache workaround, or set DROP_PAGE_CACHE=0 explicitly' >&2
  exit 2
fi
mkdir -p "$CACHE_DIR"
HEAD_ARGS=(); [[ $R == 0 ]] || HEAD_ARGS=(--headless)
# Example HCA names; configure the actual two rails for each node.
RDMA_HCA=${RDMA_HCA:-mlx5_0,mlx5_1}
RDMA_ARGS=(-e "NCCL_IB_HCA=$RDMA_HCA")
SPEC_ARGS=(); if (( ${SPEC_TOKENS:-3} > 0 )); then SPEC_ARGS=(--spec-method mtp --spec-tokens "${SPEC_TOKENS:-3}"); fi
docker run -d --name "$NAME" --gpus all --ipc=host --network=host --shm-size 16g \
  --ulimit memlock=-1 --device /dev/infiniband \
  --mount "type=bind,source=$STEP5_ROOT,target=$STEP5_ROOT,readonly" \
  --mount "type=bind,source=$CACHE_DIR,target=/cache" \
  -e HOME=/cache -e XDG_CACHE_HOME=/cache -e HF_HOME=/cache/huggingface \
  -e "NCCL_SOCKET_IFNAME=$NETWORK_IFACE" -e "GLOO_SOCKET_IFNAME=$NETWORK_IFACE" \
  -e "VLLM_HOST_IP=${NODES[$R]}" "${RDMA_ARGS[@]}" \
  -e "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}" \
  -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-2}" -e "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-2}" \
  -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS" \
  "${FP8KV_ARGS[@]}" \
  --entrypoint vllm "$IMAGE" serve "$MODEL_DIR" "${HEAD_ARGS[@]}" \
  --served-model-name step5 --host 0.0.0.0 --port "${API_PORT:-18000}" \
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$R" --master-addr "$NODE0" --master-port "${MASTER_PORT:-29511}" \
  --distributed-executor-backend mp --disable-custom-all-reduce \
  --max-model-len "$MAXLEN" --max-num-batched-tokens "${BATCH_TOKENS:-4096}" --max-num-seqs "${MAXSEQ:-8}" \
  --gpu-memory-utilization "$GMU" --enable-prefix-caching --kv-cache-dtype "$KV_DTYPE" \
  --trust-remote-code "${SPEC_ARGS[@]}" \
  --reasoning-parser step5 --enable-auto-tool-choice --tool-call-parser step5 "${DENSE_ARGS[@]}" "${LM_ARGS[@]}" "${FP8KV_VLLM_ARGS[@]}" "$@"
if [[ ${DROP_PAGE_CACHE:-1} == 1 ]]; then
  nohup "$RECIPE_DIR/scripts/drop-page-cache.sh" > "$CACHE_DIR/page-cache-drop.log" 2>&1 < /dev/null &
  echo "$!" > "$CACHE_DIR/page-cache-drop.pid"
fi
printf 'Started %s at rank %s\n'  "$NAME" "$R"
