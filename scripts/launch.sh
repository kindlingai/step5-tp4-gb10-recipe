#!/usr/bin/env bash
# Run on each compute node; additional vLLM arguments can follow the rank.
source "$(dirname -- "$0")/common.sh"
R=${1:?Usage: launch.sh rank [extra vLLM arguments]}; shift
rank_check "$R"
[[ -s $MODEL_DIR/config.json && -s $MODEL_DIR/model.safetensors.index.json ]] || {
  echo 'Model view is missing; complete storage setup first' >&2; exit 2;
}
IMAGE=${IMAGE:-vllm/vllm-openai:step-5-preview}
NAME=${CONTAINER_PREFIX:-step5}-r$R
# Do not remove an existing container automatically.
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "Container $NAME already exists; inspect and remove it explicitly before another launch" >&2; exit 2
fi
mkdir -p "$CACHE_DIR"
HEAD_ARGS=(); [[ $R == 0 ]] || HEAD_ARGS=(--headless)
RDMA_ARGS=(); [[ -z ${RDMA_HCA:-} ]] || RDMA_ARGS=(-e "NCCL_IB_HCA=$RDMA_HCA")
SPEC_ARGS=(); if (( ${SPEC_TOKENS:-3} > 0 )); then SPEC_ARGS=(--spec-method mtp --spec-tokens "${SPEC_TOKENS:-3}"); fi
docker run -d --name "$NAME" --gpus all --ipc=host --network=host --shm-size 16g \
  --ulimit memlock=-1 --device /dev/infiniband \
  --mount "type=bind,source=$STEP5_ROOT,target=$STEP5_ROOT,readonly" \
  --mount "type=bind,source=$CACHE_DIR,target=/cache" \
  -e HOME=/cache -e XDG_CACHE_HOME=/cache -e HF_HOME=/cache/huggingface \
  -e "NCCL_SOCKET_IFNAME=$NETWORK_IFACE" -e "GLOO_SOCKET_IFNAME=$NETWORK_IFACE" \
  -e "VLLM_HOST_IP=${NODES[$R]}" "${RDMA_ARGS[@]}" \
  --entrypoint vllm "$IMAGE" serve "$MODEL_DIR" "${HEAD_ARGS[@]}" \
  --served-model-name step5 --host 0.0.0.0 --port "${API_PORT:-18000}" \
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$R" --master-addr "$NODE0" --master-port "${MASTER_PORT:-29511}" \
  --distributed-executor-backend mp --disable-custom-all-reduce \
  --max-model-len "${MAXLEN:-65536}" --max-num-batched-tokens "${BATCH_TOKENS:-16384}" --max-num-seqs "${MAXSEQ:-8}" \
  --gpu-memory-utilization "${GMU:-0.87}" --enable-prefix-caching --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --trust-remote-code "${SPEC_ARGS[@]}" \
  --reasoning-parser step5 --enable-auto-tool-choice --tool-call-parser step5 "$@"
printf 'Started %s at rank %s\n' "$NAME" "$R"
