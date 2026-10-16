# Step5 FP8 KV overlay for SM121

This optional overlay stores the dense-attention KV cache in FP8 E4M3.
Queries and outputs remain BF16. Query batches of at most four tokens use
vLLM's Triton attention. Longer query batches convert the live KV pages into
shared BF16 scratch and use the image's FlashAttention 2 implementation.
It does not use native FlashInfer head192 attention.

The hook is **off unless `STEP5_FP8KV=1`**. It changes one backend selection
in memory. It does not edit the image, model source, or cached bytecode.
The model-specific split method returns views of the original packed storage
for Step5's fused QK normalization/RoPE cache writer, including graph profiling.

## Pinned runtime

- Container tag: `vllm/vllm-openai:step-5-preview`.
- Tested local image ID:
  `sha256:d39220a7c026f3ea501f76d4b8c0da4fdfb93c498584bb429f0b192a6734a876`.
  This is an image ID, **not** a registry manifest digest. Tags can move.
- The hook requires SHA256
  `6e8574c31c421b4ebeaff36a8faa3a219cd92744251c477d43df1912a8dfa6f0`
  for `vllm/models/step5/nvidia/model.py`. A mismatch raises an import error.
  Do not remove this check to use another image.
- SM121, head dimension 192, dense attention, BF16 queries, FP8 E4M3 KV,
  and block size 16. The example uses TP4 and MTP3.

The six overlay files are original integration code. vLLM, PyTorch, Triton,
FlashAttention, model weights, and their licenses remain separate dependencies.

## Install in the public recipe

Copy this whole directory into the recipe as `fp8-kv/`. Do not copy only the
Python package: the launcher also uses the manifest and integration fragment.
Put the same files at the same location on all compute nodes.

From the recipe root, apply the launcher patch once:

```bash
git apply --check fp8-kv/integration/launcher.diff
git apply fp8-kv/integration/launcher.diff
```

This patch adds two argument arrays to the existing launcher. It does not change
network interfaces, collective settings, model mounts, or startup order.
If the launcher differs, apply the three additions manually.

From `fp8-kv/`, verify the payload and source the example:

```bash
sha256sum -c overlay.sha256
source ./env.example
```

The example supplies:

```bash
export STEP5_FP8KV=1 KV_DTYPE=fp8_e4m3 FP8KV_BLOCKS=30000
export MAXLEN=442368 GMU=0.85 BATCH_TOKENS=4096 MAXSEQ=8
export STEP5_DENSE=1 LMONLY=1 SPEC_TOKENS=3
```

`env.example` sets `FP8KV_DIR` to the absolute location of its local `overlay/`
directory. Source it from this directory on each node, or export that location
explicitly. These values are an example, not a memory-fit guarantee for another
system. A recipe configuration loaded later can replace them; check the final
launcher environment before boot.

For a different launcher, source `integration/launch.fragment.sh` after its
configuration. Put `"${FP8KV_ARGS[@]}"` among the Docker options, before the
image. Put `"${FP8KV_VLLM_ARGS[@]}"` among the vLLM arguments. Pass
`--kv-cache-dtype "$KV_DTYPE"` and `--max-model-len "$MAXLEN"` through the
base launcher. Do not also pass conflicting block-size or block-cap arguments.
The fragment verifies all six overlay hashes before it starts Docker.

The resulting Docker additions are:

```bash
--mount "type=bind,source=$FP8KV_DIR,target=/opt/step5-fp8kv,readonly" \
-e STEP5_FP8KV=1 -e PYTHONPATH=/opt/step5-fp8kv
```

The resulting vLLM additions are:

```bash
--kv-cache-dtype fp8_e4m3 --block-size 16 \
--num-gpu-blocks-override "$FP8KV_BLOCKS"
```

The hook must be on `PYTHONPATH` before the model import in **every** rank.
If another integration needs `PYTHONPATH`, preserve its container paths with
`FP8KV_EXTRA_PYTHONPATH`; another `sitecustomize.py` will not automatically run.
Combined overlays require separate validation.

## Remote boot wrappers

An environment set only in the local shell does not automatically cross SSH.
`integration/boot.diff` is a **generic adaptation example**, not a patch for a
specific boot script. `integration/boot.fragment.sh` supplies
`fp8kv_remote_env`, which quotes each assignment for a remote Bash command.
Append its output to the boot wrapper's existing remote environment. Retain
all existing network and collective variables. Verify the same settings reach
every rank. The function itself does not contact nodes or boot a model.

## Capacity and limitations

The block override bypasses the profiler's automatic memory limit. Reserve
memory for both persistent FP8 KV **and** the retained BF16 scratch arrays.
Do not calculate context as `blocks * 16` for this mixed full/sliding-window attention model.

For the TP4/MTP3 geometry with a 4,096-token batch:

- Persistent pool: `blocks * 141312` bytes.
- Two retained scratch shapes: `blocks * 24584` bytes, including page flags.
- Combined known cost: `blocks * 165896` bytes, before additional workspace.
- One full-attention group and four sliding-window groups use the shared pool.
  The single-request estimate is `(blocks - 2193) * 16` tokens.
- At 30,000 blocks the known cost is about 4.635 GiB and the admission estimate
  is 444,912 tokens. The example sets 442,368 total input-plus-output tokens.

These formulas are specific to this geometry. Recompute them after changes to
TP, MTP, batch size, cache layout, layer grouping, or the model.
The startup profile does not reserve the final full-pool scratch automatically.
Check all ranks, a cold prefill, and memory after the final graph capture.
A boot or a short request does not establish maximum-context capacity.

Scratch is retained by device and full cache shape, not by layer. Additional
shapes can allocate additional scratch. Concurrent forwards on separate CUDA
streams are unsupported. This path excludes DSA, context parallelism, sinks,
ALiBi, cascade attention, multimodal prefix masks, and fused output quantization.
Prefix-page sharing and long-context quality require separate validation.
FP8 changes numerical values; this is not a bit-exact or speedup guarantee.

## CPU checks

These checks need Python's standard library, Bash, Git, and `sha256sum`.
They do not load a model or initialize CUDA. Backend imports are mocked where
needed; the tests do not prove GPU kernels, full-model boot, or quality.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
bash scripts/privacy_scan.sh
sha256sum -c overlay.sha256
```

To return to the base attention path, remove the FP8 additions and choose the
base recipe's BF16/auto KV configuration. Merely disabling the hook while
retaining FP8 KV arguments is not a BF16 rollback.
