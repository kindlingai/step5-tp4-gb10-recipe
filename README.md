# Step-5 Preview NVFP4 on four DGX Spark systems

A recipe skeleton for `stepfun-ai/Step-5-Preview-NVFP4` with vLLM tensor parallelism across four NVIDIA DGX Spark systems (GB10).

**Status:** CPU-tested recipe draft. The generalized launch and NFS scripts have not been validated on a live cluster. No performance result is claimed below.

## Hardware and software

- Four DGX Spark systems with GB10 GPUs, working GPU drivers, Docker, and NVIDIA Container Toolkit.
- A working inter-node RDMA network. Set the network interface and HCA for your installation.
- Local NVMe capacity on three nodes for the split weights, plus container images, caches, and operating-system headroom.
- NFS client utilities on all compute nodes, a kernel NFS server on the three storage holders, SSH, rsync, and Python 3.
- Image: `vllm/vllm-openai:step-5-preview`. This tag can change; record its resolved digest before a measured run.
- Model revision: **`bc246cc2`**, full pin `bc246cc2848e17b37f377eda8fd825c5b3018d1f`.

The scripts do not install system packages or change GPU clocks. Complete those prerequisites before use. Run only on machines allocated to this experiment.

## Storage: three holders, four compute ranks

The split controls disk placement, not tensor-parallel ownership. Each compute rank sees the complete model through a symlink directory. Local files use local NVMe; other files use read-only NFS mounts.

| Compute rank | Storage role | Weight files | Weight bytes (decimal GB) |
|---|---|---:|---:|
| 0 | Holder 0, plus configuration/tokenizer files | 21 | 93.58 |
| 1 | Holder 1 | 17 | 77.50 |
| 2 | Holder 2 | 57 | 223.09 |
| 3 | No local weight files | 0 | 0 |

The lists contain 95 weight files and 12 metadata files. Keep the pinned file lists unchanged. The uneven split is intentional. All ranks need network access to all three holders during model load; NFS is not a remote GPU-memory extension.

Default layout (replace these paths if required):

```text
/srv/step5/weights        local subset on holders 0–2
/srv/step5/nfs/holder0    remote holder 0, except on its own node
/srv/step5/nfs/holder1    remote holder 1, except on its own node
/srv/step5/nfs/holder2    remote holder 2, except on its own node
/srv/step5/model          complete symlink view
/var/cache/step5          writable container cache
```

The whole storage tree is bound into each container at the same absolute path, read-only. This preserves symlink targets. A separate writable cache is mounted at `/cache`.

## Configure and copy weights

All addresses in `env.example` are documentation placeholders. Set addresses, interface, HCA, storage paths, and SSH account for your cluster. Use the same storage paths on all nodes.

```sh
cp env.example config.env
# Edit config.env before continuing.
export STEP5_CONFIG="$PWD/config.env"
```

Download the complete pinned model on a staging machine with enough space. For example, with the Hugging Face CLI already installed:

```sh
hf download stepfun-ai/Step-5-Preview-NVFP4 \
  --revision bc246cc2848e17b37f377eda8fd825c5b3018d1f \
  --local-dir /mnt/model-source
```

Copy this recipe and your private configuration to each compute node. Keep `config.env` out of version control. On the staging machine, set `SOURCE_DIR` to the completed download, then copy each subset:

```sh
scripts/copy-shards.sh
```

The script uses `rsync -aS`. It copies the 12 metadata files to holder 0 and the upstream checksum manifest to every holder. It never deletes existing files. On each holder, verify its local weight files against the upstream manifest:

```sh
scripts/verify-shards.sh 0  # holder 0; use 1 or 2 on the other holders
```

## Read-only NFS and model view

Use a running kernel NFS server on each holder. This recipe uses standard host NFS tools, not a custom NFS container image. Run the export and mount commands with elevated privileges. The exports are temporary; repeat them after a server restart or configure equivalent persistent exports yourself.

A container can also provide the NFS service through the host kernel's `nfsd`. The alternative deployment uses a privileged container with host networking, a read-only export with `ro,no_root_squash,fsid=0`, and access restricted to the fabric subnet. Clients mount `server:/` with `nfs4` and `ro,nconnect=8,rsize=1048576`.

Both approaches provide the same read-only model-file access. Their export paths and account mapping are not identical: the scripts below export the weights directory by its path and retain the server's default account mapping. The container variant makes the weights directory the NFSv4 export root and disables superuser squashing. Match the mount source and file permissions to the selected server configuration. No container-based NFS setup script is included here.

On each of holders 0, 1, and 2:

```sh
scripts/export-nfs.sh
```

On each compute node, use its own rank:

```sh
scripts/mount-nfs.sh 0  # use 1, 2, or 3 on the other nodes
scripts/mklinks.sh 0
```

Mounts use NFSv4 with `ro,nconnect=8`. Check existing mounts with `findmnt` before reuse. Keep model files readable by the NFS clients. The symlink builder fails on missing files. It does not remove the model directory.

## Multi-node launch

The default launcher uses the multiprocessing executor, not Ray. Start ranks 1, 2, and 3 on their respective nodes, then rank 0. Use the same image and configuration on every rank.

```sh
scripts/launch.sh 1  # on rank 1
scripts/launch.sh 2  # on rank 2
scripts/launch.sh 3  # on rank 3
scripts/launch.sh 0  # on rank 0
```

Default vLLM configuration:

```text
--tensor-parallel-size 4 --nnodes 4
--distributed-executor-backend mp --disable-custom-all-reduce
--max-model-len 65536 --max-num-batched-tokens 16384 --max-num-seqs 8
--gpu-memory-utilization 0.87
--enable-prefix-caching --kv-cache-dtype fp8
--spec-method mtp --spec-tokens 3
--reasoning-parser step5 --enable-auto-tool-choice --tool-call-parser step5
--trust-remote-code
```

Rank 0 serves the OpenAI-compatible API on port 18000. Other ranks use `--headless`. The master address is `NODE0`; the default master port is 29511. The launcher sets `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME`. Set `RDMA_HCA` when NCCL needs an explicit HCA selection.

`SPEC_TOKENS=0` disables the speculative arguments. Extra vLLM arguments may follow the rank. No optional MoE overlay is included in this baseline.

The launcher refuses to replace an existing container with the same name. Inspect logs and stop/remove old containers explicitly before another launch. Verify image identity, model readiness, memory allocation, and the actual KV-cache token pool before recording results. FP8 KV is requested; its capacity and sliding-window behavior are not yet verified by this draft.

This configuration exposes the API on the host network without authentication. Use a trusted isolated network or add an appropriate authenticated front end. Do not expose it directly to the internet.

## Smoke test and benchmark

After the API reports readiness, use your configured rank-0 address:

```sh
curl -f http://192.168.50.11:18000/v1/models
curl -f http://192.168.50.11:18000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"step5","messages":[{"role":"user","content":"Reply with a short greeting."}],"max_tokens":64,"temperature":0}'
```

Prepare benchmark prompts on a CPU machine with `transformers` and `tokenizers` installed:

```sh
scripts/fetch-tokenizer.sh
CUDA_VISIBLE_DEVICES='' python3 scripts/prepare.py
```

Copy only `scripts/bench.py` and the generated `scripts/prompts.json` to the client machine. The client needs only Python's standard library. It must have SSH access to all four ranks for read-only clock checks.

```sh
export BENCH_HOSTS=192.168.50.11,192.168.50.12,192.168.50.13,192.168.50.14
export SSH_USER=  # leave empty to use the SSH default, or set your account
export CLIENT_LABEL=client
python3 scripts/bench.py --go \
  --base http://192.168.50.11:18000/v1 --output receipt.json
```

The default `bench` mode uses one warmup plus three measured requests at each prefill size, then three 512-token S1 decode requests for each of prose, code, and JSON, followed by three S4 batches of four concurrent 512-token streams. Each request has a different prefix. Regenerate the prompts before a repeated run to avoid prefix-cache reuse. Decode uses `ignore_eos=true` to request the full length.

- Input sizes are exactly 8192 and 32768 tokens after the pinned chat template. The receipt compares server usage with the local counts.
- TTFT runs from request start to first nonempty content or reasoning SSE text.
- Prefill throughput is `prompt_tokens / client TTFT`, not a GPU-only kernel rate.
- Decode throughput is `(usage.completion_tokens - 1) / (last - first text SSE timestamp)`. Usage includes reasoning tokens. SSE chunks can contain several tokens; this is a stream-boundary estimate.
- Per-run MTP acceptance uses accepted/drafted counter deltas from `/metrics`. No other request traffic may occur during these measurements. The default 11-second post-request metrics wait must exceed the server's actual publication interval.
- Speed rows show SM clock MHz and GPU power.draw W as min/median/max across eight boundary samples (four nodes before and after the phase). These are not continuous measurements or true phase peaks. Missing samples show N/A and the sample count.
- Before and after each phase, the harness records current/max SM clocks, temperature, power, and `gpu-clock-cap` service state on all four ranks. `fullclk` requires the service to be inactive on all four at both boundaries. Failed queries produce `unknown`. This label does not claim every sampled clock reaches the hardware maximum.
- JSON receipts contain operational addresses, clock command output, and raw metrics. Review and redact them before sharing. The privacy scanner is an automated check, not a complete privacy review.

### S4, smoke, and soak modes

```sh
python3 scripts/bench.py --go --mode s4 --output s4.json
python3 scripts/bench.py --go --mode smoke --output smoke.json
python3 scripts/bench.py --go --mode soak --soak-seconds 600 --output soak.json
```

Set `BENCH_BASE` or pass `--base` for your API address. `--mode s1` retains the original 17-request suite; `--mode bench` is the default and adds 12 S4 requests. Recreate the fixture with the current `prepare.py` before use.

**S4:** three batches, each with prose, code, JSON, and prose streams. All four streams request 512 tokens and start through a thread barrier. Reports include aggregate and individual stream rates. Aggregate decode throughput is `sum(N-1)/(latest last text SSE - earliest first text SSE)`. JSON also records aggregate end-to-end throughput over the batch wall time. Global MTP counters belong to the whole batch, not a single concurrent stream.

**Smoke:** one chat completion asks for the sentence “Two plus two equals four.” PASS requires that sentence after punctuation/case normalization, a nonempty reasoning field, normal stop, and matching prompt usage. The expected answer is a narrow arithmetic/instruction-following sanity check, not a general quality assessment.

**Soak:** the default duration is 600 seconds. Workloads cycle through short chat, an exact-8K prompt, and 512-token decode; concurrency cycles from one to four. The prepared pool is reused, so prefix-cache hits are allowed. No new batches start after the requested duration; the last batch may finish later. `--request-timeout` defaults to 180 seconds and measures socket inactivity, not a strict total wall-time limit.

Every soak request produces a JSONL record in `<output-stem>.requests.jsonl`, with usage, timings, errors, and HTTP failure details. JSON and Markdown contain PASS/FAIL. PASS requires no request errors or warnings, all three workload types, all four concurrency levels, and the full requested duration. This client checks API stability; it does not inspect kernel logs or prove that GPU-driver errors are absent.

All modes retain the explicit `--go` gate. Smoke, S4, and soak failures return a nonzero CLI status. `complete` records whether the mode finished; `passed` records its outcome. Unit tests use short local stub soaks, not a ten-minute live run.

## Results — placeholders only

| Phase | Clock label | SM MHz min/med/max | GPU W min/med/max | Measured repeats | TTFT median (s) | Prefill proxy (tok/s) | S1 decode (tok/s) | MTP acceptance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8K prefill | pending | — | — | — | — | — | — | — |
| 32K prefill | pending | — | — | — | — | — | — | — |
| Prose, 512 output | pending | — | — | — | — | — | — | — |
| Code, 512 output | pending | — | — | — | — | — | — | — |
| JSON, 512 output | pending | — | — | — | — | — | — | — |
| S4, four × 512 output | pending | — | — | — | — | — | — | — |

Record the resolved image digest, complete model revision, measured clock samples, KV-cache pool, run date, and hardware/network configuration alongside any future results. Do not compare rows with different clock settings without stating the difference.

## Local validation and privacy check

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
scripts/privacy_scan.sh
```

The tests use temporary files, a mock container command, and a local SSE stub. They test S4 concurrency, smoke failures, and soak HTTP-error logs without contacting a cluster. The scan checks working-tree text, relative filenames, and available commit history for private infrastructure details and common credential formats. It includes its own script. Keep local credentials and generated receipts out of a release bundle.

## Credits and licenses

- [StepFun model and model card](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4): model, tokenizer, and model documentation. The deployment image supplied for this model is `vllm/vllm-openai:step-5-preview`.
- [vLLM documentation](https://docs.vllm.ai/): inference engine, distributed execution, OpenAI-compatible API, and speculative-decoding metrics.

The recipe scripts use the MIT license in `LICENSE`. The model, image, and their dependencies retain their own licenses. Review the pinned model's license files before use.
