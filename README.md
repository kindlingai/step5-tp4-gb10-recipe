# Step-5 Preview NVFP4 on four DGX Spark systems

A recipe skeleton for `stepfun-ai/Step-5-Preview-NVFP4` with vLLM tensor parallelism across four NVIDIA DGX Spark systems (GB10).

**Configuration:** dense attention (DSA kernels need SM90/SM103 in this vLLM image), BF16 KV via `auto`, text-only, MTP3, 196608-token (192K) context, dual-rail NCCL with two channels. Historical 128K and 64K results remain below.

**Status:** The configuration below has measured benchmark and API-soak results on four systems. The generalized launch and NFS scripts are CPU-tested; those generalized scripts have not been validated end to end on a live cluster.

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
--max-model-len 196608 --max-num-batched-tokens 4096 --max-num-seqs 8
--gpu-memory-utilization 0.85
--enable-prefix-caching --kv-cache-dtype auto
--reasoning-parser step5 --enable-auto-tool-choice --tool-call-parser step5
--trust-remote-code --language-model-only
--spec-method mtp --spec-tokens 3
--hf-overrides '{"text_config":{"sparse_config":{"enabled":false}}}'
```

Rank 0 serves the OpenAI-compatible API on port 18000. Other ranks use `--headless`. The master address is `NODE0`; the default master port is 29511. The launcher sets `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME`. The default C2 profile selects both fabric rails through `NCCL_IB_HCA`, with `NCCL_CROSS_NIC=1`, `NCCL_MIN_NCHANNELS=2`, and `NCCL_MAX_NCHANNELS=2`.

Set `RDMA_HCA` to a comma-separated pair of actual HCA names for the two rails on each node. The `mlx5_0,mlx5_1` names in the example and fallback are documentation examples, not hardware discovery. Both rails must have working peer connectivity. The socket interface remains separately configured by `NETWORK_IFACE`. These settings come from the measured b4 C2 profile; no host-specific interface names or addresses are needed to describe the rails.

```sh
RDMA_HCA=mlx5_0,mlx5_1  # replace with this node's two fabric HCAs
NCCL_CROSS_NIC=1
NCCL_MIN_NCHANNELS=2
NCCL_MAX_NCHANNELS=2
```

The launcher passes `RDMA_HCA` as `NCCL_IB_HCA`. Each NCCL setting can be overridden in the configuration for a separately measured profile.

### Attention and KV compatibility

The current configuration is **dense attention (DSA kernels need SM90/SM103 in this vLLM image)**. GB10 is SM121, so the default override disables sparse attention:

```json
{"text_config":{"sparse_config":{"enabled":false}}}
```

This changes attention behavior. No quality-equivalence claim is made for dense versus sparse attention, and the single needle-recall check below is not a general quality evaluation. `STEP5_DENSE=1` enables the override by default.

The FA2 path on SM121 in this image does not support FP8 KV. Therefore, `KV_DTYPE=auto` selects the model's BF16 KV dtype rather than FP8. This is a KV-cache choice; the checkpoint remains NVFP4. It is not a claim that all attention backends on SM121 lack FP8 KV support.

For comparison, the separate [0xSero four-Spark recipe README](https://github.com/0xSero/Step-5-Preview-Four-Sparks/blob/main/README.md#configuration), checked on 2026-10-15, specifies **FP16 KV** matching its FP16 model dtype. It is not an FP8-KV recipe. Its configuration is different; no performance or quality equivalence is implied.

`LMONLY=1` enables `--language-model-only` by default, so this recipe serves text and does not load the vision encoder. The current defaults are memory utilization 0.85, a 4096-token batch limit, and three MTP draft tokens. The environment example and launcher use the same defaults.

`SPEC_TOKENS=3` is the default. Set `SPEC_TOKENS=0` to omit speculative arguments for a separate non-MTP configuration. Extra vLLM arguments may follow the rank. No optional MoE overlay is included in this baseline.

The launcher refuses to replace an existing container with the same name. Inspect logs and stop/remove old containers explicitly before another launch. Verify image identity, model readiness, memory allocation, and the actual KV-cache token pool before recording results. This configuration requests `--kv-cache-dtype auto`, which uses BF16 KV for this model. The default 192K profile reported a KV pool of 258,523 tokens (1.31× capacity at 196608 tokens; allocation GiB not recorded here). Historical 128K reported 202,818 tokens (4.23 GiB observed on rank 0); 64K reported 156,294 tokens (4.71 GiB per rank). The launch default is `MAXLEN=196608`. Set `MAXLEN=131072` or `65536` to request a smaller window, but changing context alone does not reproduce the historical single-rail configurations. Verify capacity again after any configuration change.

### Boot-time page-cache workaround

By default, each launcher starts `scripts/drop-page-cache.sh` on its host immediately after the inference container starts. It writes `1` to `/proc/sys/vm/drop_caches` every 10 seconds for 15 minutes (90 writes). This drops clean page cache; it does not use value `3`. Run the launcher with elevated host privileges and preserve `STEP5_CONFIG` when changing accounts.

This loop was used for the measured startup after earlier starts had NVRM errors during KV allocation. The successful startup was reported to have no such errors. This is an observed workaround, not a general driver fix or proof of causation. The benchmark client does not audit kernel logs.

The helper exits after 15 minutes. Its log and PID are stored as `page-cache-drop.log` and `page-cache-drop.pid` under `CACHE_DIR`. Set `DROP_PAGE_CACHE=0` to disable it explicitly. Do not leave the loop running indefinitely. Dropping page cache affects other host workloads and can cause extra storage reads.

To inspect the schedule without changing the host or waiting:

```sh
scripts/drop-page-cache.sh --dry-run
```

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
python3 scripts/bench.py --go --mtp-tokens 3 \
  --base http://192.168.50.11:18000/v1 --output receipt.json
```

The default `bench` mode uses one warmup plus three measured requests at each prefill size, then three 512-token S1 decode requests for each of prose, code, and JSON, followed by three S4 batches of four concurrent 512-token streams. Each request has a different prefix. Regenerate the prompts before a repeated run to avoid prefix-cache reuse. Decode uses `ignore_eos=true` to request the full length.

- Input sizes are exactly 8192 and 32768 tokens after the pinned chat template. The receipt compares server usage with the local counts.
- TTFT runs from request start to first nonempty content or reasoning SSE text.
- Prefill throughput is `prompt_tokens / client TTFT`, not a GPU-only kernel rate.
- Decode throughput is `(usage.completion_tokens - 1) / (last - first text SSE timestamp)`. Usage includes reasoning tokens. SSE chunks can contain several tokens; this is a stream-boundary estimate.
- Per-run MTP acceptance uses accepted/drafted counter deltas from `/metrics`. No other request traffic may occur during these measurements. The default 11-second post-request metrics wait must exceed the server's actual publication interval.
- Speed rows show SM clock MHz and GPU power.draw W as min/median/max across one-second samples from all four nodes during measured request windows. Warmup and metrics-settle gaps are excluded. Rows identify `in-phase 1s` or `boundary fallback`. Sampling can miss shorter peaks. Temperature and timestamped per-node samples are retained in JSON.
- Before and after each phase, the harness records current/max SM clocks, temperature, power, and `gpu-clock-cap` service state on all four ranks. `fullclk` requires the service to be inactive on all four at both boundaries. Failed queries produce `unknown`. This label does not claim every sampled clock reaches the hardware maximum.
- For an externally configured 2000 MHz cap, pass `--clock-label clk2000`. This records the reported target; it does not set or independently verify the cap value. The label requires an active cap service on all four nodes at both boundaries. A different boundary state produces `clock-policy-mismatch`; failed queries produce `unknown`. Actual sampled MHz remain in each table row.
- JSON receipts contain operational addresses, clock command output, and raw metrics. Review and redact them before sharing. The privacy scanner is an automated check, not a complete privacy review.

### In-phase telemetry

`BENCH_HOSTS` supplies four distinct comma-separated node addresses; `SSH_USER` supplies the SSH account. The defaults are documentation placeholders. Before each phase, the client starts one SSH sampler on each node:

```text
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits --loop=1
```

At phase end, the client signals each remote shell to kill and reap its sampler and then reaps the local SSH processes. Closing SSH input also triggers remote cleanup. No persistent sampler service is created, and no clock setting changes.

In-phase summaries require valid samples inside measured request windows from every node and confirmed sampler cleanup. Otherwise, the table uses the before/after boundary samples and labels the fallback. Missing boundary samples show N/A with their count. Cleanup errors stay in the receipt. The local tests exercise the exact shell wrapper with a fake telemetry executable, including stdin-EOF cleanup; they do not contact a cluster.

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

## Measured results: current profile and history

### b4 — 192K context, MTP3, clk2000, dual-rail C2 (default)

Measured on 2026-10-15, run `b4-20261015T223410Z`, with an exclusive server window.
The API listing confirms a 196608-token configured context. The reported KV pool is
258,523 tokens, or 1.31× at that length. This ratio is a capacity estimate, not a
measured concurrent-long-request result. The reported C2 profile uses both HCAs,
`NCCL_CROSS_NIC=1`, and two channels. The benchmark client did not independently
inspect the NCCL environment.

Smoke and all 29 benchmark requests passed with matching token counts and no client
warnings. The 4 GiB memory guard did not abort. Minimum host MemAvailable was
9.5595 GiB; maximum sampled GPU temperature was 56 C. All phase samplers stopped
cleanly. Every cap service was active before and after each phase; `clk2000`
records the externally reported target, not an independent configured-cap reading.

| Phase | Clock label | Sample source | SM MHz min/med/max | GPU W min/med/max | TTFT median (s) | Prefill proxy (tok/s) | Decode (tok/s) | MTP accepted/drafted |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 8K prefill | clk2000 | in-phase 1s | 1969 / 1982 / 1995 | 8.8 / 26.0 / 27.8 | 6.18 | 1324.84 | — | N/A |
| 32K prefill | clk2000 | in-phase 1s | 1969 / 1989 / 1995 | 8.4 / 29.0 / 32.0 | 26.49 | 1236.88 | — | N/A |
| S1 prose, 512 output | clk2000 | in-phase 1s | 1976 / 1982 / 1995 | 7.9 / 18.2 / 18.9 | 0.64 | — | 27.95 | 974 / 1683 (57.9%) |
| S1 code, 512 output | clk2000 | in-phase 1s | 1969 / 1976 / 1995 | 7.9 / 18.3 / 19.8 | 0.34 | — | 28.61 | 980 / 1668 (58.8%) |
| S1 JSON, 512 output | clk2000 | in-phase 1s | 1969 / 1982 / 1989 | 8.8 / 18.2 / 19.5 | 0.33 | — | 27.07 | 978 / 1677 (58.3%) |
| S4, four × 512 output | clk2000 | in-phase 1s | 1969 / 1982 / 1995 | 8.2 / 21.7 / 24.5 | — | — | 60.09 aggregate; 16.06 per stream | 3864 / 6840 (56.5%) |

**Observed prefill improvement versus the single-rail b3 run: approximately +15–16%**
(+16.1% at 8K and +15.3% at 32K). Both profiles use clk2000 and
MTP3. Context/KV capacity and boot also changed, so this is a comparison of the
measured profiles, not an isolated proof that dual rails alone caused the gain.

S4 aggregate rates were 60.09, 62.29, and 60.09 tok/s; the per-stream figure pools
all 12 streams. Rates and TTFT use three-repeat medians. No b4 soak was run.
The benchmark's largest input is 32768 tokens; these results do not prove 192K
recall or quality. The separate 190K needle result is pending and is not included.

### b2 — 64K context, MTP3, fullclk (history)

Measured on 2026-10-15 with the pinned model above, four GB10 systems, dense attention, BF16 KV, memory utilization 0.85, and a 4096-token batch limit. All benchmark requests passed without client warnings or memory-guard aborts. This b2 profile used 65536 context and a 156,294-token KV pool. Its benchmark run was `b2-20261015T201632Z`. The resolved image digest and exact fabric hardware are not included in this result summary.

Rates and TTFT are medians of three measured repetitions. Clock and power values are min/median/max across all four nodes in the measured request windows. Each row uses one-second in-phase telemetry. `fullclk` means the clock-cap service was inactive on every node at both phase boundaries, not that every sample reached maximum clock.

| Phase | Clock label | Sample source | SM MHz min/med/max | GPU W min/med/max | TTFT median (s) | Prefill proxy (tok/s) | Decode (tok/s) | MTP accepted/drafted |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 8K prefill | fullclk | in-phase 1s | 2405 / 2476 / 2561 | 12.0 / 41.8 / 45.5 | 6.97 | 1175.73 | — | N/A |
| 32K prefill | fullclk | in-phase 1s | 2405 / 2476 / 2548 | 11.1 / 47.5 / 54.2 | 29.35 | 1116.56 | — | N/A |
| S1 prose, 512 output | fullclk | in-phase 1s | 2405 / 2489 / 2574 | 15.6 / 33.6 / 37.8 | 0.38 | — | 27.93 | 948 / 1767 (53.7%) |
| S1 code, 512 output | fullclk | in-phase 1s | 2405 / 2489 / 2574 | 11.1 / 33.7 / 38.0 | 0.40 | — | 30.05 | 981 / 1659 (59.1%) |
| S1 JSON, 512 output | fullclk | in-phase 1s | 2405 / 2489 / 2574 | 11.1 / 33.6 / 36.7 | 0.41 | — | 28.79 | 963 / 1710 (56.3%) |
| S4, four × 512 output | fullclk | in-phase 1s | 2405 / 2489 / 2574 | 11.9 / 39.5 / 43.7 | — | — | 60.89 aggregate; 16.08 per stream | 3882 / 6786 (57.2%) |

S4 aggregate rates were 58.49, 62.51, and 60.89 tok/s. The per-stream figure is the median of all 12 streams. MTP ratios pool the accepted/drafted deltas across each row's runs. One-token prefill requests produced no MTP drafts. Minimum observed host MemAvailable was 5.2458 GiB; maximum GPU temperature was 67 C. A separate client supervisor aborted below 4 GiB on any rank; that private supervisor is not bundled with this recipe.

#### b2 mixed-load soak: PASS (mixed clocks)

The API stability run requested 600 seconds and finished its last batch at 614.12 seconds. It completed 100 requests in 40 batches: 34 short chat, 33 exact-8K prompts, and 33 decode requests, at concurrency levels one through four. Request/HTTP errors: **0**. No memory-guard abort occurred; minimum host MemAvailable was 5.1361 GiB.

| Clock label | Sample source | SM MHz min/med/max | GPU W min/med/max | GPU C min/med/max |
|---|---|---:|---:|---:|
| cap-active-or-mixed | in-phase 1s | 1969 / 2476 / 2574 | 18.11 / 34.725 / 42.12 | 43 / 55 / 61 |

The cap service was inactive on all four nodes before the soak and active on all four afterward. Therefore, this is **not a fullclk soak**. All four samplers stopped cleanly, with 613 in-request samples per rank. The reused prompt pool allowed prefix-cache hits. PASS covers client/API stability, not model quality or the absence of kernel/driver errors.


### b3 — 128K context, MTP3, clk2000 (single-rail history)

Measured on 2026-10-15, run `b3-20261015T205109Z`. The configured context is 131072; the reported KV pool is 202,818 tokens, with 4.23 GiB observed on rank 0. Other settings match b2: dense attention, BF16 KV, MTP3, memory utilization 0.85, batch limit 4096, and language-model-only.

Smoke and all 29 benchmark requests passed with matching token counts and no client warnings. The 4 GiB memory guard did not abort. Minimum host MemAvailable was 4.5166 GiB; maximum sampled GPU temperature was 55 C. Every phase had clean four-node sampler cleanup. `clk2000` records the externally reported 2000 MHz target; all four cap services were active before and after each phase. The harness did not independently read the configured cap value.

| Phase | Clock label | Sample source | SM MHz min/med/max | GPU W min/med/max | TTFT median (s) | Prefill proxy (tok/s) | Decode (tok/s) | MTP accepted/drafted |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 8K prefill | clk2000 | in-phase 1s | 1969 / 1982 / 1989 | 8.3 / 23.9 / 25.5 | 7.18 | 1141.46 | — | N/A |
| 32K prefill | clk2000 | in-phase 1s | 1969 / 1982 / 1995 | 7.7 / 26.5 / 29.4 | 30.53 | 1073.19 | — | N/A |
| S1 prose, 512 output | clk2000 | in-phase 1s | 1969 / 1982 / 1995 | 7.8 / 18.1 / 19.4 | 0.68 | — | 27.09 | 966 / 1704 (56.7%) |
| S1 code, 512 output | clk2000 | in-phase 1s | 1976 / 1982 / 1995 | 7.8 / 18.2 / 19.5 | 0.40 | — | 28.91 | 993 / 1620 (61.3%) |
| S1 JSON, 512 output | clk2000 | in-phase 1s | 1976 / 1982 / 1995 | 7.7 / 18.2 / 19.6 | 0.40 | — | 28.55 | 997 / 1629 (61.2%) |
| S4, four × 512 output | clk2000 | in-phase 1s | 1969 / 1982 / 1995 | 8.6 / 21.4 / 23.0 | — | — | 60.30 aggregate; 15.74 per stream | 3902 / 6732 (58.0%) |

S4 aggregate rates were 61.59, 56.98, and 60.30 tok/s; the per-stream figure pools all 12 streams. These are three-repeat medians, using the same measurement definitions as b2. The b2 and b3 clocks differ, so this is not a controlled context-size performance comparison. No b3 soak was run.

#### b3 needle recall: PASS

| Check | Result | Clock label | GPU power |
|---|---|---|---|
| Single long-prompt needle recall | Exact inserted passphrase recalled; 111,059 prompt tokens; 132.4 s client wall time | clk2000 (reported for this boot) | Not sampled for this request |

The request and response receipts verify exact recall and server-reported prompt usage. The recorded curl client time was 132.375184 seconds. This proves recall for this one 111,059-token prompt, not general long-context quality or a full 131072-token request. No secret or raw prompt is included here.

Record the resolved image digest, full model revision, clock samples, KV pool, date, and network configuration with future measurements. Do not compare different clock settings without stating the difference.


### Prior four-Spark recipe and comparison reference

[0xSero's Step-5-Preview-Four-Sparks](https://github.com/0xSero/Step-5-Preview-Four-Sparks) is a prior four-Spark recipe and our comparison reference. Its README reports the following results; these are upstream measurements, not measurements from this recipe.

| Upstream metric | Reported result | Clock label | GPU power |
|---|---|---|---|
| 8K prefill | 1,106–1,327 tok/s cold; 1,365–1,565 tok/s warm | Not reported in source table | Not reported in source table |
| 32K prefill | 1,445 tok/s, one launch | Not reported in source table | Not reported in source table |
| S1 decode | Approximately 25–30 tok/s | Not reported in source table | Not reported in source table |
| S4 decode | Code 66.5 tok/s aggregate; prose 40.3 tok/s aggregate | Not reported in source table | Not reported in source table |

Source: [upstream measured section](https://github.com/0xSero/Step-5-Preview-Four-Sparks/blob/main/README.md#measured), checked 2026-10-15. That recipe uses a different EXL3 mixed-format checkpoint, FP16 KV, MTP2, RoCEnante all-reduce, and 262144 context. Its decode runs stop naturally; ours use fixed 512-token outputs. These are comparison references, not matched benchmarks or evidence of a speed advantage.

We did **not** adopt that recipe's NCCL tuning settings for these runs. No claim is made that those settings were tested here.

## Local validation and privacy check

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
for script in scripts/*.sh; do bash -n "$script"; done
scripts/privacy_scan.sh
```

The tests use temporary files, a mock container command, and a local SSE stub. They test S4 concurrency, smoke failures, and soak HTTP-error logs without contacting a cluster. The scan checks working-tree text, relative filenames, and available commit history for private infrastructure details and common credential formats. It includes its own script. Keep local credentials and generated receipts out of a release bundle.

## Credits and licenses

- [0xSero Step-5-Preview-Four-Sparks](https://github.com/0xSero/Step-5-Preview-Four-Sparks): prior four-Spark deployment recipe and comparison reference. Its NCCL tuning settings were not used for these measurements.

- [StepFun model and model card](https://huggingface.co/stepfun-ai/Step-5-Preview-NVFP4): model, tokenizer, and model documentation. The deployment image supplied for this model is `vllm/vllm-openai:step-5-preview`.
- [vLLM documentation](https://docs.vllm.ai/): inference engine, distributed execution, OpenAI-compatible API, and speculative-decoding metrics.

The recipe scripts use the MIT license in `LICENSE`. The model, image, and their dependencies retain their own licenses. Review the pinned model's license files before use.
