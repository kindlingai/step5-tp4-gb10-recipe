"""Select opt-in staged FA2 prefill and Triton decode for Step5 on SM121."""
from vllm.platforms import current_platform
from step5_fp8kv.staged_backend import Step5StagedBackend


def select_backend(stock, head_dim, cache_config):
    # The opt-in hook is local to Step5. Do not change other GPU families.
    if not current_platform.is_device_capability(121):
        return stock
    if head_dim != 192:
        raise ValueError("STEP5_FP8KV is tested only for head_dim=192")
    dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise ValueError("STEP5_FP8KV requires BF16 or FP8 E4M3 cache")
    return Step5StagedBackend
