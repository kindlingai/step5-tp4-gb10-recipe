"""Experimental FP8 storage with Triton decode and staged stock-FA2 prefill.

Scratch is shared between sequential attention calls, not retained per layer.
Concurrent forwards on different CUDA streams are not supported by this
prototype. No full-model or scheduler integration is implied by its microgate.
"""
import torch
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, TritonAttentionImpl
from .stage import stage_fp8_kv

_SCRATCH = {}


def scratch_for(cache):
    key = (cache.device, tuple(cache.shape))
    if key not in _SCRATCH:
        pages, heads, page, content = cache.shape
        shape = (pages, page, heads, content // 2)
        _SCRATCH[key] = (
            torch.zeros(shape, dtype=torch.bfloat16, device=cache.device),
            torch.zeros(shape, dtype=torch.bfloat16, device=cache.device),
            torch.full((pages,), -1, dtype=torch.int32, device=cache.device),
        )
    return _SCRATCH[key]


class Step5StagedImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_quant_query_input = False
        if self.head_size != 192 or self.sinks is not None or self.alibi_slopes is not None:
            raise ValueError("The Step5 staged candidate requires head192 without sinks/ALiBi")
        if self.dcp_world_size != 1 or self.pcp_world_size != 1:
            raise ValueError("The Step5 staged candidate does not implement context parallelism")

    def _split_kv_cache(self, kv_cache):
        # Step5's fused QK-norm/RoPE writer calls this before Attention.forward,
        # including graph profiling with the temporary nonempty KV allocation.
        # Return NHD views of the ORIGINAL storage, not dequantized scratch:
        # the writer owns the dtype conversion and must update persistent KV.
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def supports_headwise_gate(self):
        # Step5 must apply its ordinary output gate. This backend has no fused
        # gate entry point; SM103-only gated FA2 is not the staged SM121 path.
        return False

    def fused_output_quant_supported(self, quant_key):
        return False

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        # Preserve upstream's split softmax decode, including MTP3 verification.
        if attn_metadata is None or attn_metadata.max_query_len <= 4:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("Staged FA2 does not implement fused output quantization")
        if attn_metadata.use_cascade or attn_metadata.mm_prefix_range_tensor is not None:
            raise ValueError("Staged FA2 requires ordinary causal Step5 attention")
        if query.dtype != torch.bfloat16:
            raise ValueError("Staged FA2 requires BF16 queries")
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            cache = kv_cache.view(torch.float8_e4m3fn)
            k, v, flags = scratch_for(cache)
            stage_fp8_kv(cache, k, v, attn_metadata.block_table,
                         attn_metadata.seq_lens, attn_metadata.query_start_loc,
                         layer._k_scale, layer._v_scale, attn_metadata.max_seq_len,
                         flags, self.sliding_window[0])
        else:
            k, v = self._split_kv_cache(kv_cache)
        n = attn_metadata.num_actual_tokens
        # Lazy import keeps source/CPU checks independent of loaded GPU drivers.
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        flash_attn_varlen_func(
            q=query[:n], k=k, v=v, out=output[:n],
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            block_table=attn_metadata.block_table,
            softmax_scale=self.scale, causal=attn_metadata.causal,
            window_size=self.sliding_window, softcap=self.logits_soft_cap,
            fa_version=2,
        )
        return output


class Step5StagedBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return Step5StagedImpl

    @classmethod
    def get_supported_head_sizes(cls):
        return [192]

    @classmethod
    def supports_compute_capability(cls, capability):
        return (capability.major, capability.minor) == (12, 1)
