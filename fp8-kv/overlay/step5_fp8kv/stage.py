"""One-layer FP8 -> BF16 staging for paged FA2; persistent cache remains FP8.

The scratch K and V pages are separate, contiguous NHD arrays. Only live pages
that can contribute to this batch are read. No model arithmetic is changed.
"""
import triton
import triton.language as tl


@triton.jit
def _mark(Flags, Table, Lengths, Cuq, table_stride: tl.constexpr,
          PAGE: tl.constexpr, WINDOW: tl.constexpr, MAX_PAGES: tl.constexpr,
          BLOCK: tl.constexpr):
    req = tl.program_id(0)
    logical_page = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Lengths + req)
    qlen = tl.load(Cuq + req + 1) - tl.load(Cuq + req)
    first = 0
    if WINDOW >= 0:
        first = tl.maximum(0, (length - qlen - WINDOW) // PAGE)
    live = (logical_page < MAX_PAGES) & (logical_page >= first) & (logical_page * PAGE < length)
    physical = tl.load(Table + req * table_stride + logical_page, live, 0)
    count = tl.minimum(PAGE, length - logical_page * PAGE)
    tl.atomic_max(Flags + physical, count, live, sem="relaxed")


@triton.jit
def _stage(Cache, K, V, Flags, Table, Lengths, Cuq, KS, VS,
           table_stride: tl.constexpr, cache_b: tl.constexpr,
           cache_h: tl.constexpr, cache_n: tl.constexpr,
           H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
           WINDOW: tl.constexpr, BLOCK: tl.constexpr):
    req = tl.program_id(0)
    logical_page = tl.program_id(1)
    length = tl.load(Lengths + req)
    qlen = tl.load(Cuq + req + 1) - tl.load(Cuq + req)
    first = 0
    if WINDOW >= 0:
        first = tl.maximum(0, (length - qlen - WINDOW) // PAGE)
    if (logical_page >= first) & (logical_page * PAGE < length):
        physical = tl.load(Table + req * table_stride + logical_page)
        # One owner per physical page, including shared prefix aliases.
        count = tl.atomic_xchg(Flags + physical, -1, sem="relaxed")
        if count > 0:
            x = tl.arange(0, BLOCK)
            token = x // (H * D)
            head = (x // D) % H
            channel = x % D
            source = physical * cache_b + head * cache_h + token * cache_n + channel
            valid = (x < PAGE * H * D) & (token < count)
            k = tl.load(Cache + source, valid, 0.0).to(tl.float32) * tl.load(KS)
            v = tl.load(Cache + source + D, valid, 0.0).to(tl.float32) * tl.load(VS)
            target = physical * PAGE * H * D + x
            tl.store(K + target, k, x < PAGE * H * D)
            tl.store(V + target, v, x < PAGE * H * D)


def stage_fp8_kv(cache, k_scratch, v_scratch, block_table, seq_lens, cuq,
                 k_scale, v_scale, max_seq_len, flags, window_left=-1):
    """Scratch must be zero-initialized once and reused. Output layout is NHD.

    Flags must start at -1. The stage resets each marked flag after claiming
    its page. Both launches must run serially on the same CUDA stream.
    """
    pages, heads, page, content = cache.shape
    d = content // 2
    max_pages = triton.cdiv(max_seq_len, page)
    _mark[(seq_lens.numel(), triton.cdiv(max_pages, 128))](
        flags, block_table, seq_lens, cuq, block_table.stride(0),
        page, window_left, max_pages, 128,
    )
    _stage[(seq_lens.numel(), triton.cdiv(max_seq_len, page))](
        cache, k_scratch, v_scratch, flags, block_table, seq_lens, cuq, k_scale, v_scale,
        block_table.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
        heads, d, page, window_left, triton.next_power_of_2(page * heads * d),
    )
