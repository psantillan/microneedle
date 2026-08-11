# bench/gemm

Why the encoder's projections cost what they cost. `bench/vector` says the PIE
row kernel runs at 0.24 cycles/MAC; the projections measured 1.07. This takes
the difference apart one factor at a time.

It links `engine/pie_rows.S` directly rather than copying it. A benchmark that
measures a different instruction sequence than the engine cannot explain the
engine.

## Reading it

Every stage reports cycles/MAC, instructions/MAC and CPI, taken from `mcycle`
and `minstret`. CPI is what makes the numbers diagnostic: the kernel issues a
fixed number of instructions per MAC, so

- **cycles/MAC up, instructions/MAC up, CPI flat** -- more work, running at full
  speed. Fix it by doing less work.
- **cycles/MAC up, instructions/MAC flat, CPI up** -- the same work, stalled.
  Fix it by moving data.

A wall clock cannot tell those apart, and they have opposite fixes. The first
stage prints a counter check; if `minstret` does not move, every CPI below it is
meaningless and the run should be discarded.

## What it found

```
kernel only, both operands internal          0.240 cyc/MAC  0.174 ins/MAC  CPI  1.38
+ float epilogue, y internal, contiguous     0.257                         CPI  1.40
+ y in PSRAM, contiguous                     0.275                         CPI  1.50
+ y in PSRAM, strided by LDY (the engine)    0.285                         CPI  1.55
ne_gemm_rows replica, int4 from PSRAM        0.422  0.297 ins/MAC          CPI  1.42
activation quantize, per tensor              0.077
```

The transposed output write, which looks alarming -- consecutive stores 2 KB
apart, straight into PSRAM -- costs 18%. The int4 unpack raises
instructions/MAC by 71% and CPI not at all. Nothing in the GEMM stalls.

Which meant the missing time was not in the GEMM. It was in `rope`, which was
inside the same timer, calling `cosf` and `sinf` per dimension pair:

```
rmsnorm(512) 11320 cyc    rmsnorm(64) 1477 cyc    rope(64) 8434 cyc
271 tokens x 12 layers -> 424 Mcyc, of which rope 329 Mcyc = 0.91 s
```

The rope angle depends only on position and dimension, so 12288 distinct pairs
were being recomputed 1.25 million times. Precomputing them at load took
`qkv-proj` from 3510 ms to 2463 ms at 271 tokens.

## Attention was over the cliff

Per head at 271 tokens, K is 34.7 KB and V is 36.9 KB: one head is a 71.5 KB
working set, comfortably inside L2. A *contiguous* dual-core head split hands
each core a different KV head (head h pairs with kv head h/2), so 143 KB was
live at once.

One core interleaving two heads reproduces that pressure without needing both
cores, and it is unambiguous:

```
1 head live  (71 KB, fits L2)     0.678 cyc/MAC  0.3976 ins/MAC  CPI  1.71
2 heads live (143 KB, over L2)    4.556 cyc/MAC  0.3983 ins/MAC  CPI 11.44
```

6.7x, with instructions per MAC identical to four decimal places.

**What shipped is head interleave, not a query-range split.**
`ne_attn_heads` walks heads in order 0,2,4,6 then 1,3,5,7; the dual-core split
still partitions that interleaved index range, so both cores land on the same
KV heads at once and share one 71 KB set (`engine/needle_engine.c`). A
query-range split (both cores on the same head, different query tokens) was the
early alternative and is still a valid shape, but it is not what the engine
runs. Scratch size relative to L2 still matters; the interleave is what cleared
the cliff without needing a fused QK/AV rewrite first.

The kernel's own scratch turns out not to matter on its own:

```
1 head, scratch internal          0.678 cyc/MAC  CPI 1.70
1 head, scratch in PSRAM          0.668 cyc/MAC  CPI 1.68
```

It is a few KB, re-read constantly, and simply stays pinned in L2. Moving it to
internal SRAM would buy nothing and cost 8 KB of a 133 KB budget.

## Eight dot products per pass, with QACC

`pie_dots_s16` accumulates into ACCX, which is one 40-bit accumulator, so it
yields one dot product per pass over its streamed operand. QACC is two 256-bit
accumulators, and `esp.vmulas.s16.qacc.ldbc.incp` broadcasts one operand against
a vector of eight. Broadcasting the K element and holding eight queries in the
vector computes eight dot products from one pass over K.

The lane geometry is not in the documentation, so `stage_qacc_probe` reads it
off the hardware with known inputs. Eight lanes, **64 bits each**, landing at
int32 indices 0, 2, 4 ... 14 across the four `esp.st.qacc.*` stores. 64-bit
lanes mean overflow is not a consideration at these ranges.

Two sequencing facts the probe also settled: `esp.vldbc.16.ip` takes immediates
in steps of 4, so a preload cannot advance by one 16-bit element and has to be
followed by an `addi`; and `.incp` advances by 2, so the pointer ends one
element past the next row and has to be walked back.

`pie_qk8_s16` is the kernel, verified exact against `pie_dots_s16` on every lane
before being timed:

```
accx: eight passes over K        0.602 cyc/MAC  0.4540 ins/MAC  CPI 1.33
qacc: one pass, eight lanes      0.400 cyc/MAC  0.2716 ins/MAC  CPI 1.47
```

1.5x, and instructions per MAC fall 40% -- the per-call overhead the ACCX path
pays eight times is paid once. The win is not only traffic.

It is not in the engine, because it was tried there and lost. Prefill went
6365 ms to 6451, and the softmax tail rose with it, 1063 ms to 1146.

The reason is the output layout, and it is not fixable by tuning. QACC drains as
four 128-bit stores holding eight 64-bit lanes, so the kernel emits 64 bytes per
key and lane q of key j sits at `out[j*16 + 2*q]`. Whatever consumes it reads
one score every 64 bytes -- a fresh cache line per element. That strided read
costs more than the kernel saves, and the buffer itself is 24.6 KB per core at
S=384, charged against the same L2 the head interleave had just cleared.

Every way out adds scratch: compact the lanes into contiguous rows (another
8.7 KB and another pass), or dequantize all eight queries together into eight
live score rows (8.7 KB instead of 1.5 KB). Scratch competing with K and V is
what the interleave fixed.

So the kernel stays here, proven and measured, as the thing to reach for if the
consumer ever changes shape -- a path that walks keys outermost, or a V-side
kernel whose 64-byte-per-key output is read once rather than eight times, would
not pay the stride. It is not a speedup for attention as this engine computes
it.

## The L2 cliff

The sweep at the end streams a weight matrix of increasing size. Instructions
per MAC do not move; cycles per MAC move 11x.

```
   64 KB   0.251 cyc/MAC   CPI  1.43
  128 KB   0.525           CPI  2.99     <- L2 is 128 KB
  256 KB   2.773           CPI 15.74
    2 MB   2.851           CPI 16.15
```

This is the constraint behind the engine's shape. A kernel whose streamed
operand fits in L2 runs at the vector unit's real rate; one whose operand does
not runs at PSRAM's. It is why `ne_gemm_rows` unpacks each int4 row into a
512-byte stack buffer and sweeps a whole token block against it -- that turns
the streamed operand into a resident one and drops PSRAM traffic to 256 bytes
per 32768 MACs.

`CONFIG_CACHE_L2_CACHE_256KB` moves the cliff and is worth about 7% end to end.
It also takes 128 KB of internal RAM, after which the HTTP server fails to
start. Not taken.

## Running it

```
idf.py -p /dev/ttyACM1 flash monitor
```

Reflash the firmware afterwards; this replaces it.
