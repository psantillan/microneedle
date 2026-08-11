# Porting the engine, and the memory ceiling

Needle 2's pitch for MCU-class parts — "bounded session memory... a
deterministic 28MB ceiling, not a curve that grows with conversation
length... compiles single-threaded for bare metal" — describes properties
this engine already has, so state them precisely and keep them true.

## The deterministic ceiling

Every allocation happens once, in ne_alloc, sized by compile-time bounds
(NE_MAX_ENC=384, NE_MAX_GEN=64). Nothing grows at runtime — not with prompt
length, not with generation length, not with anything, because the engine is
single-shot: there is no session state at all.

| Region | MiB |
|---|---:|
| model (.npk, int4 proj / int8 embed) | 14.66 |
| encoder activations (enc_x) | 0.75 |
| cross-attention K/V (fp32) | 6.00 |
| cross-attention int16 mirrors | 3.05 |
| decoder self K/V | 1.00 |
| scratch arena (partitioned in ne_encode) | 3.68 |
| logits | 0.03 |
| **total** | **29.18** |

(The s8 attention configuration halves the int16 mirror/scratch regions,
~0.4 MiB back. Internal SRAM use is separate and small: the activation
block, quant buffers, and stacks — the firmware's 133 KB budget.)

A port to any part with ~32 MB of RAM (ESP32-P4 PSRAM, STM32H7 / NXP
i.MX RT with SDRAM) is therefore a memory fit by inspection, before any
code moves.

## The portability boundary

The engine is one C99 file. Everything platform-specific enters through
exactly two kinds of seam:

1. **Two assembly files**, used only under -DNE_PIE:
   - engine/pie_rows.S — int8/int4 row kernels (projections)
   - engine/pie_attn.S — int16 (and s8) attention dot kernels
   Without NE_PIE the same call sites run scalar C. tools/pie_stub.c is the
   scalar reference for the kernels' exact contracts (including the
   deliberate over-reads a port must preserve slack for).
2. **Five optional function pointers**, all NULL-safe:
   - ne_split_rows / ne_split_gemm / ne_split_attn / ne_split_for — second
     core. NULL = single-threaded, which is the bare-metal default.
   - ne_critical_enter/exit — scheduler guard around vector bursts (an
     IDF-specific context-save bug; a bare-metal port passes NULL).
   - ne_now_us — phase timing; NULL disables.

So the porting recipe for, say, a Cortex-M55: compile needle_engine.c as-is
(scalar, single-threaded, NULL hooks) — that is already a working port, at
scalar speed — then reimplement the two .S files with Helium/MVE intrinsics
behind the same signatures, proving each against tools/pie_stub.c on host
and against the 8 parity fixtures on target, the same way the PIE kernels
were proven. The certification methodology (fixtures byte-identical x3
runs) is architecture-independent and travels with the port.

What does NOT travel: the PIE kernels themselves, the dual-core split
geometry (tuned to this chip's 128 KB L2 — see bench/gemm/README), and any
timing number in README.md. A port re-earns its numbers.
