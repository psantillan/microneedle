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

## Hardware constraints (measured, binding)

- **`esp.vld.128` requires 16-byte alignment** — it mis-loads *silently*
  otherwise; use `heap_caps_aligned_alloc(16, ...)` for anything the vector
  unit reads.
- **IDF 5.4.2's lazy PIE context save is broken** (`portasm.S` skips `q3`),
  so inference runs with the scheduler suspended, per vector burst.
- A second core adds ~1.3× aggregate PSRAM bandwidth, not 2× (`bench/dualcore/`).
- Flash is 32 MB (GigaDevice; needs `CONFIG_SPI_FLASH_SUPPORT_GD_CHIP=y`);
  PSRAM runs at 200 MHz only behind `CONFIG_IDF_EXPERIMENTAL_FEATURES=y`.
- Flashing another IDF project replaces the partition table and orphans the
  weights blob; full `idf.py flash` restores it.

## Model semantics that break naive ports

All verified against the reference. The first three produce wrong output rather
than a crash:

1. Cross-attention gets **no** RoPE; every other attention path does.
2. Per-head q/k norms come **before** the GQA repeat, before RoPE.
3. Norm weights apply as `(1 + w)`; residual gates as `sigmoid(g)`.
4. RoPE is half-split (dims 0–31 against 32–63), theta 10000.
5. Embedding lookups scale by `sqrt(512)` on encoder and decoder.
6. The embedding is tied and stored once: encoder input, decoder input, and
   (transposed) the output projection.
7. The encoder has a final norm; the decoder has its own, before logits.
8. Upstream `encode()` returns a *tuple* of (output, mask).

## Limitations

- **Eight fixtures are a smoke test, not a benchmark.** They prove the port
  reproduces the reference; they say little about model quality.
- One user at a time: inference holds the CPU per vector burst; this is not a
  server.
- Tokenization is not on the chip — the board speaks token ids; the tokenizer
  runs in the browser or on the host, both gated against the oracle.
- `NE_MAX_ENC` is 384 tokens and `NE_MAX_GEN` 64, compile-time.
- The engine holds one model in file-scope state; not reentrant.
- The fixtures record what the *model* does, quirks included — one fixture
  answers `"mains power"` where the tool wants `"mains"`.
- The parity oracle uses group-32 fake-quantized weights while the shipped
  artifact is group-512: the board is scored against a slightly stricter
  reference than itself.

## Attribution and license

MIT, see `LICENSE`; third-party notices in `THIRD_PARTY_NOTICES.md`; weight
provenance and licenses in `WEIGHTS.md`.

The Needle model, reference implementation and vocabulary are the work of
[Cactus Compute](https://github.com/cactus-compute/needle) (MIT).
`tokenizer/vocab.txt` derives from theirs. This is an independent port, not
affiliated with or endorsed by them.
