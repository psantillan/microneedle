# KERNEL-S8: the int8 attention kernel, host-proven, board-ready

Branch `s8-kernel` (on top of `int8-attn`). Everything here ran on the host
or through the xesppie toolchain as a build; nothing touched the board.

## What was built

| Piece | Where |
|---|---|
| `pie_dots_s8` — fused s8 dots kernel, 16 lanes/vector, 4x unrolled (64 MACs/iter), len % 64 == 0 | `engine/pie_attn.S` |
| Scalar stub twin with 16-element over-read tripwire | `tools/pie_stub.c` |
| `ne_aq_t` element type: NE_ATTN_I8 selects int8 storage + `pie_dots_s8` (`ne_dots`), Sp rounds to 64 (`NE_SPALIGN`); default build is int16 exactly as before | `engine/needle_engine.h`, `engine/needle_engine.c` |
| K/V mirrors, arena carves, quantizers all typed `ne_aq_t` — mirror memory halves under NE_ATTN_I8 (~406 KB saved at S=384) | same |
| `stage_dots_s8` — s8 vs s16 on the two engine shapes, exactness-gated before timing | `bench/gemm/main/gemm_main.c` |

Overflow arithmetic (in the kernel header): scores 64·127·127 = 1,032,256;
values 127·127 = 16,129 (probabilities sum to ~NE_PQ). Both trivially inside
int32, let alone the 40-bit ACCX; zero drain shift.

## Host gates (all run, verbatim results)

| Gate | Result |
|---|---|
| `needle_host` (scalar) fixtures | `8/8 cases match the JAX oracle token-for-token` |
| `needle_pie` (NE_PIE stub, int16 default) fixtures | `8/8` — the ne_aq_t refactor is identity for the default build |
| `needle_i8` (NE_PIE + NE_ATTN_I8 stub) fixtures | `8/8` — int8 storage, Sp=64 packing, s8 stub with over-read |
| ASan+UBSan build of needle_i8, full fixture run | clean, `8/8` — the 16-element slack provably exists at every A operand |
| `make -C engine pie-check` (both flag states) | clean |
| `./verify.sh` (house suite) | all green: engine 8/8, stub parity 8/8, tokenizer 8/8, JSON 8/8 |
| `idf.py build` of `bench/gemm` (xesppie) | builds — the s8 asm assembles |
| `idf.py build` of `firmware` (NE_ATTN_I8 off) | builds clean — refactor is firmware-safe |

## Board checklist (integrator; the only steps that need hardware)

1. `cd bench/gemm && idf.py -p /dev/ttyACM1 flash monitor` — read the
   `s8 dots kernel vs s16` stage. The exactness line must say
   `exact on both shapes` BEFORE the timings mean anything. Expected:
   ~0.12–0.15 cyc/MAC on the score shape vs s16's ~0.24, larger relative win
   on the S=271 value shape (half the streamed bytes).
2. Add `NE_ATTN_I8` to the firmware engine build; suggested boot self-test
   next to `UT pie alignment` (mirrors the existing s16 UT at fw_main.c:195):
   ```c
   static int8_t a8[80] __attribute__((aligned(16)));   /* 64 + 16 slack */
   static int8_t b8[3 * 64] __attribute__((aligned(16)));
   int32_t o8[3];
   for (int i = 0; i < 64; i++) a8[i] = (int8_t)((i * 37 % 255) - 127);
   for (int i = 64; i < 80; i++) a8[i] = 0;
   for (int r = 0; r < 3; r++) for (int i = 0; i < 64; i++)
       b8[r * 64 + i] = (int8_t)(((i * 37 + r * 11) % 255) - 127);
   pie_dots_s8(a8, b8, 64, 3, o8);
   /* compare against the scalar loop, "UT dots s8" PASS/FAIL */
   ```
3. Reflash firmware; certification is against the **int8 host reference**
   (`needle_i8` outputs), not the old int16 board build: fixtures are 8/8
   byte-identical either way, but the tools battery differs on exactly one
   case (brd03 flips to board_status — the correct tool; RESULTS-INT8.md).
   3 board runs diffed against each other + against needle_i8 host output.
4. Ledger: prefill at S=271 and ms/token before/after, plus the freed
   ~406 KB of PSRAM mirrors.

## Risks — confirmed vs assumed

- **Confirmed**: `esp.vmulas.s8.xacc.ld.ip` semantics and lane order — the
  identical instruction sequence is the boot-verified fused path of
  `pie_rowsum_i8` (engine/pie_rows.S), running on this exact board today.
  My kernel assembles under xesppie (bench build above).
- **Confirmed**: the engine's whole int8 data path (quantize, pack, strides,
  Sp=64, slack) — host stub 8/8 + ASan. The asm is the ONLY board-new code.
- **Assumed until stage 1 runs**: that my per-row xacc reset + 4x fused
  unroll is correct/stall-free on hardware. That is precisely what the
  bench exactness gate checks first, at zero integration risk.
- **Known interaction** (parent session already tracking): merging with
  `dec-micro` extends NE_AQ/NE_PQ=127 into decoder self-attention via the
  shared helpers — re-run `tools/measure_attn_i8.py` on the merged tree
  before certifying.
