# RESULTS-INT8: int8 encoder attention feasibility

Host-only measurement of whether signed-int8 attention quantization preserves
outputs relative to the fixture-certified fp32 scalar path. No board, no
firmware change, no assembly. The PIE s8 kernel is worth writing only if this
says so.

Branch: `int8-attn`. Ranges and path selected by `#ifdef NE_ATTN_I8` on top of
the existing `NE_PIE` integer attention path; host target `make -C engine needle_i8`
links `tools/pie_stub.c` the same way as the int16 stub-parity build.

## What changed

| Item | Detail |
|---|---|
| Flag | `#ifdef NE_ATTN_I8` (requires `NE_PIE`) |
| Ranges | `NE_AQ=127`, `NE_PQ=127` (int16 defaults `4096` / `16384` untouched) |
| Scope | Encoder self-attention **and** decoder cross-attention (they share `quant_s16` / `quant_kT` / `quant_vT` / `NE_AQ` / `NE_PQ`) |
| Storage | Still `int16_t` + `pie_dots_s16` for this experiment; quantized values sit in the signed-int8 range a future s8 kernel would use |
| Off path | `NE_ATTN_I8` undefined → today's binaries bit-for-bit |

### Overflow-bound arithmetic

Same shape as the int16 comment above `#define NE_AQ` in `engine/needle_engine.c`.
Accumulators are int32; the kernel drains with a zero shift.

| Accumulator | Formula | int16 (`AQ=4096`, `PQ=16384`) | int8 (`AQ=127`, `PQ=127`) |
|---|---|---|---|
| Scores (`q·k`) | `HEADDIM · AQ · AQ` | `64 · 4096 · 4096 = 1,073,741,824` | `64 · 127 · 127 = 1,032,256` |
| Values (`p·v`) | `PQ · AQ` (probs sum ≈ `PQ`, **not** `S · PQ · AQ`) | `16384 · 4096 = 67,108,864` | `127 · 127 = 16,129` |

Both int8 sums sit far under `INT32_MAX` (2,147,483,647). Value-side headroom is
what the probability-sum-to-one property buys: at `S=512` the naive bound would
be `S · PQ · AQ = 8,257,536`, still fine, but the real widest sum is `PQ · AQ`.

`NE_AQ=127` / `NE_PQ=127` is the signed-int8 max so a future s8×s8 kernel needs
no saturating cast; probabilities stay non-negative inside the same type.

## Gate 1 — untouched paths

| Check | Result |
|---|---|
| `make -C engine needle_host` + fixtures | **8/8** |
| `make -C engine pie-check` (with and without `NE_ATTN_I8`) | clean |
| Stub-parity int16 (`needle_pie`, `NE_PIE` only) vs fixtures | **8/8** |

## Gate 2 — int8 vs fp32 on the 8 fixtures

Weights: `weights/needle_v3_g512.npk`, vectors: `weights/vectors_int4p_int8e.txt`.

| Fixture | Byte-identical to JAX oracle |
|---|---|
| weather_sf | PASS |
| single_tool_no_args | PASS |
| two_tools_pick_second | PASS |
| numeric_arg | PASS |
| long_tools_context | PASS |
| no_matching_tool | PASS |
| ups_status | PASS |
| unicode_and_punct | PASS |
| **Total** | **8/8** |

int8 attention is token-for-token identical to the oracle on every acceptance
fixture, including the 271-token prefill case.

## Gate 3 — 36-case tools battery

Battery = 29 standard-schema cases from `tools/probe_latent_audio.py` (no
causal schema rewrites) + 7 `tools/eval_tools.py` cases, same construction as
`tools/eval_warmschema.py`. Weights: `weights/needle_tools_g512.npk` (fine-tune).
Exact = `needle_trace` (fp32 scalar). int8 = `needle_i8_trace` (`NE_PIE` +
`NE_ATTN_I8` + stub). Driver: `tools/measure_attn_i8.py`.

| Metric | Count |
|---|---|
| Tool-name agreement (exact vs int8) | **35/36** |
| Full byte-identical sequences | **35/36** |
| Expected-tool hit, exact | 35/36 |
| Expected-tool hit, int8 | **36/36** |

### Divergent pair (verbatim)

Only `brd03` — query `"What chip are you running on?"`, expected `board_status`.

```
exact name='play_tone'
  ids=[4, 356, 294, 264, 549, 8062, 4673, 265, 393, 282, 5171, 264, 2260, 270, 503]
  text: [{"name":"play_tone","arguments":{"sound":"chime"}}]

i8    name='board_status'
  ids=[4, 356, 294, 264, 1073, 8062, 682, 265, 393, 630]
  text: [{"name":"board_status","arguments":{}}]
```

Exact is wrong here (routes to `play_tone`); int8 is right. The exact path's
first-name-piece margin for id 549 (`play`) is only **+0.386** over the best
rival name-piece — a coin-flip residual that int8 tips the other way. Not a
routing regression under int8; a case the exact model already misroutes with
almost no margin.

## Gate 4 — margin analysis

Margin = logit of the emitted first-name-piece minus the best rival among
`{play, get, look, set, board}` name-piece ids, at the step after
`"name":"` (same definition as `tools/probe_latent_audio.py`). Shift is
reported only when exact and int8 share the token prefix through that step
(33 of 36 cases; `poem00` / `e7_poem` emit no call; `brd03` diverges).

| Statistic | Value |
|---|---|
| Aligned name-steps | 33 |
| Margin shift mean (`i8 − exact`) | **+0.639** |
| Margin shift min | **−0.656** (`wea02`) |
| Margin shift max | **+2.106** (`lok02`) |
| Margin shift std | 0.622 |
| Exact margin mean / min | 12.13 / 8.57 |
| int8 margin mean / min | 12.77 / 9.45 |
| Cases with exact > 0 but int8 < 0 | **0** |
| Winner flips at name-piece | **1** (`brd03`, exact wrong → int8 correct) |

Typical decision margins (~9–15 logits) dwarf the largest adverse shift
(−0.66). The brief's prior (+12.6 at the final layer on audio) matches
`aud00` exact margin +12.64.

## Gate 5 — mutation test

Deliberately halved `NE_AQ` under `NE_ATTN_I8` from 127 → 63, rebuilt
`needle_i8`, re-ran fixtures, restored.

| Config | Fixtures |
|---|---|
| Mutated `NE_AQ=63` | **7/8** — `no_matching_tool` FAIL (first divergence at arg index 12: ref `295` vs got `4306`) |
| Restored `NE_AQ=127` | **8/8** |

The gate can go red; the chosen range is load-bearing, not decorative.

## Verdict

**Yes — the int8 PIE attention kernel is worth building.**

Measured accuracy cost on this host experiment:

- **Fixtures (port gate):** zero — 8/8 byte-identical to the JAX oracle, same as
  int16.
- **Tools battery (routing gate):** 35/36 byte-identical to exact; the single
  divergence is a case exact already loses with a +0.39 margin, and int8
  *fixes* it. Tool-name agreement with the expected label is 36/36 under int8
  vs 35/36 under exact.
- **Margin cost:** mean shift **+0.64** (int8 slightly *sharper* on average);
  worst adverse shift **−0.66**, against typical margins of 9–15. No
  aligned case loses a positive margin to a negative one.

Encoder self-attention and decoder cross-attention both used the int8 ranges
(shared helpers). That is the full integer attention surface the board would
run; no separate encoder-only escape hatch was needed.

Recommended next step: an s8 `pie_dots` (or s8 QACC variant) with the same
`NE_AQ=127` / `NE_PQ=127` contract, behind the existing `NE_ATTN_I8` switch,
validated first by this host stub and then on-board against the same two
gates. Expected upside is the remaining ~2× on the attention MAC loop (int16
already took 30 s → 6 s at 271 tokens); this measurement says the accuracy
budget is there.
