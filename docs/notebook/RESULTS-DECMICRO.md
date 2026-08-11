# RESULTS-DECMICRO

Decoder micro-optimizations on branch `dec-micro`:

1. **Requant hoist** — under `NE_PIE`, decoder self-attention q/k/v share one
   `quant_act` of the normed row before three `gemv_xq` calls (encoder-style).
2. **int16 self-attention** — under `NE_PIE`, int16 mirrors of the self-KV
   cache; scores/values via `pie_dots_s16` with `NE_AQ`/`NE_PQ` as in
   `attend_cross_q16`. Host scalar path still uses fp32 `attend()`.

Exact-path changes: fixture tokens byte-identical to the JAX oracle.

---

## Accumulator-bound arithmetic

Same ranges as the encoder/cross int16 path (`NE_AQ = 4096`, `NE_PQ = 16384`),
documented on `attend_self_q16`:

| Pass    | Widest int32 sum                         | Value                         | Fits int32 |
|---------|------------------------------------------|-------------------------------|------------|
| Scores  | `NE_HEADDIM * NE_AQ * NE_AQ`             | `64 × 4096 × 4096 = 1.074e9`  | yes (`< 2^31`) |
| Values  | `NE_PQ * NE_AQ` (probs sum to one)       | `16384 × 4096 = 6.711e7`      | yes |

`S ≤ 64` does not tighten the score bound beyond the head-dim product (that
product is already the per-row width). The value bound does not grow with `S`
because softmax probabilities are unit-sum. Drain shift stays zero
(`pie_attn.S`).

Packing: K is appended once per step into `[kv][NE_MAX_GEN][64]` (capacity
stride `T`, not live `S`). V is re-packed from the fp32 cache with `quant_vT`
at the fixed `self_sp = round_up(NE_MAX_GEN, 32)` so pack and read share one
stride. Prior stride bugs were pack-time/read-time disagreements; `self_sp` is
stored on the context at alloc.

---

## Estimated cycles saved per token

Board wall time is measured after integration; this is a work-count estimate.

### 1. Eliminated activation quantize passes

Per generated token, per decoder layer, self-attn ran **3×** int8 activation
quantizes of the same 512-vector (q, k, v). Now **1×**.

- Eliminated quantizes: `2 × NE_DEC_LAYERS = 16` per token
- Work per quantize (group-512 layout): one max-reduce + one scale/round over
  512 lanes ≈ **1e3** scalar ops
- Total eliminated: ~**1.6e4** scalar ops/token  
  Brief candidate: **~1.2 ms/token** on the P4 (matches the audit leftover).

Cross-attn still has a single q quantize (no shared sibling projections).

### 2. int16 vs fp32 self-attention MACs

Per layer, causal self-attn over live length `S = pos+1`:

| Path  | Score MACs              | Value MACs              | Total / layer     |
|-------|-------------------------|-------------------------|-------------------|
| fp32  | `NE_HEADS × S × 64`     | `NE_HEADS × S × 64`     | `1024 × S` fp32   |
| int16 | same count via `pie_dots_s16` | same                  | `1024 × S` int16  |

Over a full generate of `G` tokens, Σ_S S ≈ `G(G+1)/2`. For `G = 20`
(typical tool call): ~**1.7e5** MACs/layer × 8 layers ≈ **1.4e6** MACs/token
average path length, moved from fp32 scalar to int16 vector kernels.

At the board’s measured ~12 cyc/fp32 MAC vs ~0.24 cyc/int16 MAC
(`bench/vector`, `bench/gemm`), the attention body alone is roughly a
**~50×** arithmetic cut; brief candidate **~7 ms/token**. Overhead not
subtracted here: one `quant_s16` per query head, `quant_self_k_row` once per
layer, and `quant_vT` over `S ≤ 64` each step (small next to the MAC win).

### Combined (order-of-magnitude)

| Lever            | Audit candidate | Work removed (this change)        |
|------------------|-----------------|-----------------------------------|
| Requant hoist    | ~1.2 ms/token   | 16 quantize passes / token        |
| int16 self-attn  | ~7 ms/token     | ~1e6 int16 vs fp32 MACs / token   |
| **Total**        | **~8 ms/token** | board number after integration    |

Decode was 94–130 ms/token; these two leftovers are the audit-verified slice.

---

## Gate outputs (verbatim)

### Gate 1 — `make -C engine needle_host` + fixtures

```
weather_sf             PASS  enc=48   ref=20  got=20     1105 ms
single_tool_no_args    PASS  enc=22   ref=10  got=10      523 ms
two_tools_pick_second  PASS  enc=87   ref=18  got=18     1688 ms
numeric_arg            PASS  enc=42   ref=13  got=13      895 ms
long_tools_context     PASS  enc=271  ref=22  got=22     4699 ms
no_matching_tool       PASS  enc=45   ref=15  got=15      902 ms
ups_status             PASS  enc=63   ref=19  got=19     1400 ms
unicode_and_punct      PASS  enc=70   ref=30  got=30     1521 ms

8/8 cases match the JAX oracle token-for-token
```

### Gate 2 — stub-parity `NE_PIE` host (`-DNE_PIE` + `tools/pie_stub.c`)

Both changes active. 8/8 byte-identical to the oracle:

```
weather_sf             PASS  enc=48   ref=20  got=20      261 ms
single_tool_no_args    PASS  enc=22   ref=10  got=10      135 ms
two_tools_pick_second  PASS  enc=87   ref=18  got=18      406 ms
numeric_arg            PASS  enc=42   ref=13  got=13      204 ms
long_tools_context     PASS  enc=271  ref=22  got=22     1536 ms
no_matching_tool       PASS  enc=45   ref=15  got=15      226 ms
ups_status             PASS  enc=63   ref=19  got=19      306 ms
unicode_and_punct      PASS  enc=70   ref=30  got=30      422 ms

8/8 cases match the JAX oracle token-for-token
```

### Gate 3 — `python3 tools/eval_tools.py weights/needle_tools_g512.npk`

Baseline captured before the change; post-change output identical:

```
  OK  What's the weather in Oslo?          -> get_weather    {"location": "Oslo"}
  OK  When does the sun set in Reykjavik?  -> get_sun_times  {"location": "Reykjavik"}
  OK  Tell me about tardigrades            -> look_up        {"topic": "tardigrades"}
  OK  Set a timer for 5 minutes            -> set_timer      {"minutes": 5}
  OK  Play a chime                         -> play_tone      {"sound": "chime"}
  OK  How long have you been running?      -> board_status   
  OK  Write me a poem about the sea        -> None           

  7/7 correct with ALL SIX tools offered (250 token prompt)
```

### Gate 4 — `make -C engine pie-check`

```
make: Entering directory '.../engine'
cc -O2 -std=c99 -Wall -Wextra -DNE_PIE -c needle_engine.c -o /dev/null
make: Leaving directory '.../engine'
```

Warning-free (exit 0).

### Gate 5 — mutation test

Corrupted the self-attn K read stride in `attend_self_q16`:
`kv * T * NE_HEADDIM` → `kv * S * NE_HEADDIM` (capacity-T pack vs live-S read).

Mutated gate 2:

```
weather_sf             FAIL  enc=48   ref=20  got=63      548 ms
single_tool_no_args    FAIL  enc=22   ref=10  got=63      449 ms
two_tools_pick_second  FAIL  enc=87   ref=18  got=63      762 ms
numeric_arg            FAIL  enc=42   ref=13  got=63      601 ms
long_tools_context     FAIL  enc=271  ref=22  got=63     1929 ms
no_matching_tool       FAIL  enc=45   ref=15  got=63      527 ms
ups_status             FAIL  enc=63   ref=19  got=63      589 ms
unicode_and_punct      FAIL  enc=70   ref=30  got=63      652 ms

0/8 cases match the JAX oracle token-for-token
```

Reverted the stride; gate 2 returned to **8/8**.

---

## Files touched

- `engine/needle_engine.h` — self-KV int16 mirror fields (`self_kq`, `self_vq`,
  `self_ksc`, `self_vsc`, `self_sp`, raw pointers)
- `engine/needle_engine.c` — `quant_act` / `gemv_xq` hoist; `quant_self_k_row`,
  `attend_self_q16`; `ne_step` / `ne_alloc` / `ne_free` wiring
