# RESULTS-WARMF4: product warm-schema F=4

Host-only productization on branch `warm-f4-prod`, based on `warm-hybrid`.
Exact `ne_encode` path untouched. Opt-in via `-DNE_WARMSCHEMA` (product freeze
depth pinned to **F=4**).

Weights: `weights/needle_tools_g512.npk` (routing battery).  
Parity: `weights/needle_v3_g512.npk` (8 fixtures).  
Reference capture: first battery case, `"Play a chime"` (full six-tool schema,
ns=243).

---

## Product API

```c
#define NE_WARM_F 4
ne_schema_cache_t *ne_schema_capture(ne_ctx_t *ctx, const int32_t *enc_ids, int n);
void   ne_schema_cache_free(ne_schema_cache_t *cache);
size_t ne_schema_cache_bytes(const ne_schema_cache_t *cache);   /* measured */
int    ne_schema_cache_matches(const ne_schema_cache_t *cache,
                               const int32_t *enc_ids, int n);
int ne_encode_warm(ne_ctx_t *ctx, const int32_t *enc_ids, int n,
                   const ne_schema_cache_t *cache);   /* -1 on key mismatch */
int ne_generate_warm(..., const ne_schema_cache_t *cache);
```

Key = tools-JSON token sequence (ids from `<tools>` marker through end).
Mismatch returns **-1**; there is no silent fallback to exact encode (caller
must call `ne_encode` / `ne_generate`).

Research multi-F curve remains in `RESULTS-HYBRID.md`;
`tools/eval_warmschema.py` delegates to the product harness.

---

## Cache design (F=4)

### Contents (only what F=4 needs)

| Tensor | Shape | Role |
|--------|-------|------|
| schema K | `[4][ns][256]` fp32 | layers 0..3, post head-norm |
| schema V | `[4][ns][256]` fp32 | layers 0..3, post head-norm |
| residual x | `[ns][512]` fp32 | after layer 3 (splice source) |
| schema_ids | `[ns]` int32 | cache key |

Nothing above layer 3 is stored (no late K/V, no final-norm schema).

### General formula (fp32)

```
bytes = F * ns * KVDIM * 4 * 2   # K and V
      + ns * DMODEL * 4          # residual
      + ns * 4                   # key ids
      = 4*ns*256*4*2 + ns*512*4 + ns*4
      = 10244 * ns
```

With `F=4`, `DMODEL=512`, `KVDIM=256`.

### Measured size (6-tool schema, ns=243)

| | bytes | KiB |
|--|------:|----:|
| formula | 2 489 292 | 2430.95 |
| **measured** (`ne_schema_cache_bytes`) | **2 489 292** | **2430.95** |
| match | yes | |

Breakdown (formula):

| piece | bytes |
|-------|------:|
| K (4 × 243 × 256 × 4) | 995 328 |
| V | 995 328 |
| residual x | 497 664 |
| key ids | 972 |
| **total** | **2 489 292** |

### int16 vs fp32

**Decision: store fp32.** Acceptance is byte-identical token sequences on the
scalar host path. int16 mirrors are a PIE-path concern (per-row scale); caching
int16 would either (a) dequant→requant and break host identity, or (b) require
a full PIE warm path that is out of scope for this host productization. Board
PIE can requantize from these fp32 rows at splice time later without changing
the cache layout. Residual must stay fp32 for phase-B arithmetic either way.

---

## Capture cost

| | host measurement |
|--|--|
| one full encode of reference prompt (ns=243, n≈247) | **~4.0–4.6 s** wall (`capture_ms` from `CLOCKS_PER_SEC`) |
| when to run on board | **boot** for a fixed catalogue, or **first request per schema key** if schemas can change |

Capture is one exact `ne_encode` with hooks that copy only F=4 tensors; cost ≈
one full prefill of a representative prompt carrying that schema.

---

## Tool-retrieval interaction (k=2)

Lexical top-k=2 over the 36-case battery yields **11 distinct tool subsets**.
Cache size scales with schema token length:

| scope | distinct entries | total payload |
|-------|-----------------:|--------------:|
| one cache per k=2 subset | 11 | **10 786 932 B (~10.5 MiB)** |
| single full-6 cache | 1 | **2 489 292 B (~2.4 MiB)** |

Per-subset examples (formula bytes):

| subset (names) | n cases | ns | bytes |
|----------------|--------:|---:|------:|
| get_sun_times,set_timer | 7 | 86 | 880 984 |
| board_status,get_sun_times | 6 | 77 | 788 788 |
| get_sun_times,play_tone | 5 | 93 | 952 692 |
| … | | | |
| full-6 (OOD poem fallback) | 1 | 243 | 2 489 292 |

**v1 scope: cache the FULL six-tool schema only** (retrieval OFF, or the OOD
full-list fallback under retrieval). Quantitatively: multi-entry caching of
every k=2 subset is ~4.3× the memory of one full-6 cache and needs a map keyed
by schema tokens on every request. A single full-6 cache still wins for
fixed-schema deployments and for any path that sends the full catalogue; with
retrieval ON the warm path simply does not apply unless the pruned schema
matches a stored key (guard returns -1 → exact encode).

---

## Acceptance gates (verbatim)

### 1. Exact path + pie-check

```
./engine/needle_host weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt
→ 8/8 cases match the JAX oracle token-for-token

make -C engine pie-check          # -DNE_PIE only
→ clean

make -C engine pie-check-warm     # -DNE_PIE -DNE_WARMSCHEMA
→ clean
```

### 2. Warm-F4 vs exact — 36-case battery

```
python3 tools/eval_warmf4.py weights/needle_tools_g512.npk
```

```
CACHE measured: ns=243 bytes=2489292 capture_ms=4235.48
formula_bytes=2489292  measured_matches_formula=True

[29-case battery] tool-name exact 28/29 warm 28/29
[29-case battery] byte-identical 29/29
[29-case battery] row-layers mean 2024/2996 = 67.6% (1.48x)

[7-case eval_tools] tool-name exact 7/7 warm 7/7
[7-case eval_tools] byte-identical 7/7

OVERALL F=4: tool-name warm 35/36 (exact 35/36)  byte-identical 36/36
encoder work mean 2028/3000 = 67.6% (1.48x saving)
```

(`brd03` mis-routes under exact as well — warm matches exact.)

### 2b. Fixture warm parity (matching-schema capture each)

The 8 parity fixtures **do not share a schema** (7 unique keys). Per fixture:
capture on that prompt, warm-encode the same ids, compare to exact.

```
fixtures=8 unique_schema_keys=7
OK weather_sf            ns=34
OK single_tool_no_args   ns=17
OK two_tools_pick_second ns=81
OK numeric_arg           ns=34
OK long_tools_context    ns=259
OK no_matching_tool      ns=34
OK ups_status            ns=52
OK unicode_and_punct     ns=53
fixture warm parity: 8/8
```

### 3. Memory — measured

`ne_schema_cache_bytes` = **2 489 292** for ns=243 (matches formula).

### 4. Mutation test

```
aud01 k_mutate: layer=3 row=121 enc_x_max_abs=0.00516152 state_diverged=1
aud01 k_restore: enc_x_max_abs=0 state_restored=1
aud01 x_mutate_all: token_diverged=1
aud01 x_restore: token_identical=1
MUTATE_OK 1
```

One cached K row (layer 3, mid-schema): **encoder state diverges**
(max-abs ~5e-3 on `enc_x`); restore returns to bit-match. Greedy **tokens**
often stay put after a single K-row poke at F=4 (phase B recomputes layers
4..11 over the intact residual). Zeroing the residual splice source flips
tokens hard; restore recovers exact identity. Gate records both.

### 5. Wrong-schema negative control

```
WRONG_SCHEMA matches=0 warm_rc=-1 (expect matches=0 warm_rc=-1)
WRONG_SCHEMA_OK 1
```

All 36 battery cases share the full-6 key, so the harness flips one schema
token on the reference prompt. `ne_schema_cache_matches` is 0 and
`ne_generate_warm` returns **-1** — silent wrong-cache use is impossible.

---

## Reproduce

```bash
make -C engine needle_host pie-check pie-check-warm needle_warmschema
./engine/needle_host weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt
python3 tools/eval_warmf4.py weights/needle_tools_g512.npk
```

---

## Verdict — ship-to-board recommendation

**Yes, as an opt-in engine mode for fixed-schema (full catalogue) deployments.**

| claim | value |
|-------|------:|
| correctness | byte-identical on 36/36 battery + 8/8 fixtures (per-schema capture) |
| prefill saving | **1.48×** encoder row-layers (67.6% of full) |
| memory budget | **~2.4 MiB** fp32 payload per cached schema (ns=243) |
| capture | one full encode per schema (~4 s host; board: boot or first request) |
| retrieval | v1 = full-6 only; k=2 multi-entry would need ~10.5 MiB for 11 subsets |

Caveats: saving is modest vs full-freeze fantasy (~35×, which destroys routing).
Worth it when ~2.4 MiB PSRAM is available and early-layer schema projections
are a measured prefill cost. Not a substitute for tool-retrieval’s token-length
win (~2.5× shorter prompts at k=2); the two compose only if you also cache
pruned schemas. Firmware enablement is a flag flip (`NE_WARMSCHEMA` / future
`NE_WARMF4`) plus a capture call at boot for the installed tool catalogue.
