# RESULTS-HYBRID: layer-split warm schema (freeze depth F)

Host-only experiment on branch `warm-hybrid`, extending the `NE_WARMSCHEMA`
prototype from `schema-cache`. Exact engine path untouched.

Weights: `weights/needle_tools_g512.npk` (fine-tune; routing battery).  
Parity weights: `weights/needle_v3_g512.npk` (base; 8/8 fixtures).  
Reference capture: first case, `"Play a chime"` (standard schema, ns=243).

---

## Method

Capture (unchanged + residual already stored per layer): for the reference
query, store schema-row residual after every checkpoint L (0=post-embed,
1..12=post-layer, 13=final norm) and schema K/V at every encoder layer.

Warm pass with freeze depth **F** ∈ {0..12}:

| Phase | Layers | Behavior |
|-------|--------|----------|
| A (partial) | 0 .. F−1 | Recompute **query rows only**. Query attends over `[fresh query K/V ‖ cached schema K/V]`. |
| Splice | at F | Schema residual ← cached post-layer-(F−1) state (checkpoint F). |
| B (full) | F .. 11 | Full normal computation over **all** rows (schema now sees the real query). |
| Tail | final norm, cross-K/V, decode | Unchanged; over all rows when F&lt;12. |

Special cases:

- **F=0**: no partial phase; splice post-embed schema residual; full layers 0..11 → must be byte-identical to exact.
- **F=12**: full freeze (legacy prototype); final-norm schema from cache.

Estimated encoder work (row-layers):

```
row_layers = F * nq + (12 − F) * ntot
full       = 12 * ntot
```

Partial layers count only the recomputed query rows; attention still runs
`nq` queries over `ntot` keys (same `nq/ntot` fraction as the projections).

---

## Acceptance gates

| Gate | Result |
|------|--------|
| `make -C engine needle_host && ./engine/needle_host weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt` | **8/8** |
| `make -C engine pie-check` | clean |
| F=0 vs exact on 36 cases | **36/36 byte-identical** |
| F=12 (full freeze) | **11/36** tool-name, **9/36** byte-identical (matches RESULTS.md) |

---

## Sanity anchors

### F=0 (exact path via splice)

| Set | tool-name exact | tool-name warm | byte-identical |
|-----|----------------:|---------------:|---------------:|
| 29-case battery | 28/29 | 28/29 | **29/29** |
| 7-case eval_tools | 7/7 | 7/7 | **7/7** |
| Combined | 35/36 | **35/36** | **36/36** |

Encoder work: 100% (1.00×). No splice bug.

### F=12 (full freeze — legacy)

| Set | tool-name warm | byte-identical |
|-----|---------------:|---------------:|
| 29-case battery | 9/29 | 7/29 |
| 7-case eval_tools | 2/7 | 2/7 |
| Combined | **11/36** | **9/36** |

Encoder work: ~2.8% of full (≈35.7× saving). Routing collapses to the
capture call `play_tone(sound=chime)` for nearly every non-reference tool —
same failure mode as RESULTS.md Phase 2.

---

## Accuracy-vs-F curve (all 36 cases)

Tool-name accuracy is against the gold label (exact path itself is 35/36:
`brd03` mis-routes under exact). Byte-identical is warm token sequence vs exact.

| F | tool-name | byte-identical | row-layers (mean) | frac | prefill saving |
|--:|----------:|---------------:|------------------:|-----:|---------------:|
| 0 | 35/36 | **36/36** | 3000 / 3000 | 100.0% | 1.00× |
| 4 | 35/36 | **36/36** | 2028 / 3000 | 67.6% | **1.48×** |
| 6 | 35/36 | 33/36 | 1542 / 3000 | 51.4% | **1.95×** |
| 8 | 11/36 | 9/36 | 1056 / 3000 | 35.2% | 2.84× |
| 9 | 11/36 | 9/36 | 813 / 3000 | 27.1% | 3.69× |
| 10 | 11/36 | 9/36 | 570 / 3000 | 19.0% | 5.26× |
| 11 | 11/36 | 9/36 | 327 / 3000 | 10.9% | 9.17× |
| 12 | 11/36 | 9/36 | 84 / 3000 | 2.8% | 35.71× |

### F=6 argument-level divergences (tool name still correct)

Three cases (two unique queries) keep the right tool but corrupt an argument:

```
[sun00 / e7_1] When does the sun set in Reykjavik?
  exact: [{"name":"get_sun_times","arguments":{"location":"Reykjavik"}}]
  warm:  [{"name":"get_sun_times","arguments":{"location":"the sun set"}}]

[lok01] look up the Antikythera mechanism
  exact: [{"name":"look_up","arguments":{"topic":"the Antikythera mechanism"}}]
  warm:  [{"name":"look_up","arguments":{"topic":"the Antikythera"}}]
```

### Cliff between F=6 and F=8

Freezing through layer 6 still lets layers 6–11 recompute schema with the real
query — enough for routing (tool-name matches exact on all 36). Freezing
through layer 8 leaves only layers 8–11 for query→schema mixing; that is past
the routing-critical band (Phase 1 drift: layers 9–13 are where query bleed
into the schema is large). Accuracy collapses to the full-freeze floor.

This lines up with Phase 1 (RESULTS.md): deep schema cosine ≥ 0.999 through
layer 6, then progressive query dependence from layer 7 onward.

---

## Reproduce

```bash
make -C engine needle_host && ./engine/needle_host \
  weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt   # 8/8
make -C engine pie-check
make -C engine needle_warmschema
python3 tools/eval_warmschema.py weights/needle_tools_g512.npk \
  0 4 6 8 9 10 11 12
```

Optional single-F run: `python3 tools/eval_warmschema.py ... 4`  
Or: `./engine/needle_warmschema model.npk vectors.txt 4`

---

## Verdict

**F=4 is the deepest freeze that stays byte-identical to the exact engine** on
the full 36-case set (36/36 token identity; tool-name 35/36 matching exact).
Prefill encoder work drops to 67.6% of full — a **1.48×** row-layer saving.
No F yields gold-label 36/36 tool-name because the base exact path is itself
35/36 (`brd03`).

**F=6** is the accuracy frontier for *routing*: tool-name still 35/36 (matches
exact on every case) with **1.95×** saving, but 3/36 calls diverge at the
argument level (location/topic truncated or substituted). Not safe if argument
fidelity matters.

**F≥8** is not usable: tool-name falls to 11/36, the same collapse as full
freeze (F=12). The hybrid confirms the Phase 1 reading — routing is computed
in the late encoder by query→schema attention — and shows that freezing only
through mid-encoder (F≈4–6) preserves that computation while still cutting
early-layer schema work. The saving at the safe point (F=4, 1.48×) is modest
compared with the full-freeze fantasy (~35×); a board implementation of this
hybrid is only worth it if a ~50% cut of early schema projections is valuable
enough to justify the dual-path encoder. Argument-sensitive serving should
stop at **F=4** (or F=0).
