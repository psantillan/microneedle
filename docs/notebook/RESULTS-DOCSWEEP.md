# RESULTS-DOCSWEEP

Branch: `doc-sweep` (from `integration`). Documentation sweep only — comment
edits in `.c`/`.h`, plus the allowed `demos/ask.py` exception. No board access.

## Gates

| Gate | Result |
|---|---|
| `make -C engine needle_host` | clean |
| `./engine/needle_host weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt` | **8/8** |
| `make -C engine pie-check` (NE_PIE and NE_PIE+NE_ATTN_I8) | clean |
| `demos/ask.py` compiles; id path = `bpe.build_encoder_input` | OK (host-only proof; no board) |

## Claims fixed

### 1. README limitations: encoder attention "still scalar fp32" / 40 s

| | |
|---|---|
| **Was** | `README.md` Limitations: "Encoder attention is blocked and cache-friendly but still scalar fp32. A 271-token prompt takes 40 s." |
| **Evidence stale** | Engine path under `NE_PIE` is integer attention (`attend` → `ne_attn_heads` / `pie_dots_s16`, dual-core via `ne_split_attn`); certified table already has prefill **5.9 s at 271** and decode **94–130 ms/token**. `RESULTS-DECMICRO.md`, `RESULTS.md`, `KERNEL-S8.md`, `RESULTS-TOOLRAG.md` are host-verified, board pending. |
| **Fix** | Replaced with int16 dual-core fact, last board-certified numbers, and an explicit "host-verified, board pending" list for dec-micro / grammar skip / s8 / tool retrieval. |

### 2. `pie_dots_s16` header: QACC "the next real lever"

| | |
|---|---|
| **Was** | `engine/needle_engine.h` ~L225–230: eight-lane QACC "is the next real lever". |
| **Evidence stale** | `bench/gemm/README.md` "Eight dot products per pass, with QACC": in-engine prefill 6365→6451 ms (stride from 64-byte QACC drain). `pie_dots_s8` landed under `NE_ATTN_I8` (`KERNEL-S8.md`). |
| **Fix** | Documented recorded negative; pointed successor at `pie_dots_s8` (host-proven, board pending). |

### 3. README intro vs table: 0.42 s vs 0.38 s

| | |
|---|---|
| **Was** | Intro "about 0.42 s"; Results table "0.38 s at 22 tokens". |
| **Evidence stale** | Same file disagreement; table is the certified figure. |
| **Fix** | Intro now says **0.38 s** to match the table. |

### 4. Scheduler-guard "~40 ms worst case" / every-token implication

| | |
|---|---|
| **Was** | Several places state ~40 ms for the logits GEMV; some implied a whole-query multi-tens-of-seconds hold from the pre-int-attn era. |
| **Evidence** | Guard bound for the logits GEMV is still ~40 ms when it runs (`fw_main.c`, `needle_engine.h`). Grammar skip (`ne_step_f`, `RESULTS.md`) removes that GEMV on forced steps — so the claim is stale only if it says every token pays it. `firmware/main/net.c` still said "up to 41 s" / "40 s queue" for whole-query hold (pre-int-attn). |
| **Fix** | `needle_engine.h`: clarify forced steps skip the 40 ms window. `README.md` Limitations: same. `net.c`: whole-query hold → board-certified ~5.9 s prefill + decode; queue wording de-staled. `fw_main.c`: note grammar skip is host path today. Bound kept where it still applies. |

### 5. `bench/vector/README`: no pointer to `bench/gemm`

| | |
|---|---|
| **Was** | No link to the successor instrument. |
| **Fix** | One paragraph pointing at `bench/gemm`. |

### 6. `bench/gemm/README`: "the fix is to split by query range"

| | |
|---|---|
| **Was** | Early "Attention was over the cliff" section prescribed query-range split + fuse-first. |
| **Evidence stale** | Shipped path is **head interleave** in `ne_attn_heads` (`engine/needle_engine.c` L1390–1396); later gemm README sections already recorded the interleave / QACC negative. |
| **Fix** | Early section rewritten to what shipped; query-range left as the unused alternative. |

### 7. GEMV comment: decode time "nearly all" in gemv / logits

| | |
|---|---|
| **Was** | `engine/needle_engine.c` gemv block: "every generated token spends nearly all its time here". Cross-KV alloc comment still cited "230 ms/token" and "two thirds". |
| **Evidence stale** | Board decode is 94–130 ms/token; grammar skips logits GEMV on forced steps; dec-micro moved self-attn off pure gemv (`RESULTS-DECMICRO.md`). Absolute 230 ms/token is pre-int16-attention. |
| **Fix** | Softened gemv to "hot loop for the projection path" with explicit outsides + grammar skip. Cross-KV comment drops the 230 ms absolute; keeps qualitative long-prompt dominance note as unremeasured. |

### 8. `demos/ask.py` hand-built encoder prompt

| | |
|---|---|
| **Was** | Hand-built `[query] + tools_id + tools_json` without snake_case / shared truncation. |
| **Evidence stale** | `tokenizer/bpe.py::build_encoder_input` is the contract used by `eval_tools`, web, tool retrieval. `FINETUNE.md` did not point at `ask.py` (no doc-risk edit needed there). |
| **Fix** | `ask.py` now calls `bpe.build_encoder_input`. Host proof: builds ids for a sample query; file compiles. Board not available. |

### 9. `web/index.html` served perf copy

| | |
|---|---|
| **Was** | "one tool is under a second and a big catalogue is 7 s. Decode sits around 100 ms/token"; worker comment "7-second prefill". |
| **Evidence stale** | Certified: 0.38 s / 5.9 s @271 / 94–130 ms/token. |
| **Fix** | Aligned to certified figures; worker text generalized to "multi-second prefill". |

### 10. PORTING.md / OPTIMIZATIONS.md cross-links

| | |
|---|---|
| **Was** | README did not mention either file. |
| **Note** | Both exist on `master`; not present on this `integration`-based worktree (OPTIMIZATIONS.md owned by another process — not restored/edited). |
| **Fix** | README Limitations: one line each that **PORTING.md** and **OPTIMIZATIONS.md** are the porting notes / optimization ledger. Files themselves untouched. |

## Engine comment sweep (other hits)

Searched `engine/*.{c,h}` for `next`, `TODO`, `still`, `worst case`, `scalar`.

| Hit | Action |
|---|---|
| QACC "next real lever" | Fixed (#2). |
| Guard "worst case ~40 ms" | Fixed (#4) — bound kept, every-token implication removed. |
| GEMV "nearly all" / "230 ms/token" | Fixed (#7). |
| `attend_seq` "fp32 scalar … 30 s down to 6 s" | **Left.** Describes the scalar `attend_seq` path and the historical int16 win; still true for `#else` / host scalar. |
| "measured worst case in the fixture set: 271" | **Left.** Still true of fixture lengths. |
| Remaining "scalar path" / "still" wording | **Left.** Correct as written (host reference, residual, etc.). |
| No `TODO` in engine headers/source | — |

## Doubtful (listed, not guessed)

1. **Post-merge board numbers** for dec-micro, grammar skip, s8, tool retrieval — host-verified only; not invented in prose.
2. **Firmware still calls `ne_step` in a loop** (`fw_main.c::infer`), not `ne_generate_cb` — grammar force is not on the board path until that wires up. Comments now say host path where relevant.
3. **Cross-attention share of decode** after int16 attention — qualitative dominance kept; no new percentage invented.
4. **PORTING.md / OPTIMIZATIONS.md** content not re-copied onto this branch (OPTIMIZATIONS owned elsewhere). README only mentions they exist.
5. **`analysis/`** still referenced from README Limitations; directory is gitignored here — left as-is (pre-existing).

## Files touched

- `README.md`
- `engine/needle_engine.h` (comments)
- `engine/needle_engine.c` (comments)
- `bench/vector/README.md`
- `bench/gemm/README.md`
- `firmware/main/net.c` (comments)
- `firmware/main/fw_main.c` (comments)
- `web/index.html`
- `demos/ask.py` (allowed functional exception)
- `RESULTS-DOCSWEEP.md` (this file)
