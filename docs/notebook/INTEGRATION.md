# INTEGRATION: dec-micro + int8-attn on top of master

Branch `integration`, built from master a99ef9e. Host-verified only — the
board has NOT been flashed from this branch. Merge order and every gate
output below.

## Merge order

1. `dec-micro` (82cf1fa) — requant hoist + int16 decoder self-attention.
   Git auto-merged (f44f8ab); no textual conflicts.
2. **Ported the uncommitted NE_TRACE ablation machinery** (c737cd6).
   The trace_ablate hooks, NE_ABLATE env parsing, ne_trace_dump_weights,
   and tools/ablate_heads.py existed only as UNCOMMITTED changes in the
   master worktree — master's a99ef9e does not contain them. Applied that
   diff here (3-way; one conflict in ne_step where dec-micro's int16
   self-attention branch and the self-attn trace_ablate hook overlap —
   resolved by keeping dec-micro's #ifdef NE_PIE/else structure and placing
   `trace_ablate(merged, 1, NE_DMODEL, 2, L)` after the closing brace, the
   same shape as the cross-attention hook).
   **Action for the integrator: commit those changes on master too, or
   master's working tree and this branch will drift.**
3. `int8-attn` (91a007a) — NE_ATTN_I8 ranges + measure harness. Clean merge
   (also brought the needle_i8 / needle_i8_trace / needle_trace Makefile
   targets forward).

Weights are untracked; the .npk files are symlinked from the master
worktree's weights/ directory (weights/*.npk symlinks, not copies).

## Gate results (all run on this tree, verbatim summary lines)

After dec-micro + machinery, and re-run after int8 merge — identical:

| Gate | Result |
|---|---|
| needle_host fixtures (v3 npk) | `8/8 cases match the JAX oracle token-for-token` |
| stub-parity NE_PIE build (exercises attend_self_q16 + requant hoist) | `8/8 cases match the JAX oracle token-for-token` |
| pie-check | 0 warnings/errors |
| needle_trace build | 0 warnings/errors |
| eval_tools (tools npk) | `7/7 correct with ALL SIX tools offered (250 token prompt)` |

## The hazard check (the point of this branch)

NE_ATTN_I8 redefines NE_AQ/NE_PQ to 127/127; dec-micro's attend_self_q16
uses those constants, so the merged int8 build quantizes decoder
self-attention to int8 too — territory the pre-merge battery never
measured. Measured here:

| Metric | Pre-merge (int8 branch) | Merged tree |
|---|---|---|
| int8 fixtures | 8/8 | **8/8** |
| Battery name agreement | 35/36 | **35/36** |
| Byte-identical sequences | 35/36 | **35/36** |
| Winner flips | 1 (brd03, exact wrong → i8 right) | **1 (same case, same direction)** |
| Margin shift mean / min / max / std | +0.639 / −0.656 / +2.106 / 0.622 | **+0.608 / −0.746 / +2.125 / 0.637** |

**Verdict: benign.** int8 decoder self-attention rides along within noise.
No range decoupling needed; NE_ATTN_I8 covering all three attention sites
is now the measured configuration, and the s8 kernel plan can treat
decoder self-attention as in-scope if it ever wants to.

## Instrument checks on the merged tree

- probe_latent_audio lens table reproduces the session's numbers exactly:
  `play` rank 0 from cross5, margin +12.637 at cross7, lens-vs-engine
  max |diff| 4.1962e-05.
- Ablation smoke: NE_ABLATE="enc:0:0" flips brd03 (play_tone → look_up),
  unset matches baseline byte-for-byte. (aud00 does NOT flip under this
  ablation — consistent with the sweep's finding that single-head encoder
  ablations only break the fragile case.)

## Remaining for the human integrator

(The handoff list above is superseded by the completion section below.)

---

# Completion pass: grammar-bake + tool-retrieval + s8-kernel

Merge order and resolutions:

1. `grammar-bake` (576b608 -> 51bdd1d). Conflicts: engine/Makefile and
   .gitignore only (target lists) -- resolved as unions. Engine sources
   auto-merged; the DFA lives in the generation loop and does not touch the
   trace hooks or dec-micro's ne_step edits.
2. `tool-retrieval` (dc1002c -> b77279c). Clean merge, web/ + tools/ only.
3. `s8-kernel` (7e4867b -> ea2c2b4 + 59ecc16). Auto-merge compiled with
   incompatible-pointer warnings under NE_ATTN_I8: the s8 refactor retypes
   the shared quantizers to ne_aq_t while dec-micro's self path passes
   int16_t buffers. Resolution (59ecc16): decoder self-attention keeps
   int16 STORAGE and pie_dots_s16 in both flag states, and its quantization
   RANGES follow the flag -- int16-pinned twins quant_row_i16/quant_vT_i16
   compile only under NE_ATTN_I8 and alias to the shared helpers otherwise.
   Tried first: pinning self to 4096/16384 under the flag -- that is a
   never-measured hybrid and it flipped the marginal no_matching_tool
   fixture (7/8). The certified configuration (flag ranges reaching self,
   from the c932227 hazard check) restores 8/8.

## Gate scoreboard (merged tree, all run on this worktree)

| Gate | Result |
|---|---|
| scalar needle_host fixtures | **8/8** |
| stub-parity needle_pie fixtures | **8/8** |
| needle_i8 (NE_PIE + NE_ATTN_I8) fixtures | **8/8** |
| pie-check, both flag states | clean |
| eval_tools | **7/7** |
| measure_attn_i8 | **35/36** name + byte-identity; single flip brd03 (exact wrong -> i8 correct, margins +0.386 -> +0.205); margin shift mean +0.640 / min -0.551 / max +1.995 (pre-merge: +0.639 / -0.656) |
| tool_retrieval | recall@{1,2,3} 36/36; mutation PASS (9/9 -> 6/9 with stripped keywords); twin PASS; k=2 end-task fixes brd03 |
| probe_latent_audio lens | reproduces exactly: play rank 0 from cross5, +12.637 at cross7, lens-vs-engine 4.196e-05 |
| bench_grammar | builds and runs; see timing below |
| xesppie firmware build (flag off) | **clean** -- build/needle_fw.bin produced, board untouched |
| xesppie bench/gemm build | **clean** |

## bench_grammar (host wall clock, 20 reps x 7 eval cases)

Host `bench_grammar` against `weights/needle_tools_g512.npk`, 20 reps × 7
eval_tools vectors.

```
grammar_force=0  cases=7  reps=20  tokens=1860  skips=0  cpu_s=518.327
grammar_force=1  cases=7  reps=20  tokens=1860  skips=360  cpu_s=519.628
```

| Mode | grammar_force | tokens | steps skipped | CPU time (s) |
|---|---|---|---|---|
| without skip | 0 | 1860 | 0 | 518.327 |
| with skip | 1 | 1860 | 360 | 519.628 |
| **delta** | | 0 | **+360 skips** | **+1.3 s (~0.3%, noise)** |

Host end-to-end CPU remains encode-dominated on the 250-token tool
prompt, so the logits skip is lost in run-to-run noise at this scale
(same pattern as the 100-rep gate in RESULTS.md). Board decode-side
fraction should be larger: the 8192-row logits GEMV is the longest
per-step matrix on device.

## Remaining for the human integrator (the board session)

1. Flash bench/gemm from this branch; read the stage_dots_s8 numbers
   (exactness gate prints before timing).
2. Flash firmware (flag off = today's int16 config); 3-run byte-identity
   certification against the current build.
3. If the s8 numbers hold, the NE_ATTN_I8 firmware build is next: certify
   against the needle_i8 HOST reference, not the fp32 one -- brd03
   legitimately flips to board_status under int8.
4. Board timing ledger + README refresh; fill the Board effect column in
   OPTIMIZATIONS.md.
5. Merge integration -> master only after all of the above is green.

---

# Post doc-sweep + warm-f4 merges

Merge order on top of the completion-pass tip (5703873):

1. `doc-sweep` (988cb81) — documentation corrections + `demos/ask.py`
   switch to `bpe.build_encoder_input`. Clean auto-merge (ort).
2. `warm-f4-prod` (21ef57e) — productize warm-schema F=4
   (`ne_schema_capture` / `ne_encode_warm`, opt-in `-DNE_WARMSCHEMA`).

## Pre-merge claim check (warm-f4 vs integration)

`git diff 5703873...21ef57e -- engine/needle_engine.c`: productization
commit body is under `#ifdef NE_WARMSCHEMA` only. The branch also brings
`trace_enc_full` (`encx`) under `#ifdef NE_TRACE` from the earlier
warm-schema experiment lineage (schema drift measurement). Exact
`ne_encode` / default decode path unchanged when both flags are off.

## Conflict resolutions (warm-f4-prod)

| File | Resolution |
|---|---|
| `engine/needle_engine.c` | Keep integration's `ne_trace_dump_weights` + grammar-bake `ne_step_f` / `ne_forced_next`; keep warm-f4 `trace_enc_full` and full NE_WARMSCHEMA product API (capture / warm encode / generate). |
| `engine/Makefile` | Union of targets: `needle_pie` / `needle_i8` / `bench_grammar` and `needle_warmschema` / `pie-check-warm`. |
| `.gitignore` | Add `engine/needle_warmschema` alongside existing host binaries. |
| `RESULTS.md` | Concatenate grammar-bake results + warm-schema research notes (product F=4 gates live in `RESULTS-WARMF4.md`). |
| `engine/needle_engine.h` | Clean auto-merge (grammar + WARMSCHEMA decls). |

## Gate results (re-run on the merged tree)

| Gate | Result |
|---|---|
| `make -C engine needle_host` + fixtures (v3 npk) | **8/8** `cases match the JAX oracle token-for-token` |
| stub-parity `needle_pie` (NE_PIE) fixtures | **8/8** |
| `needle_i8` (NE_PIE + NE_ATTN_I8) fixtures | **8/8** |
| `make -C engine pie-check` (NE_PIE and NE_PIE+NE_ATTN_I8) | clean (0 warnings) |
| `python3 tools/eval_tools.py weights/needle_tools_g512.npk` | **7/7** correct with ALL SIX tools offered |
| `python3 tools/tool_retrieval.py weights/needle_tools_g512.npk` | recall@{1,2,3} **36/36**; mutation=**PASS** (9→6); twin=**PASS**; k=2/3 end-task name-acc 36/36 (full6 still 35/36, brd03) |
| warm-f4: `make -C engine needle_warmschema` + `python3 tools/eval_warmf4.py weights/needle_tools_g512.npk` | battery byte-identical **36/36**; fixture warm parity **8/8**; measured cache 2 489 292 bytes (ns=243); mutation **PASS**; wrong-schema guard **PASS**; prefill ~1.48× row-layers |

### warm-f4 verbatim summary lines

```
OVERALL F=4: tool-name warm 35/36 (exact 35/36)  byte-identical 36/36
encoder work mean 2028/3000 = 67.6% (1.48x saving)
fixture warm parity: 8/8
MUTATE_OK 1
WRONG_SCHEMA_OK 1
  battery byte-identical: 36/36
  mutation:               PASS
  wrong-schema guard:     PASS
```

**Verdict:** both merges host-green. Board still untouched; remaining
work is the board session listed above (flash gemm / firmware, etc.).
