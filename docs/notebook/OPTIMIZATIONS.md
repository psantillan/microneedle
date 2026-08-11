# Optimization ledger

Every optimization from the mechanistic program, its branch, and where it
sits in the pipeline. States, in order:

  implemented -> host-verified (my gates, not the agent's claim)
  -> merged (integration branch) -> board-certified (flash + 3-run
  byte-identity + timing) -> shipped (master, board serving it)

Nothing counts until board-certified; nothing is DONE until shipped.

| Optimization | Branch | State | Measured effect (host) | Board effect |
|---|---|---|---|---|
| Grammar-forced decode (skip logits GEMV on 3 forced steps) | grammar-bake | **merged** (8/8, mutation-tested; brief's 4th rule correctly dropped) | 3 forced steps/call skip the 8192x512 GEMV | pending |
| Decoder requant hoist (1x not 3x per layer) | dec-micro | **merged** | 16 quantize passes/token eliminated | pending |
| int16 decoder self-attention | dec-micro | **merged** | fp32 attend() off the decode path | pending |
| int8 attention ranges (AQ=PQ=127) | int8-attn | **merged**; post-s8 battery re-run: 35/36, brd03 correcting flip, margin mean +0.64 min -0.55 | 8/8 fixtures byte-identical; margins ~12 vs worst shift -0.66 | pending |
| s8 PIE kernel (16-lane fused) | s8-kernel | **merged**; decoder self-attention pinned to int16 storage, ranges follow the flag (the measured config) | all host gates 8/8; asm assembles under xesppie | pending, needs bench/gemm session |
| Client-side tool retrieval (k=2) | tool-retrieval | **merged**; battery re-run on merged tree: recall 36/36, mutation PASS, twin PASS | prefill tokens x0.396, fixes brd03 | pending |
| Layer-split warm schema | warm-hybrid | experiment archived: F=4 gives 1.48x byte-identical on battery; F=6+ degrades argument copy; cliff at F=8 | curve in RESULTS-HYBRID.md | not queued (toolrag shrinks the same cost) |
| Warm schema full freeze | schema-cache | **closed: rejected with evidence** (routing collapse 28/29 -> 9/29) | n/a | n/a |
| Head pruning | (blocked on ablation matrix) | data gathering | TBD | n/a |
| Asymmetric-mask retrain (exact schema cache) | mask-calibration | experiment archived: masking schema-from-query at inference collapses tool-name accuracy 35/36 -> 9/36 (failure shape: empty calls, not wrong calls), so the retrain would relocate routing, not preserve it -- a real architecture bet, priced before any GPU time | n/a | fine-tune tier |
| Run A data fine-tune (retrieval schemas + brd03 phrasings) | FINETUNE-NEXT.md | spec'd from a 120-phrasing margin census: one true coin-flip misroute (identity board_status at +0.39 logits), weak-spot classes = identity phrasings, indirect weather, look-up-framed sun queries; recipe fed the training run | n/a | fine-tune tier |

## Accumulation queue (serial, in order)

All five implementation tracks are merged and gated on `integration`
(INTEGRATION.md has the scoreboard). One step remains before
integration -> master:

1. THE BOARD SESSION (human integrator, one sitting): flash bench/gemm for
   the s8 stage numbers; flash firmware; 3-run byte-identity certification
   against the current build; timing ledger + README refresh. The Board
   effect column above gets filled here and nowhere else.

Rule this file exists to enforce: an optimization that is verified but not
merged is NOT accumulated; check this table before starting anything new.

## Board session results (2026-08-11)

| Item | Board result |
|---|---|
| Merged int16 firmware | certified 3x 8/8 byte-identical (v3 fixtures); again 2x 8/8 after UI rebuild |
| Grammar skips (review fix: fw was not wired) | decode 97-121 -> 89-116 ms/tok |
| s8 kernel (bench/gemm, exactness-gated) | scores 0.403 cyc/MAC (1.5x vs s16), values 0.269 (1.85x) |
| int8 firmware | **ROOT-CAUSED, not a code bug**: the divergent step (no_matching_tool, index 12) has a 0.0097-logit gap between 'se' and 'the' (neighbors: 5-12 logits). fp32 libm/codegen jitter -- which the engine header has always disclaimed as non-identical across platforms -- decides that argmax; host with -ffast-math flips the same fixture with the same scalar stub, exonerating the board asm. int8's coarser probability buckets amplify jitter enough to cross a 0.01 gap; int16 clears it. Board int8 is deterministic (3x stable) and semantically fine. Enabling requires a certification-criterion decision: board-generated references + 3-run self-consistency + a margin audit flagging sub-threshold decisions, instead of host token references on knife-edge steps. Prize: prefill 5911 -> 5428 ms @271 (-8%) plus 0.4 MiB PSRAM (unblocks warm-F4). |
| Tool retrieval k=2 (demo default) | round trip 6.77 s (full-6) -> **2.87 s** measured live |
| runa npk (Run A fine-tune) | **shipped**: under the demo's k=2 retrieval it wins the weak-spot set 13/15 vs the incumbent's 12/15 and fixes the identity misroute in the weights; the known trade is one look_up paraphrase now declining ("What is a quasar?" -> []) |
| warm-F4 | host-verified; memory-blocked: 2.4 MiB cache vs 2.09 MiB PSRAM free (unblocks if int8 lands) |
| dec-micro | in the certified config; wall-clock contribution not isolated (phase-timer session pending) |

## int8 enablement (2026-08-11, criterion change approved)

Certification criterion for the int8 configuration: board-generated
reference (weights/board_reference_i8.json) + 3-run self-consistency +
margin audit (tools/margin_audit.py). Audit results: 4 knife-edge decisions
(<1.0 logit gap) in the 8 fixtures, exactly one of which (0.0097) diverges
on this board's libm; ZERO knife-edges in the demo battery on the runa
model -- the product path is unconditionally safe.

Shipped: NE_ATTN_I8 firmware + runa npk, certified 3x 8/8 vs the board
reference. Board results: prefill 5911 -> 5414 ms @271 (-8%), decode
89-108 ms/tok, demo k=2 round trip 2.81 s, PSRAM free 2141 -> 3709 KB
(warm-F4's 2.4 MiB cache now fits: UNBLOCKED, still opt-in/unwired).

## Warm cache shipped (2026-08-11, second board session)

enc_from_layer refactor: exact path byte-identical through scalar, both
stub configs, and 3x board certification. Warm path now PIE-correct
(quantizers, dual-core phase B, decode mirrors). Firmware: lazy one-slot
capture at schema >= 128 tokens, transparent in infer(), path reported in
the API response. Measured on the board: warm encoder 4799 -> 3754 ms
(-22%), full-schema round trip 6.6 -> 5.6 s; k=2 exact path untouched
(2.82 s). Certification bar: battery (36/36 call-identical on the shipping
npk), not byte-identity -- the cache's rope-position approximation is
inherent and documented. One process lesson recorded in the fixture
runner: references belong to a model; flash it first.

## Addendum: README source records

Two README figures whose measurements predated this ledger's board-session
tables, recorded here so every published number has a source row:

- Full six-tool prompt, cold (int8 + warm firmware, runa pack, /api/tokens
  round trip): **5.81 s** (measured 2026-08-11, brd03 query, 247-token
  prompt; same session as the 2.81 s k=2 figure).
- Prefill at 48 tokens (int16 reference config): **0.73 s** (weather_sf
  fixture, enc_ms 706-727 across certification runs).
