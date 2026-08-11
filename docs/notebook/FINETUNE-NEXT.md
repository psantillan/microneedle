# FINETUNE-NEXT: what belongs in the next training runs

The boundary between engine work and training work is the .npk (docs/finetuning.md).
This file is the other side of that boundary: changes the mechanistic program
(tools/probe_latent_audio.py, tools/ablate_heads.py, the branch experiments)
has shown belong at the fine-tune step, split into a safe data run and a
research run so the safe fixes are never hostage to an architecture bet.
Every run produces a new .npk and therefore a full re-certification: new
fixture dump, 3 board runs diffed, battery re-scored.

## Run A — data only, one retrain, low risk

1. **Retrieval-shaped schemas.** Sample k in {1..6} tools per example,
   shuffled order, plus deliberate misses whose label is `[]`. Rationale:
   client-side tool retrieval (branch `tool-retrieval`) sends the model
   pruned schemas; pruning must be in-distribution. The causal probes say the
   model already tolerates shuffle and remove — this is insurance, and if the
   toolrag battery routes 36/36 on pruned schemas with the current npk, this
   item shrinks to "add near-miss distractor sets" (schemas containing only
   wrong-but-plausible tools, label `[]`).
2. **brd03-class phrasing.** "What chip are you running on?" misroutes to
   play_tone with a +0.386 logit margin — a coin flip, and int8 noise tips
   it (RESULTS-INT8). This is a data gap: board_status training phrasings
   are uptime/memory-shaped, not identity-shaped. Add device-identity
   phrasings ("what chip/board/hardware are you", "what am I talking to").
   Cheapest certain win in this file.

That is the whole of run A. The argument enums are closed sets; there is
nothing to diversify there.

## Rejected, with reasons (so they stay rejected)

- **Functional tokens** (Octopus-style: one special token per tool).
  Rejected: it deletes the copy mechanism. The traced router reads the tool
  name out of the schema at inference time (rename play_tone -> make_noise
  in the schema and the model emits make_noise), which is why unseen tools
  and pruned schemas work at all. Functional tokens would also optimize
  decode, which is not the cost center; prefill is.
- **Quantization-aware training for int8 attention.** Rejected: measured
  unnecessary. int8 ranges are 8/8 byte-identical on fixtures; decision
  margins average ~12 logits against a worst adverse shift of -0.66
  (RESULTS-INT8). Do not train for a problem the margins already absorb.

## Run B — research retrain, separate, may fail

3. **Asymmetric encoder mask: schema tokens blind to the query.** Today the
   encoder is bidirectional, and the measured query->schema bleed in layers
   9-13 IS the router (branch `schema-cache`: freezing schema states
   collapses routing 28/29 -> 9/29). Masking schema-side attention to the
   query during fine-tuning would make schema states query-independent BY
   CONSTRUCTION — turning warm-schema caching from a failed approximation
   into an exact, bit-identical cache: prefill for a known schema collapses
   to query-only work (~35x on the demo prompt). The open question is
   whether a 26M no-FFN model can relearn routing on the decoder side of
   the mask. Calibration step before any GPU time: apply the mask at
   inference to the CURRENT model (one host flag). Expected result is total
   collapse; that confirms the retrain is asking the model to relocate
   routing, not preserve it, and prices the bet honestly.
4. **Prune-then-heal.** Blocked on the 224-head ablation matrix
   (tools/ablate_heads.py). If the matrix yields a real do-nothing head
   list, prune those heads and fine-tune the remainder to recover. Joins
   run B only with the matrix in hand; a pruned model is a new architecture
   and re-certifies like one.
