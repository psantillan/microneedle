# The engineering notebook

This directory is the unedited optimization log that took the board from 37 s
prefill / 236 ms-per-token to the shipped numbers. It includes every experiment,
measurement, and rejection with its supporting data. These working documents
were written during the runs. Start with `OPTIMIZATIONS.md` (every optimization,
its pipeline state, and measured effect), then follow its references.

Ledger and integration: `OPTIMIZATIONS.md` (state of every optimization),
`INTEGRATION.md` (merge order, gate outputs, hazard checks).

Experiment records: `RESULTS.md` (warm-schema full freeze — rejected),
`RESULTS-HYBRID.md` (freeze-depth curve), `RESULTS-WARMF4.md` (the shipped
F=4 cache), `RESULTS-INT8.md` (int8 attention feasibility + margins),
`RESULTS-DECMICRO.md` (decoder micro-optimizations), `RESULTS-TOOLRAG.md`
(client-side tool retrieval), `RESULTS-DOCSWEEP.md` (doc corrections with
evidence), `KERNEL-S8.md` (the s8 PIE kernel and its board checklist).

Forward plans: `FINETUNE-NEXT.md` (the two training runs this data
justifies), `PORTING.md` (memory ceiling and the portability boundary).

These documents are the development log as written. Commit hashes, branch
names, and session references belong to the pre-release private history and
intentionally resolve to nothing here. They preserve the original lab notebook.
