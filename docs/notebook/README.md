# The engineering notebook

This directory is the complete, unedited log of the optimization program that
took the board from 37 s prefill / 236 ms-per-token to the shipped numbers:
every experiment, every measurement, and — deliberately — every rejection,
each with the data that killed it. Files are working documents written during
the runs they describe; they trade polish for fidelity. Start with
`OPTIMIZATIONS.md` (the ledger: every optimization, its pipeline state, and
its measured effect) and follow its references inward.

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

A note on fidelity: these documents are the development log as written.
Commit hashes, branch names, and session references in them belong to the
pre-release private history and intentionally resolve to nothing here —
they are retained because an edited lab notebook is worth less than an
honest one.
