# Contributing

## Scope

**No prior thread needed:** ports, kernels, tools, fixtures, and docs.

**Open an issue first:** engine semantics changes, or anything that touches
certification (token-identity of the acceptance fixtures, the packed `.npk`
layout, or the oracle dump).

## Rules

1. `./verify.sh` must pass. If you do not have `weights/needle_v3_g512.npk`,
   pack it first (`./verify.sh --pack`, needs
   `$NEEDLE_HOME`), or fetch it with `./tools/get_weights.sh`.
2. Byte-identity changes — different tokens from the host harness, a different
   `.npk` layout, or a regenerated oracle — require re-certification. Say so
   in the PR and do not land them as drive-by cleanups.

## Dev setup

```
# gcc, python3, numpy, node
python3 -m pip install -r requirements.txt   # numpy (+ optional host extras)
./tools/get_weights.sh --from-dir /path/to/npks   # or RELEASES_BASE_URL=...
make -C engine && ./verify.sh
```
