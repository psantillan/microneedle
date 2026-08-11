#!/usr/bin/env bash
# Every gate that does not need the board.
#
#   ./verify.sh                 with a packed artifact already in weights/
#   ./verify.sh --pack          pack it first (needs $NEEDLE_HOME)
#
# The board-side gate is separate and needs hardware:  python3 tools/run_board_fixtures.py
set -euo pipefail
cd "$(dirname "$0")"
NPK=${NPK:-weights/needle_v3_g512.npk}
fail=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

if [ "${1:-}" = "--pack" ]; then
  step "pack $NPK"
  python3 tools/pack_npk.py --proj int4 --embed int8 --group 512 --out "$NPK"
fi

step "engine builds clean (host, -Wall -Wextra)"
make -C engine clean >/dev/null && make -C engine

step "vector path compiles"
make -C engine pie-check

step "vector path logic (kernels stubbed in C)"
if [ -f "$NPK" ]; then
  ${CC:-cc} -O2 -std=c99 -DNE_PIE -o /tmp/_needle_pie \
      engine/needle_engine.c engine/main_host.c tools/pie_stub.c -lm
  /tmp/_needle_pie "$NPK" weights/vectors_int4p_int8e.txt | tail -2 || fail=1
else
  echo "  no $NPK -- run tools/get_weights.sh (see WEIGHTS.md), or ./verify.sh --pack if you have an upstream checkout"; fail=1
fi

step "host parity against the JAX oracle"
if [ ! -f "$NPK" ]; then
  echo "  no $NPK -- run tools/get_weights.sh (see WEIGHTS.md), or ./verify.sh --pack if you have an upstream checkout"; fail=1
else
  ./engine/needle_host "$NPK" weights/vectors_int4p_int8e.txt || fail=1
fi

step "the vectors file is reproducible from the oracle dump"
python3 tools/ref_to_vectors.py weights/reference_int4p_int8e.json /tmp/_v.txt >/dev/null
if cmp -s weights/vectors_int4p_int8e.txt /tmp/_v.txt; then echo "  identical"; else echo "  VECTORS DRIFTED from oracle dump"; fail=1; fi

step "demo page has no undefined identifiers"
if command -v node >/dev/null; then node web/check_page.mjs web/index.html || fail=1
else echo "  skipped (no node)"; fi

step "browser tokenizer matches the oracle"
if command -v node >/dev/null; then node web/verify_bpe.mjs || fail=1
else echo "  skipped (no node)"; fi

step "packer refuses a layout with no kernel"
# A nonzero exit alone would also pass on ImportError (no numpy) -- require
# the actual refusal message, so a broken environment cannot fake this gate.
_pk_out=$(python3 tools/pack_npk.py --group 64 --out /tmp/_bad.npk 2>&1 || true)
if printf '%s' "$_pk_out" | grep -q "must be 32 or 512"; then
  echo "  refused, as it should be"
else echo "  FAIL: expected the group-size refusal message"; fail=1; fi

printf '\n'
[ "$fail" = 0 ] && echo "all host-side gates passed" || { echo "SOMETHING FAILED"; exit 1; }
