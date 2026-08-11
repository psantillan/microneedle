#!/usr/bin/env bash
# Fetch and sha256-verify the three packed weight artifacts the host gates need.
#
# Usage:
#   RELEASES_BASE_URL=https://github.com/OWNER/REPO/releases/download/TAG \
#       ./tools/get_weights.sh
#   ./tools/get_weights.sh --from-dir /path/to/dir   # offline copy + verify
#
# Artifacts land in weights/ next to this repo's root:
#   needle_v3_g512.npk      base model (parity / board fixtures)
#   needle_tools_g512.npk needle_runa_g512.npk   fine-tune on the six demo tools
#
# Passing transcript (2026-08-11, --from-dir /path/to/local/weights):
#
#   $ ./tools/get_weights.sh --from-dir /path/to/local/weights
#   == weights from /path/to/local/weights
#     needle_v3_g512.npk
#       sha256 27ab3dbf53893649d0943b6716710ce80f99bb7687f12cd64092cb23f40779ba  ok
#     needle_tools_g512.npk
#       sha256 56b7978efe4a9c543db4d14529602c48f14645fe8d21923b516b5a112f08102c  ok
#   weights ready under weights/
#
set -euo pipefail
cd "$(dirname "$0")/.."

V3=needle_v3_g512.npk
TOOLS=needle_tools_g512.npk
V3_SHA=27ab3dbf53893649d0943b6716710ce80f99bb7687f12cd64092cb23f40779ba
TOOLS_SHA=56b7978efe4a9c543db4d14529602c48f14645fe8d21923b516b5a112f08102c
RUNA=needle_runa_g512.npk
RUNA_SHA=1856ad55e935b90b9be20b0a2bf5c245d2bdfc55dc05b8b7ae32e707f0512cfa

OUT=weights
mkdir -p "$OUT"

from_dir=""
if [ "${1:-}" = "--from-dir" ]; then
  from_dir="${2:?--from-dir needs a directory}"
fi

sha_of() { sha256sum "$1" | awk '{print $1}'; }

verify_one() {
  local path="$1" expect="$2"
  local got
  got=$(sha_of "$path")
  if [ "$got" != "$expect" ]; then
    echo "  FAIL sha256 $path" >&2
    echo "    expected $expect" >&2
    echo "    got      $got" >&2
    exit 1
  fi
  echo "      sha256 $got  ok"
}

fetch_one() {
  local name="$1" expect="$2"
  local dest="$OUT/$name"
  if [ -n "$from_dir" ]; then
    if [ ! -f "$from_dir/$name" ]; then
      echo "missing $from_dir/$name" >&2
      exit 1
    fi
    cp -f "$from_dir/$name" "$dest"
  else
    local base="${RELEASES_BASE_URL:-https://github.com/psantillan/microneedle/releases/download/v1.0.0}"
    base="${base%/}"
    echo "    GET $base/$name"
    curl -fsSL -o "$dest" "$base/$name" || {
      echo "download failed: $base/$name" >&2
      echo "see WEIGHTS.md; offline: $0 --from-dir DIR" >&2
      exit 1
    }
  fi
  echo "    $name"
  verify_one "$dest" "$expect"
}

if [ -n "$from_dir" ]; then
  echo "== weights from $from_dir"
else
  echo "== weights from \$RELEASES_BASE_URL (${RELEASES_BASE_URL:-unset})"
fi

fetch_one "$V3" "$V3_SHA"
fetch_one "$TOOLS" "$TOOLS_SHA"
fetch_one "$RUNA" "$RUNA_SHA"
echo "weights ready under $OUT/"
