#!/usr/bin/env bash
# One-command MicroNeedle fine-tune loop:
#   dataset -> needle.cli finetune -> pack .npk -> eval_tools.py
#
#   export NEEDLE_HOME=~/needle   # required
#   ./tools/finetune.sh
#
# Optional:
#   ./tools/finetune.sh --jsonl path/to/data.jsonl
#   OUT=weights/mine.npk EPOCHS=8 BATCH=4 PER_TOOL=200 ./tools/finetune.sh
#
# Preflight requires a valid NEEDLE_HOME, a JAX-visible GPU, and a training
# JSONL after the dataset step.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

die() {
  printf '\033[1;31merror:\033[0m %s\n' "$*" >&2
  exit 1
}

step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \?//'
  exit 0
}

# Capture env override before applying defaults.
ENV_JSONL="${JSONL:-}"
OUT="${OUT:-$REPO/weights/mine.npk}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-4}"
PER_TOOL="${PER_TOOL:-200}"
CHECKPOINT_DIR_REL="${CHECKPOINT_DIR:-checkpoints}"

JSONL=""
SKIP_DATASET=0

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage ;;
    --jsonl)
      [ $# -ge 2 ] || die "--jsonl needs a path"
      JSONL="$2"; SKIP_DATASET=1; shift 2 ;;
    --out)
      [ $# -ge 2 ] || die "--out needs a path"
      OUT="$2"; shift 2 ;;
    --epochs)
      [ $# -ge 2 ] || die "--epochs needs a value"
      EPOCHS="$2"; shift 2 ;;
    --batch-size)
      [ $# -ge 2 ] || die "--batch-size needs a value"
      BATCH="$2"; shift 2 ;;
    --per-tool)
      [ $# -ge 2 ] || die "--per-tool needs a value"
      PER_TOOL="$2"; shift 2 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

if [ -z "$JSONL" ] && [ -n "$ENV_JSONL" ]; then
  JSONL="$ENV_JSONL"
  SKIP_DATASET=1
fi
if [ -z "$JSONL" ]; then
  JSONL="$REPO/train/tools.jsonl"
fi

# Absolute path: training runs with cwd = NEEDLE_HOME.
case "$JSONL" in
  /*) ;;
  *) JSONL="$REPO/$JSONL" ;;
esac
case "$OUT" in
  /*) ;;
  *) OUT="$REPO/$OUT" ;;
esac

# ── preflight: NEEDLE_HOME ──────────────────────────────────────────────────
step "preflight: NEEDLE_HOME"

if [ -z "${NEEDLE_HOME:-}" ]; then
  die "NEEDLE_HOME is not set.

  Point it at an upstream Needle checkout, then re-run:

    git clone https://github.com/cactus-compute/needle.git \"\$HOME/needle\"
    cd \"\$HOME/needle\" && source ./setup
    export NEEDLE_HOME=\"\$HOME/needle\"

  Then from this repo:

    ./tools/finetune.sh"
fi

if [ ! -d "$NEEDLE_HOME" ]; then
  die "NEEDLE_HOME=$NEEDLE_HOME is not a directory.

  Clone upstream and export NEEDLE_HOME:

    git clone https://github.com/cactus-compute/needle.git \"\$HOME/needle\"
    export NEEDLE_HOME=\"\$HOME/needle\""
fi

if [ ! -f "$NEEDLE_HOME/needle/cli.py" ]; then
  die "NEEDLE_HOME=$NEEDLE_HOME does not look like a Needle checkout
  (missing needle/cli.py).

  Expected layout after:

    git clone https://github.com/cactus-compute/needle.git
    cd needle && source ./setup"
fi

# Prefer the upstream venv python (jax, flax, needle package live there).
if [ -x "$NEEDLE_HOME/.venv/bin/python" ]; then
  NEEDLE_PY="$NEEDLE_HOME/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  NEEDLE_PY="$(command -v python3)"
else
  die "no python3 on PATH and no \$NEEDLE_HOME/.venv/bin/python.
  From the upstream checkout:  cd \"\$NEEDLE_HOME\" && source ./setup"
fi

# Editable install puts needle on sys.path; bare checkouts need PYTHONPATH.
export PYTHONPATH="${NEEDLE_HOME}${PYTHONPATH:+:$PYTHONPATH}"

if ! "$NEEDLE_PY" -c "import needle.cli" 2>/dev/null; then
  die "python at $NEEDLE_PY cannot import needle.cli.

  From the upstream checkout install the package and deps:

    cd \"\$NEEDLE_HOME\" && source ./setup

  Entry point is needle.cli:main (python -m needle.cli finetune ...),
  not python -m needle."
fi

echo "  NEEDLE_HOME=$NEEDLE_HOME"
echo "  python=$NEEDLE_PY"

# ── preflight: JAX sees a GPU ───────────────────────────────────────────────
step "preflight: JAX GPU"

set +e
GPU_ERR=$("$NEEDLE_PY" -c "
import sys
try:
    import jax
except ImportError as e:
    print('jax is not installed in this python:', e)
    print('  Fix: cd \"\$NEEDLE_HOME\" && source ./setup')
    print('  (setup installs jax[cuda12] when nvidia-smi is present)')
    sys.exit(1)
backend = jax.default_backend()
devices = jax.devices()
print('  jax backend=%s  devices=%s' % (backend, devices))
if backend != 'gpu':
    print('JAX default backend is %r, need gpu.' % (backend,))
    print('  Fine-tuning on CPU is impractically slow for this loop.')
    print('  Checks:')
    print('    nvidia-smi                          # driver sees a CUDA GPU')
    print('    cd \"\$NEEDLE_HOME\" && source ./setup  # reinstalls jax[cuda12]')
    print('  On Windows, run under WSL2 (JAX CUDA wheels are Linux-only).')
    print('  If you SSH into WSL2, hold the session open (tmux/screen):')
    print('  a backgrounded job dies when the SSH session ends.')
    sys.exit(1)
" 2>&1)
GPU_RC=$?
set -e
printf '%s\n' "$GPU_ERR"
[ "$GPU_RC" -eq 0 ] || die "JAX does not see a GPU (see above)."

# ── dataset ─────────────────────────────────────────────────────────────────
if [ "$SKIP_DATASET" -eq 0 ]; then
  step "dataset: train/make_dataset.py --per-tool $PER_TOOL"
  python3 "$REPO/train/make_dataset.py" --per-tool "$PER_TOOL" --out "$JSONL"
else
  step "dataset: using existing $JSONL"
fi

if [ ! -f "$JSONL" ]; then
  die "dataset does not exist: $JSONL

  Generate the six-tool demo corpus:

    python3 train/make_dataset.py --per-tool 200 --out train/tools.jsonl

  Or pass your own JSONL:

    ./tools/finetune.sh --jsonl /path/to/your.jsonl

  Each line needs query / tools / answers (tools and answers are JSON inside strings).
  See docs/finetuning.md."
fi

NLINES=$(grep -c . "$JSONL" || true)
if [ "${NLINES:-0}" -lt 3 ]; then
  die "dataset $JSONL has fewer than 3 examples (needle finetune refuses that).
  Need ~120 examples per tool for a real run; the demo generator uses --per-tool 200."
fi
echo "  $JSONL  ($NLINES lines)"

# ── train ───────────────────────────────────────────────────────────────────
step "train: python -m needle.cli finetune ($(basename "$JSONL")) --epochs $EPOCHS --batch-size $BATCH"

CKPT_DIR="$NEEDLE_HOME/$CHECKPOINT_DIR_REL"
mkdir -p "$CKPT_DIR"
STAMP="$CKPT_DIR/.finetune_sh_stamp_$$"
touch "$STAMP"

# Run from NEEDLE_HOME so relative paths in the trainer resolve as upstream expects.
(
  cd "$NEEDLE_HOME"
  "$NEEDLE_PY" -m needle.cli finetune "$JSONL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --checkpoint-dir "$CKPT_DIR"
)

BEST="$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'needle_finetuned_*_best.pkl' -newer "$STAMP" 2>/dev/null | sort | tail -n 1 || true)"
rm -f "$STAMP"

if [ -z "$BEST" ]; then
  BEST="$(ls -t "$CKPT_DIR"/needle_finetuned_*_best.pkl 2>/dev/null | head -n 1 || true)"
fi

if [ -z "$BEST" ] || [ ! -f "$BEST" ]; then
  die "no needle_finetuned_*_best.pkl found under $CKPT_DIR after training.

  Look for checkpoints the trainer wrote and pack manually:

    python3 tools/pack_npk.py --checkpoint <best.pkl> \\
        --proj int4 --embed int8 --group 512 --out weights/mine.npk"
fi
echo "  best checkpoint: $BEST"

# ── pack ────────────────────────────────────────────────────────────────────
step "pack: tools/pack_npk.py -> $OUT"

mkdir -p "$(dirname "$OUT")"
python3 "$REPO/tools/pack_npk.py" \
  --checkpoint "$BEST" \
  --proj int4 --embed int8 --group 512 \
  --out "$OUT"

# ── eval ────────────────────────────────────────────────────────────────────
step "eval: tools/eval_tools.py $OUT"

if [ ! -x "$REPO/engine/needle_host" ]; then
  echo "  building engine/needle_host (needed by eval_tools.py)..."
  make -C "$REPO/engine"
fi

python3 "$REPO/tools/eval_tools.py" "$OUT"

printf '\n\033[1mdone\033[0m\n'
echo "  checkpoint: $BEST"
echo "  artifact:   $OUT"
echo "  The trainer's own eval is not the board path — only the score above counts."
echo "  Flash when satisfied:"
echo "    esptool.py --chip esp32p4 -p /dev/ttyACM1 -b 921600 write_flash 0x210000 $OUT"
