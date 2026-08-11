# Fine-tuning Needle for your tools

The stock 26M-parameter model routes unseen tools unreliably. Fine-tune it for
*your* tools.

Build a JSONL corpus, train with upstream Needle, pack a `.npk`, score it with
the shipped C engine, then flash. Upstream owns the trainer. This repo owns the
data shape, packer, board-matched eval, and one-command loop.

---

## What you need

- **Any CUDA GPU** and a driver visible to `nvidia-smi`. CPU works in theory;
  this loop assumes GPU.
- **Linux, or Windows under WSL2.** JAX's CUDA wheels are Linux-only. On Windows,
  use WSL2, where `nvidia-smi` works and JAX finds the GPU without extra
  configuration.
- **An upstream Needle checkout** at `$NEEDLE_HOME`:

  ```bash
  git clone https://github.com/cactus-compute/needle.git
  cd needle && source ./setup
  export NEEDLE_HOME="$PWD"    # put this in your shell profile
  ```

- **This repo** (MicroNeedle), for `train/make_dataset.py`, `tools/pack_npk.py`,
  `tools/eval_tools.py`, and `tools/finetune.sh`.

**WSL2 / SSH:** if you SSH into WSL2, launch training in a held session
(`tmux`, `screen`, or keep the terminal open). A backgrounded job dies when the
SSH session ends.

---

## One-command path

From this repo, with `$NEEDLE_HOME` set and a GPU visible to JAX:

```bash
./tools/finetune.sh
```

The loop is:

```text
python3 train/make_dataset.py --per-tool 200 --out train/tools.jsonl
  ->  cd $NEEDLE_HOME && python -m needle.cli finetune <jsonl> --epochs 8 --batch-size 4
  ->  python3 tools/pack_npk.py --checkpoint <best.pkl> --proj int4 --embed int8 --group 512 --out weights/mine.npk
  ->  python3 tools/eval_tools.py weights/mine.npk
```

The entry point is `needle.cli:main`. Invoke it as `python -m needle.cli finetune ...`,
not `python -m needle`.

Optional knobs:

```bash
./tools/finetune.sh --jsonl path/to/your.jsonl   # skip make_dataset
./tools/finetune.sh --epochs 8 --batch-size 4 --out weights/mine.npk
EPOCHS=8 BATCH=4 PER_TOOL=200 OUT=weights/mine.npk ./tools/finetune.sh
```

Preflight requires a valid Needle checkout at `NEEDLE_HOME`, JAX's `gpu`
backend, and a training JSONL with at least three lines after the dataset step.
Failures say what to run next.

Then flash the scored artifact:

```bash
esptool.py --chip esp32p4 -p /dev/ttyACM1 -b 921600 \
    write_flash 0x210000 weights/mine.npk        # ~3 min
```

---

## Warnings

**The trainer's own eval is misleading.** Its checkpoint score uses a different
generation path from the board. **Only the packed artifact scored by
`tools/eval_tools.py` counts**: the same `.npk`, int4 kernels, and C engine that
you will flash. Do not flash on the trainer's call_f1 alone.

**The checkpoint reader has three silent traps.** `tools/pack_npk.py --checkpoint`
handles them; errors are silent:

1. Layers are **stacked on axis 0** (one `(12, 512, 512)` array, not twelve).
2. Kernels are **`[in, out]`** the JAX way, not `[out, in]`.
3. `contrastive_*` / `log_temp` are an unused speech head. Drop them; do not
   treat them as model weights.

To prove the reader on your own checkout, pack the *base* model from its `.pkl`
and diff it against the safetensors export:

```bash
python3 tools/pack_npk.py --checkpoint $NEEDLE_HOME/checkpoints/needle.pkl \
                    --verify-against $NEEDLE_HOME/checkpoints/model.safetensors \
                    --proj int4 --embed int8 --group 512 --out /tmp/check.npk
```

Expect `228 tensors, max |diff| 3.125e-02`, exactly the bf16 quantum at the
largest weight's magnitude. The difference is the safetensors round-trip. The
stronger check is `/tmp/check.npk` scoring 8/8 on the parity fixtures.

---

## What the trainer reads

`needle.cli finetune` takes JSONL. Three string fields per line; `tools` and
`answers` are JSON *inside* strings:

```json
{"query":"What's the weather in Oslo?","tools":"[{\"name\":\"get_weather\",...}]","answers":"[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Oslo\"}}]"}
```

Corpus constraints:

- It refuses fewer than 3 examples, and warns below **120 per tool**
  (100 train / 10 val / 10 test).
- Tool names are snake_cased before tokenization. `getWeather` in your tools
  list must be `get_weather` in `answers`. `Needle.prompt()` and `tools/pack_npk.py`
  apply the same rule. Don't hand-roll it.
- `NE_MAX_ENC` is 384 tokens. A prompt is your query plus the full JSON of every
  offered tool. Verbose schemas consume the budget.

## What the data has to do

At 26M parameters, the corpus determines most routing quality. Four properties
matter:

- **Vary offered tools per example.** A fixed menu teaches position instead of
  selection. Sample a different 2–4 tool subset each time, including the right
  answer.
- **Include negatives.** Examples where nothing fits, with `answers` of `"[]"`.
  Otherwise the model learns that every request needs a call and answers
  "write me a poem" with a weather lookup.
- **Pin argument types by example.** A corpus that is inconsistent about
  `{"minutes": 3}` versus `{"minutes": "3 minutes"}` produces the string form.
- **Vary phrasing, including terse and indirect.** "weather, Oslo" and "do I
  need a jacket in Oslo" should both land.

`train/make_dataset.py` demonstrates all four for six tools. It is a template,
not a dependency:

```bash
python3 train/make_dataset.py --per-tool 200 --out train/tools.jsonl
```

## Manual steps (if you are not using the script)

```bash
python3 train/make_dataset.py --per-tool 200 --out train/tools.jsonl

cd $NEEDLE_HOME
python -m needle.cli finetune /absolute/path/to/train/tools.jsonl \
    --epochs 8 --batch-size 4
# -> checkpoints/needle_finetuned_<stamp>_best.pkl

cd /path/to/microneedle
python3 tools/pack_npk.py --checkpoint $NEEDLE_HOME/checkpoints/needle_finetuned_..._best.pkl \
                    --proj int4 --embed int8 --group 512 \
                    --out weights/mine.npk

python3 tools/eval_tools.py weights/mine.npk      # score it before flashing
```

## What forces a firmware rebuild

A fine-tune retaining the base architecture needs only a repack and reflash.
Changes to these compile-time values require a firmware rebuild:

- `NE_MAX_ENC` 384, `NE_MAX_GEN` 64, `NE_VOCAB` 8192.
- The architecture constants in `needle_engine.h` — 12 encoder layers, 8
  decoder, `d_model` 512, 8 query / 4 KV heads.
- `--group` is 32 or 512 only. The packer and `ne_load` reject other layouts
  because no kernel supports them.

## Scoring the result

`tools/eval_tools.py` runs seven cases with all six demo tools offered together,
including one request no tool can serve. It prints each case's selected tool:

```bash
python3 tools/eval_tools.py weights/mine.npk
```

The board's shipped artifact scores 7/7. It needs `engine/needle_host` built
(`make -C engine`); `tools/finetune.sh` builds it if missing.

---

## Troubleshooting

### `NEEDLE_HOME is not set`

`./tools/finetune.sh` preflight with `NEEDLE_HOME` unset:

```text
== preflight: NEEDLE_HOME
error: NEEDLE_HOME is not set.

  Point it at an upstream Needle checkout, then re-run:

    git clone https://github.com/cactus-compute/needle.git "$HOME/needle"
    cd "$HOME/needle" && source ./setup
    export NEEDLE_HOME="$HOME/needle"

  Then from this repo:

    ./tools/finetune.sh
```

### JAX does not see a GPU

Preflight prints the backend and fix. Typical causes are no NVIDIA driver,
CPU-only jax, or Windows without WSL2. Re-run `source ./setup` inside
`$NEEDLE_HOME` after `nvidia-smi` works.

### Training dies mid-run over SSH / WSL2

A backgrounded job dies with the SSH session. Use `tmux`, `screen`, or keep the
session attached during training.

### `cannot import needle.cli`

In the upstream checkout, `source ./setup` creates `.venv`, runs `pip install -e .`,
and installs the matching JAX build. Invoke training as
`python -m needle.cli finetune ...`, not `python -m needle`.

### Trainer metrics look great, board routes badly

Pack the best checkpoint and run `tools/eval_tools.py` on the `.npk`. Only that
score uses the board path.
