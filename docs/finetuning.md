# Fine-tuning Needle for your tools

The stock model routes tools it has never seen unreliably. That is not a defect;
it is 26M parameters. If you want it reliable on *your* tools, fine-tune it.

This guide is the path that was proven on real runs: build a JSONL corpus, train
with upstream Needle, pack a `.npk`, score it with the C engine that ships, then
flash. Upstream owns the trainer; this repo owns the data shape, the packer, the
eval that matches the board, and the one-command loop that ties them together.

---

## What you need

- **Any CUDA GPU** and a driver that `nvidia-smi` can see. CPU works in theory;
  the proven loop assumes GPU.
- **Linux, or Windows under WSL2.** JAX's CUDA wheels are Linux-only — on Windows
  that means WSL2, where `nvidia-smi` works and JAX finds the GPU without extra
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

That is the whole proven loop:

```text
python3 train/make_dataset.py --per-tool 200 --out train/tools.jsonl
  ->  cd $NEEDLE_HOME && python -m needle.cli finetune <jsonl> --epochs 8 --batch-size 4
  ->  python3 tools/pack_npk.py --checkpoint <best.pkl> --proj int4 --embed int8 --group 512 --out weights/mine.npk
  ->  python3 tools/eval_tools.py weights/mine.npk
```

The entry point is `needle.cli:main` — invoke it as `python -m needle.cli finetune ...`,
not `python -m needle`.

Optional knobs:

```bash
./tools/finetune.sh --jsonl path/to/your.jsonl   # skip make_dataset
./tools/finetune.sh --epochs 8 --batch-size 4 --out weights/mine.npk
EPOCHS=8 BATCH=4 PER_TOOL=200 OUT=weights/mine.npk ./tools/finetune.sh
```

Preflight refuses to start unless `NEEDLE_HOME` is set and looks like a Needle
checkout, JAX's default backend is `gpu`, and the training JSONL exists (and has
at least three lines) after the dataset step. Failures are loud and tell you
what to run next.

When the script finishes, flash the artifact you just scored:

```bash
esptool.py --chip esp32p4 -p /dev/ttyACM1 -b 921600 \
    write_flash 0x210000 weights/mine.npk        # ~3 min
```

---

## Hard-won warnings (read before you trust a number)

**The trainer's own eval is misleading.** The score printed beside a checkpoint
measures a different generation path than the board runs. **Only the packed
artifact scored by `tools/eval_tools.py` counts** — same `.npk`, same int4
kernels, same C engine you are about to write to flash. Do not flash on the
trainer's call_f1 alone.

**The checkpoint reader has three silent traps.** `tools/pack_npk.py --checkpoint`
handles them; they are silent if you get them wrong, which is why this is code
rather than a sticky note:

1. Layers are **stacked on axis 0** (one `(12, 512, 512)` array, not twelve).
2. Kernels are **`[in, out]`** the JAX way, not `[out, in]`.
3. `contrastive_*` / `log_temp` are an unused speech head — drop them, do not
   treat them as model weights.

To prove the reader on your own checkout, pack the *base* model from its `.pkl`
and diff it against the safetensors export:

```bash
python3 tools/pack_npk.py --checkpoint $NEEDLE_HOME/checkpoints/needle.pkl \
                    --verify-against $NEEDLE_HOME/checkpoints/model.safetensors \
                    --proj int4 --embed int8 --group 512 --out /tmp/check.npk
```

Expect `228 tensors, max |diff| 3.125e-02` — exactly the bf16 quantum at the
magnitude of the largest weight, i.e. the whole difference is the safetensors
round-trip. The stronger check is that `/tmp/check.npk` scores 8/8 on the
parity fixtures.

---

## What the trainer reads

`needle.cli finetune` takes JSONL. Three string fields per line; `tools` and
`answers` are JSON *inside* strings:

```json
{"query":"What's the weather in Oslo?","tools":"[{\"name\":\"get_weather\",...}]","answers":"[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Oslo\"}}]"}
```

Constraints worth knowing before you generate a corpus:

- It refuses fewer than 3 examples, and warns below **120 per tool**
  (100 train / 10 val / 10 test).
- Tool names are snake_cased before tokenization, so `getWeather` in your tools
  list must be `get_weather` in `answers`. `Needle.prompt()` and `tools/pack_npk.py`
  apply the same rule — don't hand-roll it.
- `NE_MAX_ENC` is 384 tokens. A prompt is your query plus the full JSON of every
  tool you offer, so verbose schemas eat the budget fast.

## What the data has to do

At 26M parameters the corpus carries most of the routing quality. Four
properties matter:

- **Vary which tools are on offer per example.** A fixed menu teaches position
  rather than selection. Sample a different 2–4 tool subset each time, always
  containing the right answer.
- **Include negatives.** Examples where nothing fits, with `answers` of `"[]"`.
  Without them the model learns that a call is always required and answers
  "write me a poem" with a weather lookup.
- **Pin argument types by example.** A corpus that is inconsistent about
  `{"minutes": 3}` versus `{"minutes": "3 minutes"}` gets you the string form.
- **Vary phrasing, including terse and indirect.** "weather, Oslo" and "do I
  need a jacket in Oslo" should both land.

`train/make_dataset.py` is a worked example that does all four for six tools.
Read it as a template, not a dependency:

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

A fine-tune that keeps the base architecture is a repack and a reflash. These
are compile-time and are not:

- `NE_MAX_ENC` 384, `NE_MAX_GEN` 64, `NE_VOCAB` 8192.
- The architecture constants in `needle_engine.h` — 12 encoder layers, 8
  decoder, `d_model` 512, 8 query / 4 KV heads.
- `--group` is 32 or 512 only. Both the packer and `ne_load` refuse anything
  else, because no kernel exists for other layouts.

## Scoring the result

`tools/eval_tools.py` runs seven cases with all six demo tools offered at once,
including one request no tool can serve, and prints the tool each case routed
to:

```bash
python3 tools/eval_tools.py weights/mine.npk
```

The artifact the board ships scores 7/7 on it. Needs `engine/needle_host`
built (`make -C engine`); `tools/finetune.sh` builds it if missing.

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

Preflight prints the backend it found and how to fix it. Typical causes: no
NVIDIA driver in this environment, CPU-only jax installed, or Windows without
WSL2. Re-run `source ./setup` inside `$NEEDLE_HOME` after `nvidia-smi` works.

### Training dies mid-run over SSH / WSL2

A backgrounded job dies with the SSH session. Use `tmux` or `screen`, or keep
the session attached for the whole train.

### `cannot import needle.cli`

`source ./setup` in the upstream checkout (creates `.venv`, `pip install -e .`,
installs the matching JAX build). Invoke training as
`python -m needle.cli finetune ...`, not `python -m needle`.

### Trainer metrics look great, board routes badly

You trusted the trainer eval. Pack the best checkpoint and run
`tools/eval_tools.py` on the `.npk`. Only that score is the board path.
