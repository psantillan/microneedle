# Weight artifacts

Provenance, license, and integrity hashes for the `.npk` packs used by
MicroNeedle. Artifacts live under `weights/` when built or downloaded; they
are not committed to git (see `.gitignore`: `weights/*.npk`).

## License

Every `.npk` listed here is an **MIT derivative** of the upstream
[cactus-compute/needle](https://github.com/cactus-compute/needle) model
weights (MIT; see `THIRD_PARTY_NOTICES.md` and the upstream `LICENSE`).
Base packs quantize and repack those checkpoints. Fine-tuned packs start from
the same base, train on synthetic tool-routing data from
`train/make_dataset.py`, and pack with `tools/pack_npk.py`.

## Per-artifact provenance

Hashes were computed with `sha256sum` over the bytes of each file.

### Base packs (upstream MIT checkpoints)

Packed from the upstream Needle checkpoint (default path:
`$NEEDLE_HOME/checkpoints/model.safetensors` or the matching `.pkl`) via
`tools/pack_npk.py`. No additional training.

| File | Size | Recipe | SHA-256 |
|---|---:|---|---|
| `weights/needle_v3_g512.npk` | 14.66 MiB | int4 projections, int8 embedding, group **512** (shipped base / parity artifact) | `27ab3dbf53893649d0943b6716710ce80f99bb7687f12cd64092cb23f40779ba` |
| `weights/needle_v2_int4p_int8e.npk` | 16.12 MiB | int4 projections, int8 embedding, group **32** (earlier NPK2 layout) | `7f0407fe6cc82a1fa12476a0fdf42d69d118a55428d183b0980d249a07e019f5` |
| `weights/needle_int4p_int8e.npk` | 16.12 MiB | int4 projections, int8 embedding, group **32** (legacy NPK1 layout) | `7dfd6c831fc0b17d1a2c0e147f017e893b9ae873a1f929d8ef180513bc2daf71` |
| `weights/needle_int8p_int8e.npk` | 26.62 MiB | int8 projections, int8 embedding, group **32** (legacy NPK1 layout) | `f9974270521dc3ab8ae0659a9522009ff8a059183a19284b3194d9fd79fdf2ed` |

`needle_v3_g512.npk` is the base model the 8/8 host and board parity fixtures
are scored against.

### Fine-tuned variants (synthetic data from `train/make_dataset.py`)

Trained on tool-routing JSONL from `train/make_dataset.py` (six demo tools:
`get_weather`, `get_sun_times`, `look_up`, `set_timer`, `play_tone`,
`board_status`), then packed with `--proj int4 --embed int8 --group 512`.

| File | Size | Provenance | SHA-256 |
|---|---:|---|---|
| `weights/needle_tools_g512.npk` | 14.66 MiB | Fine-tune on the six demo tools; used by the tools routing battery (`tools/eval_tools.py`) | `56b7978efe4a9c543db4d14529602c48f14645fe8d21923b516b5a112f08102c` |
| `weights/needle_runa_g512.npk` | 14.66 MiB | “Run A” fine-tune variant of the demo-tools corpus (shipping product pack alongside the tools fine-tune) | `1856ad55e935b90b9be20b0a2bf5c245d2bdfc55dc05b8b7ae32e707f0512cfa` |

## Related non-pack files under `weights/`

Small fixtures checked into the repo for gates; not model weight packs and not
release downloads:

- `vectors_int4p_int8e.txt` — host parity vector set
- `reference_int4p_int8e.json` — oracle dump used to build the vectors
- `board_reference_i8.json` — board-generated reference for the int8 path

### Non-pack Release asset

| file | what it is | sha256 |
|---|---|---|
| `runa_best.pkl` | Raw trainer checkpoint the runa pack derives from (numpy pickle; only needed to re-pack or continue training — packs above suffice for everything else) | `a288bce2f6367002ae58a96cffd9e8d60502dfcd1dc5ffde08bc6f3214eb565a` |

## Download

**GitHub Releases is the only supported download route for strangers.**

Upstream’s public git repository does **not** contain `model.safetensors`
(or other checkpoints): their `.gitignore` excludes `checkpoints/`. Packing
from upstream requires their checkpoint by other means (Hugging Face weight
drop or a local training run). A Release `.npk` needs nothing from upstream.

| Artifact | Download URL |
|---|---|
| `needle_v3_g512.npk` | https://github.com/psantillan/microneedle/releases/download/v1.0.0 |
| `needle_tools_g512.npk` | https://github.com/psantillan/microneedle/releases/download/v1.0.0 |
| `needle_runa_g512.npk` | https://github.com/psantillan/microneedle/releases/download/v1.0.0 |
| `needle_v2_int4p_int8e.npk` | https://github.com/psantillan/microneedle/releases/download/v1.0.0 |
| `needle_int4p_int8e.npk` | https://github.com/psantillan/microneedle/releases/download/v1.0.0 |
| `needle_int8p_int8e.npk` | https://github.com/psantillan/microneedle/releases/download/v1.0.0 |

Replace each https://github.com/psantillan/microneedle/releases/download/v1.0.0 with the concrete GitHub Release asset link
when the first public release is published. Until then, build packs locally
with `tools/pack_npk.py` (and `verify.sh --pack` for the base artifact) if
you already have an upstream checkpoint.

## Reproducing a base pack locally

```bash
# Requires $NEEDLE_HOME with checkpoints/model.safetensors present
python3 tools/pack_npk.py --proj int4 --embed int8 --group 512 \
    --out weights/needle_v3_g512.npk
sha256sum weights/needle_v3_g512.npk
```

Fine-tunes: generate data with `train/make_dataset.py`, run
`needle finetune` in the upstream tree, then
`tools/pack_npk.py --checkpoint path/to/*_best.pkl ...` as described in
`docs/finetuning.md`.
