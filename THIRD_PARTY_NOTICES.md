# Third-party notices

This file records third-party material that MicroNeedle uses or derives from.
MicroNeedle itself is MIT licensed; see `LICENSE`.

## cactus-compute/needle (as of upstream commit ffb1c51; the 26M encoder-decoder checkpoint) (MIT)

Upstream project: [cactus-compute/needle](https://github.com/cactus-compute/needle).

The upstream license text, verified from the upstream `LICENSE` file, is
reproduced verbatim below.

```
MIT License

Copyright (c) 2026 Cactus Compute

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### What in this repository derives from it

| Item | Role |
|---|---|
| Model architecture | The encoder–decoder "Simple Attention Network" constants and forward-pass structure ported into `engine/` (C99 + RISC-V vector assembly). Independent reimplementation of the maths, not a line-for-line copy of the JAX sources. |
| `tokenizer/bpe.py` | SentencePiece-BPE tokenizer re-implementation driven by the upstream vocabulary. |
| `web/bpe.js` | Browser twin of `tokenizer/bpe.py`. |
| `tokenizer/vocab.txt` | Vocabulary / merge table derived from the upstream Needle tokenizer. |
| All `.npk` weight artifacts under `weights/` | Packed (and, where noted, fine-tuned) derivatives of upstream Needle checkpoints. See `WEIGHTS.md`. |

This repository is an independent port of the model’s forward pass and related
tooling. It is not affiliated with or endorsed by Cactus Compute.

## Other third-party code in the tree

A scan of this repository found **no other vendored third-party source trees**
(no `vendor/`, `third_party/`, or similar bundled libraries). Runtime and
build dependencies used by tools and firmware (for example NumPy, pyserial, and Espressif ESP-IDF) are obtained separately by the developer; they
are not shipped as source in this tree.

## External services used by the demo (not third-party code)

The interactive demo (`web/index.html`, `demos/ask.py`) calls live HTTP APIs
at runtime. These are **network services**, not code distributed with this
repository:

| Service | Used for |
|---|---|
| [wttr.in](https://wttr.in) | Weather lookups (`get_weather`) |
| [api.sunrisesunset.io](https://api.sunrisesunset.io) | Sunrise / sunset times (`get_sun_times`) |
| [Wikipedia](https://en.wikipedia.org) (REST summary + MediaWiki API) | Fact lookups (`look_up`) |

Their terms of use and rate limits apply to callers of the demo, not as
license terms for MicroNeedle’s own source.
