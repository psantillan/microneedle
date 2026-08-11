"""Dump the parity oracle for the C engine.

For each fixture case: the exact encoder token ids, the greedy unconstrained
token sequence, and the first-step logits -- from the JAX model with the SAME
fake-quantization the .npk packer applied: int4 projections, int8 embedding, at
the group size in $NEEDLE_GROUP, which defaults to 32. The shipped artifact is
packed at group 512, so the default dump is the stricter reference; set
NEEDLE_GROUP=512 to regenerate a matched pair. The C engine is wrong wherever
it disagrees with this file, with no room to blame quantization: both sides saw
identical weights.

Unconstrained on purpose: constrained.py is a separate component with its own
future port and its own tests. The engine parity gate must test the engine.
"""

import json
import sys

import os

# Upstream Needle checkout and this repo; override with $NEEDLE_HOME.
NEEDLE_HOME = os.environ.get("NEEDLE_HOME", os.path.expanduser("~/needle"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NEEDLE_HOME)

import numpy as np
import jax
import jax.numpy as jnp

from needle.model import SimpleAttentionNetwork
from needle.model.architecture import make_causal_mask, make_padding_mask
from needle.model.run import load_checkpoint, _build_encoder_input
from needle.model.quantize import _fake_quantize_int4, _fake_quantize_int8
from needle.model.run import normalize_tools
from needle.dataset.tokenizer import get_tokenizer, EOS_ID

FIXTURE = os.path.join(REPO, "test/parity_set.json")
OUT = os.path.join(REPO, "weights/reference_int4p_int8e.json")

# The shipped artifact is packed at group 512; this oracle is dumped at 32,
# which makes it the stricter reference. Override to regenerate a matched pair.
GROUP_SIZE = int(os.environ.get("NEEDLE_GROUP", "32"))
MAX_GEN = 48


def quantize(params, kernel_prec, embed_prec, group_size=GROUP_SIZE):
    fns = {"int4": _fake_quantize_int4, "int8": _fake_quantize_int8}
    kfn, efn = fns[kernel_prec], fns[embed_prec]

    def visit(path, leaf):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if name == "kernel":
            if leaf.ndim == 3:
                return jax.vmap(lambda w: kfn(w, group_size=group_size))(leaf)
            if leaf.ndim == 2:
                return kfn(leaf, group_size=group_size)
        if name == "embedding":
            return efn(leaf, group_size=group_size)
        return leaf

    return jax.tree_util.tree_map_with_path(visit, params)


def main():
    cases = json.load(open(FIXTURE))["cases"]
    params, config = load_checkpoint(os.path.join(NEEDLE_HOME, "checkpoints", "needle.pkl"))
    params = quantize(params, "int4", "int8")
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    out = {"quant": "int4 proj / int8 embed / group 32",
           "max_gen": MAX_GEN, "cases": []}

    for case in cases:
        # exactly what generate() feeds the encoder, including tool
        # normalization -- the C tokenizer must reproduce these ids someday,
        # and the C engine must reproduce the outputs from them today
        tools_norm, _ = normalize_tools(case["tools"])
        enc_ids = _build_encoder_input(tokenizer, case["query"], tools_norm)

        # mirror generate() exactly: masked encode returns (enc_out, enc_mask),
        # decode runs over a pad-filled fixed buffer under a causal mask, and
        # the next token is read at position i of a full-buffer forward pass
        enc = jnp.array([enc_ids], dtype=jnp.int32)
        src_mask = make_padding_mask(enc, 0)
        enc_out, enc_mask = model.apply({"params": params}, enc,
                                        src_mask=src_mask, method="encode")

        causal = make_causal_mask(MAX_GEN)
        dec_buffer = jnp.zeros((1, MAX_GEN), dtype=jnp.int32)
        dec_buffer = dec_buffer.at[0, 0].set(EOS_ID)
        gen = []
        first_logits = None
        for i in range(MAX_GEN - 1):
            logits = model.apply({"params": params}, dec_buffer, enc_out,
                                 self_mask=causal, cross_mask=enc_mask,
                                 method="decode")
            step = np.asarray(logits[0, i], dtype=np.float32)
            if first_logits is None:
                first_logits = step
            nxt = int(step.argmax())
            if nxt == EOS_ID:
                break
            gen.append(nxt)
            dec_buffer = dec_buffer.at[0, i + 1].set(nxt)
        out["cases"].append({
            "id": case["id"],
            "query": case["query"],
            "enc_ids": [int(t) for t in enc_ids],
            "gen_ids": gen,
            "gen_text": tokenizer.decode([t for t in gen if t != EOS_ID]),
            "first_logits_top8": [[int(i), float(first_logits[i])]
                                  for i in np.argsort(-first_logits)[:8]],
            "first_logits_sum": float(first_logits.sum()),
            "first_logits_max": float(first_logits.max()),
        })
        print(f"{case['id']:<22} enc={len(enc_ids):4d} gen={len(gen):3d} "
              f"text={out['cases'][-1]['gen_text'][:60]!r}")

    json.dump(out, open(OUT, "w"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
