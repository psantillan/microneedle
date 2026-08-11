"""Pack Needle weights from model.safetensors into the .npk format the C engine loads.

Rules this packer obeys:

- The default source is model.safetensors (bf16, HF/torch [out,in] layout). Its
  layers arrive unstacked and pre-transposed, so the packer is a quantizer
  rather than a format converter, and its tensor names match the CQ4 bundle's
  manifest. `--checkpoint` reads a .pkl instead; see load_checkpoint_pkl.

- lm_head.weight is dropped: byte-identical to model.embed_tokens.weight
  (sha256-verified upstream). The engine uses one buffer for both the input
  gather and the output projection. That is the 2.52 MiB dedup.

- Quantization is group-32 along the INPUT dim (the k of y[n] = sum_k w[n,k]x[k]),
  symmetric, scale = max|w|/(2^(bits-1)-1), fp16 scales -- the same recipe as
  needle/model/quantize.py, so the accuracy measured in QUANT_GATE2.json applies
  to this artifact. Per tensor the width is int8 or int4; 1-D tensors (norms,
  gates) stay fp16 raw, exactly like the CQ4 bundle's fallback set.

- Norm weights are stored RAW (the engine applies 1+w) and gates RAW (the engine
  applies sigmoid). Keeping them unfolded is what lets the engine be diffed
  tensor-by-tensor against the JAX reference mid-layer.

Layout (all little-endian, blob 16-byte aligned):
  u32 magic 'NPK2', u32 n_tensors, u64 blob_bytes
  n_tensors x 96-byte records:
    char name[64] NUL-padded
    u32 dtype     0=f32 1=f16 2=int8_g32 3=int4_g32
    u32 rows, cols   (cols=0 for 1-D)
    u32 group
    u64 scale_off, u64 data_off   (blob-relative; scale_off=0xFFFFFFFFFFFFFFFF if none)
  blob:
    int8_g32: scales fp16 [rows, cols/group] then values int8 [rows, cols]
    int4_g32: scales fp16 [rows, cols/group] then values packed 2/byte in
              SPLIT order per 32-element chunk -- byte j holds elem j in the
              low nibble and elem j+16 in the high nibble, two's-complement
              in [-8,7]. The 32-element chunk is fixed by the PIE vector
              registers and is INDEPENDENT of `group`, which only controls
              how many scales a row carries.
    f16: raw values
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# Upstream checkout: override with --safetensors or $NEEDLE_HOME.
NEEDLE_HOME = os.environ.get("NEEDLE_HOME", os.path.expanduser("~/needle"))
ST = os.path.join(NEEDLE_HOME, "checkpoints", "model.safetensors")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAGIC = 0x324B504E  # 'NPK2' -- int4 groups store elems 0..15 in low nibbles, 16..31 in high
GROUP = 32
DT_F32, DT_F16, DT_I8, DT_I4 = 0, 1, 2, 3
NO_SCALE = 0xFFFFFFFFFFFFFFFF


def load_checkpoint_pkl(path):
    """Read a `needle finetune` / `needle train` .pkl into the same
    {name: [out, in] fp32} dict load_safetensors returns.

    Three differences from the safetensors export, all of them silent if you
    get them wrong:
      - layers are STACKED on axis 0 (one (12, 512, 512) array, not twelve),
      - kernels are [in, out] the JAX way, not [out, in],
      - the contrastive_* / log_temp leaves are the unused speech head.
    Pass --verify-against to check this mapping against the safetensors export.
    """
    import pickle
    with open(path, "rb") as f:
        ck = pickle.load(f)
    P = ck["params"] if "params" in ck else ck
    out = {}

    def g(*path_parts):
        d = P
        for k in path_parts:
            d = d[k]
        return np.asarray(d, dtype=np.float32)

    out["model.embed_tokens.weight"] = g("embedding", "embedding")
    out["model.encoder.final_norm.weight"] = g("encoder", "final_norm", "scale")
    out["model.decoder.norm.weight"] = g("decoder", "ZCRMSNorm_0", "scale")

    enc = P["encoder"]["layers"]["EncoderBlock_0"]
    n_enc = np.asarray(enc["attn_gate"]).shape[0]
    for i in range(n_enc):
        b = f"model.encoder.layers.{i}"
        out[f"{b}.input_layernorm.weight"] = g("encoder","layers","EncoderBlock_0","ZCRMSNorm_0","scale")[i]
        out[f"{b}.attn_gate"] = g("encoder","layers","EncoderBlock_0","attn_gate")[i].reshape(1)
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            out[f"{b}.self_attn.{proj}.weight"] = g(
                "encoder","layers","EncoderBlock_0","self_attn",proj,"kernel")[i].T
        for nm in ("q_norm", "k_norm"):
            out[f"{b}.self_attn.{nm}.weight"] = g(
                "encoder","layers","EncoderBlock_0","self_attn",nm,"scale")[i]

    dec = P["decoder"]["layers"]["DecoderBlock_0"]
    n_dec = np.asarray(dec["self_attn_gate"]).shape[0]
    for i in range(n_dec):
        b = f"model.decoder.layers.{i}"
        out[f"{b}.input_layernorm.weight"] = g("decoder","layers","DecoderBlock_0","ZCRMSNorm_0","scale")[i]
        out[f"{b}.encoder_attn_layer_norm.weight"] = g("decoder","layers","DecoderBlock_0","ZCRMSNorm_1","scale")[i]
        out[f"{b}.self_attn_gate"] = g("decoder","layers","DecoderBlock_0","self_attn_gate")[i].reshape(1)
        out[f"{b}.cross_attn_gate"] = g("decoder","layers","DecoderBlock_0","cross_attn_gate")[i].reshape(1)
        for src, dst in (("self_attn", "self_attn"), ("cross_attn", "encoder_attn")):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                out[f"{b}.{dst}.{proj}.weight"] = g(
                    "decoder","layers","DecoderBlock_0",src,proj,"kernel")[i].T
            for nm in ("q_norm", "k_norm"):
                out[f"{b}.{dst}.{nm}.weight"] = g(
                    "decoder","layers","DecoderBlock_0",src,nm,"scale")[i]

    out["lm_head.weight"] = out["model.embed_tokens.weight"]   # tied
    return out


def load_safetensors(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for name, m in meta.items():
            if name == "__metadata__":
                continue
            assert m["dtype"] == "BF16", (name, m["dtype"])
            f.seek(base + m["data_offsets"][0])
            raw = f.read(m["data_offsets"][1] - m["data_offsets"][0])
            u16 = np.frombuffer(raw, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            out[name] = f32.reshape(m["shape"]).copy()
    return out


def quant_groups(w, bits, group):
    """w [rows, cols] -> (scales fp16 [rows, cols/group], q int8 [rows, cols]).

    The scale must be rounded to fp16 BEFORE the weights are quantized against
    it, because fp16 is what the engine reads back. Rounding afterwards
    quantizes against a scale that never exists at inference time, shifting
    ~0.6% of codes by one step, and the recipe stops matching
    needle/model/quantize.py.
    """
    rows, cols = w.shape
    assert cols % group == 0, (rows, cols)
    qmax = 2 ** (bits - 1) - 1
    g = w.reshape(rows, cols // group, group)
    scale = np.maximum(np.abs(g).max(axis=2, keepdims=True) / qmax, 1e-8)
    scale = scale.astype(np.float16).astype(np.float32)
    q = np.clip(np.round(g / scale), -qmax - 1, qmax).astype(np.int8)
    return scale[:, :, 0].astype(np.float16), q.reshape(rows, cols)


PACK = 32   # nibble-packing chunk, fixed by the PIE kernel -- NOT the scale group


def pack_int4(q):
    """int8 in [-8,7] -> packed bytes, SPLIT order per chunk of 32:
    byte j of a chunk = elem[j] | elem[j+16] << 4.

    Split order is what the PIE kernel can consume: it loads one 16-byte chunk
    with a single vld.128, masks the low nibbles for elements 0..15 and shifts
    for 16..31, and both land in lane order matching the activation registers
    with zero shuffling. The interleaved layout would need a byte-permute PIE
    does not have.

    This chunk size is a property of the vector registers and is independent of
    the quantization group: a row may carry one scale (group=cols) or sixteen
    (group=32) and the bytes are laid out identically either way.
    """
    rows, cols = q.shape
    g = (q.astype(np.int16) & 0xF).astype(np.uint8).reshape(rows, cols // PACK, PACK)
    return (g[:, :, :16] | (g[:, :, 16:] << 4)).reshape(rows, cols // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", choices=["int8", "int4"], default="int4",
                    help="width for attention projections")
    ap.add_argument("--embed", choices=["int8", "int4"], default="int8",
                    help="width for the tied embedding")
    ap.add_argument("--group", type=int, default=32,
                    help="quantization group along the input dim; 512 = one "
                         "scale per row, which lets the PIE kernel drain its "
                         "accumulator once instead of once per group")
    ap.add_argument("--safetensors", default=ST,
                    help="upstream model.safetensors (default: $NEEDLE_HOME/checkpoints)")
    ap.add_argument("--checkpoint", default=None,
                    help="pack a `needle finetune` .pkl instead of the safetensors "
                         "export -- this is how a fine-tune reaches the board")
    ap.add_argument("--verify-against", default=None,
                    help="safetensors to diff the loaded tensors against; use with "
                         "--checkpoint on the BASE model to prove the mapping")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # the group is part of the on-disk contract (different kernels read them),
    # so it belongs in the default filename -- otherwise two incompatible
    # artifacts collide on one name
    out_path = args.out or os.path.join(
        REPO, "weights", f"needle_{args.proj}p_{args.embed}e_g{args.group}.npk")
    global GROUP
    GROUP = args.group
    # The engine implements exactly two scale layouts: 32 (per-group kernels)
    # and cols (single-drain kernels). Anything else assembles into an
    # artifact whose PIE path silently multiplies only part of each row.
    if GROUP != 32 and GROUP != 512:
        raise SystemExit(f"--group must be 32 or 512, got {GROUP}: the engine "
                         "has no kernel for other layouts")

    # the default source path is one machine's checkout; on any other clone the
    # miss otherwise surfaces as a FileNotFoundError from inside the loader
    if not args.checkpoint and not os.path.exists(args.safetensors):
        raise SystemExit(
            f"no weights at {args.safetensors}\n"
            f"  $NEEDLE_HOME is {NEEDLE_HOME} -- point it at your upstream Needle\n"
            "  checkout, or pass --safetensors PATH (or --checkpoint PATH for a\n"
            "  `needle finetune` .pkl)")

    if args.checkpoint:
        tensors = load_checkpoint_pkl(args.checkpoint)
        print(f"  source  : {args.checkpoint} (checkpoint)")
        if args.verify_against:
            ref = load_safetensors(args.verify_against)
            miss = set(ref) ^ set(tensors)
            if miss:
                raise SystemExit(f"tensor sets differ: {sorted(miss)[:6]}")
            worst, wname = 0.0, ""
            for k in ref:
                if ref[k].shape != tensors[k].shape:
                    raise SystemExit(f"{k}: {tensors[k].shape} vs {ref[k].shape}")
                d = float(np.abs(ref[k] - tensors[k]).max())
                if d > worst: worst, wname = d, k
            print(f"  verify  : {len(ref)} tensors, max |diff| {worst:.3e} (worst {wname})")
    else:
        tensors = load_safetensors(args.safetensors)
    emb = tensors.pop("model.embed_tokens.weight")
    lm = tensors.pop("lm_head.weight")
    assert np.array_equal(emb, lm), "tie assumption broken -- do not dedup"
    del lm

    records = []   # (name, dtype, rows, cols, scale_bytes, data_bytes)
    blob = bytearray()

    def align16():
        while len(blob) % 16:
            blob.append(0)

    def add(name, arr, width):
        assert len(name) < 64, name
        if arr.ndim == 1:
            align16()
            off = len(blob)
            blob.extend(arr.astype(np.float16).tobytes())
            records.append((name, DT_F16, arr.shape[0], 0, NO_SCALE, off))
            return
        bits = 8 if width == "int8" else 4
        scales, q = quant_groups(arr, bits, GROUP)
        align16()
        soff = len(blob)
        blob.extend(scales.tobytes())
        align16()
        doff = len(blob)
        blob.extend((q.tobytes() if bits == 8 else pack_int4(q).tobytes()))
        records.append((name, DT_I8 if bits == 8 else DT_I4,
                        arr.shape[0], arr.shape[1], soff, doff))

    add("embed_tokens", emb, args.embed)
    for name in sorted(tensors):
        short = name.replace("model.", "")
        add(short, tensors[name], args.proj)

    hdr = struct.pack("<IIQ", MAGIC, len(records), len(blob))
    rec_bytes = b""
    for (name, dt, rows, cols, soff, doff) in records:
        rec_bytes += struct.pack("<64sIIIIQQ", name.encode(), dt, rows, cols,
                                 GROUP, soff, doff)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(hdr)
        f.write(rec_bytes)
        f.write(bytes(blob))

    total = len(hdr) + len(rec_bytes) + len(blob)
    print(f"{out_path}")
    print(f"  tensors : {len(records)}")
    print(f"  blob    : {len(blob):,} B")
    print(f"  total   : {total:,} B = {total/1048576:.2f} MiB")
    print(f"  proj={args.proj} embed={args.embed} group={GROUP}")


if __name__ == "__main__":
    main()
