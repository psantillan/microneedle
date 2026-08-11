#!/usr/bin/env python3
"""Phase 1: cross-query cosine of schema-token encoder states, by depth.

Uses the 29 standard-schema cases from tools/probe_latent_audio.py (audio +
other tools + poem; no causal probes that rewrite the schema). Extends the
NE_TRACE encx records (full per-position state) emitted by needle_trace.

    python3 tools/measure_schema_drift.py [weights/needle_tools_g512.npk]

Prints a table of mean/min pairwise cosine over schema positions bucketed by
depth past the <tools> marker, at every encoder layer checkpoint (0=post-embed,
1..12=post-layer, 13=final norm).
"""
import os
import struct
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tokenizer'))
sys.path.insert(0, os.path.join(REPO, 'train'))
import bpe
from make_dataset import TOOLS

DM = 512
NPK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'weights', 'needle_tools_g512.npk')
OUT = os.environ.get('NE_DRIFT_DIR', '/tmp/ne_schema_drift')
tok = bpe.NeedleBPE()
ALL = list(TOOLS.values())

# Depth buckets: offset past the <tools> marker (inclusive of marker at 0).
BUCKETS = [
    ('0-15',   0, 16),
    ('16-63',  16, 64),
    ('64-127', 64, 128),
    ('128+',   128, 10**9),
]

# 29 standard-schema cases (probe_latent_audio.cases without causal probes)
AUDIO = ["Play a chime", "play the alarm sound", "make a success noise",
         "sound the alarm", "Play a success", "make a chime noise",
         "Play an alarm", "sound the chime"]
OTHER = {
    'wea': ["What's the weather in Oslo?", "weather in San Diego",
            "Is it raining in London?", "How hot is it in Phoenix?"],
    'sun': ["When does the sun set in Reykjavik?", "sunrise time in Tokyo",
            "When is sunset in Lisbon?", "golden hour in Denver"],
    'lok': ["Tell me about tardigrades", "look up the Antikythera mechanism",
            "What is a quasar?", "Who was Ada Lovelace?"],
    'tim': ["Set a timer for 5 minutes", "start a 20 minute timer",
            "countdown 3 minutes", "timer for half an hour"],
    'brd': ["How long have you been running?", "board status",
            "How much memory is free?", "What chip are you running on?"],
}


def cases():
    out = []
    for i, q in enumerate(AUDIO):
        out.append((f"aud{i:02d}", q))
    for k, qs in OTHER.items():
        for i, q in enumerate(qs):
            out.append((f"{k}{i:02d}", q))
    out.append(("poem00", "Write me a poem about the sea"))
    return out


def records(path):
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(24)
            if len(hdr) < 24:
                return
            tag = hdr[:8].rstrip(b'\0').decode()
            layer, head, step, n = struct.unpack('<4i', hdr[8:])
            yield tag, layer, head, step, np.frombuffer(f.read(4 * n), dtype='<f4')


def run(case_list):
    os.makedirs(OUT, exist_ok=True)
    vec, meta = [], {}
    for cid, q in case_list:
        ids, tj = bpe.build_encoder_input(tok, q, ALL)
        vec.append(f"{cid} {len(ids)} {' '.join(map(str, ids))} 1 1")
        meta[cid] = {'q': q, 'enc': ids, 'tp': ids.index(5)}
    vf = os.path.join(OUT, 'vec.txt')
    open(vf, 'w').write("\n".join(vec) + "\n")
    exe = os.path.join(REPO, 'engine', 'needle_trace')
    if not os.path.isfile(exe):
        subprocess.check_call(['make', '-C', os.path.join(REPO, 'engine'), 'needle_trace'])
    r = subprocess.run([exe, NPK, vf, OUT], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"needle_trace failed:\n{r.stdout}\n{r.stderr}")
    return meta


def load_encx(cid):
    """layer -> float32 array [S, DMODEL]."""
    out = {}
    for tag, layer, head, step, v in records(os.path.join(OUT, f'{cid}.bin')):
        if tag != 'encx':
            continue
        S = step
        out[layer] = v.reshape(S, DM)
    return out


def cosine(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def pairwise_stats(vecs):
    """Mean and min cosine over all unordered pairs."""
    n = len(vecs)
    if n < 2:
        return float('nan'), float('nan'), 0
    acc, mn, cnt = 0.0, 1.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            c = cosine(vecs[i], vecs[j])
            acc += c
            if c < mn:
                mn = c
            cnt += 1
    return acc / cnt, mn, cnt


def main():
    cl = cases()
    assert len(cl) == 29, f"expected 29 standard-schema cases, got {len(cl)}"
    meta = run(cl)
    states = {cid: load_encx(cid) for cid in meta}
    layers = sorted(next(iter(states.values())).keys())

    # Confirm all cases share the same schema token sequence length past marker.
    schema_lens = {cid: len(meta[cid]['enc']) - meta[cid]['tp'] for cid in meta}
    assert len(set(schema_lens.values())) == 1, schema_lens
    ns = next(iter(schema_lens.values()))
    print(f"== schema drift: {len(meta)} queries, schema_len={ns}, "
          f"layers={layers[0]}..{layers[-1]} ==")
    print(f"   npk={NPK}")
    print(f"   depth = offset past <tools> marker (token 0 is the marker itself)")
    print()

    bnames = [b[0] for b in BUCKETS]
    hdr = f"{'layer':>6}"
    for bn in bnames + ['marker']:
        hdr += f"  | {bn + ' mean/min':>18}"
    print(hdr)
    print('-' * len(hdr))

    rows_out = []
    for L in layers:
        row = f"{L:6d}"
        row_data = {'layer': L}
        for bname, lo, hi in BUCKETS:
            # mean state over positions in bucket, per query; then pairwise
            per_q = []
            for cid, st in states.items():
                tp = meta[cid]['tp']
                S = st[L].shape[0]
                # positions in schema coordinates [0, ns)
                pos = []
                for d in range(lo, min(hi, ns)):
                    p = tp + d
                    if p < S:
                        pos.append(p)
                if not pos:
                    per_q.append(np.zeros(DM, dtype=np.float64))
                    continue
                # mean over positions in the bucket
                per_q.append(st[L][pos].mean(axis=0))
            mean_c, min_c, _ = pairwise_stats(per_q)
            row += f"  | {mean_c:7.4f}/{min_c:7.4f}"
            row_data[bname] = (mean_c, min_c)

        # marker alone (depth 0)
        per_q = [states[cid][L][meta[cid]['tp']] for cid in states]
        mean_c, min_c, _ = pairwise_stats(per_q)
        row += f"  | {mean_c:7.4f}/{min_c:7.4f}"
        row_data['marker'] = (mean_c, min_c)
        print(row)
        rows_out.append(row_data)

    # Also: per-position mean cosine at final layer (compact summary)
    final = layers[-1]
    print()
    print(f"== per-position mean pairwise cosine at layer {final} "
          f"(final norm), first 16 + every 32nd ==")
    pos_means = []
    for d in range(ns):
        per_q = []
        for cid, st in states.items():
            tp = meta[cid]['tp']
            per_q.append(st[final][tp + d])
        m, mn, _ = pairwise_stats(per_q)
        pos_means.append((d, m, mn))
    show = list(range(0, min(16, ns))) + list(range(16, ns, 32))
    show = sorted(set(show))
    for d in show:
        m, mn = pos_means[d][1], pos_means[d][2]
        print(f"  depth {d:3d}: mean={m:.5f}  min={mn:.5f}")

    # Deep-schema summary for docs/notebook/RESULTS.md
    print()
    print("== summary (final layer) ==")
    for bname, lo, hi in BUCKETS:
        mean_c, min_c = rows_out[-1][bname]
        print(f"  bucket {bname:>7}: mean cosine={mean_c:.5f}  min={min_c:.5f}")
    mean_c, min_c = rows_out[-1]['marker']
    print(f"  marker only     : mean cosine={mean_c:.5f}  min={min_c:.5f}")
    deep = [pos_means[d][1] for d in range(128, ns)]
    if deep:
        print(f"  depths 128+ avg of per-pos means: {np.mean(deep):.5f}")
    print()
    print("done.")


if __name__ == '__main__':
    main()
