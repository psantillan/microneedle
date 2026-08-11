#!/usr/bin/env python3
"""The 224-head ablation matrix: which heads the six-tool task actually needs.

Zeroes one attention head's output for a whole run (NE_ABLATE, see
engine/main_trace.c) and diffs the generations against baseline over an
8-case battery. Instrument verified before use: an encoder-layer-0 ablation
destroys the output, and a cross-layer-7 ablation moves the hidden state by
3.6e3 max-abs while leaving tokens unchanged -- the hook fires either way.

    python3 tools/ablate_heads.py weights/needle_tools_g512.npk out.json
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tokenizer'))
sys.path.insert(0, os.path.join(REPO, 'train'))
import bpe
from make_dataset import TOOLS

NPK = sys.argv[1]
OUTJSON = sys.argv[2]
SC = os.environ.get('NE_TRACE_DIR', '/tmp/ne_trace') + '/ablate'
os.makedirs(SC, exist_ok=True)
tok = bpe.NeedleBPE()
ALL = list(TOOLS.values())

CASES = [("aud00", "Play a chime"), ("wea00", "What's the weather in Oslo?"),
         ("sun00", "When does the sun set in Reykjavik?"),
         ("lok00", "Tell me about tardigrades"), ("tim00", "Set a timer for 5 minutes"),
         ("brd00", "How long have you been running?"),
         ("brd03", "What chip are you running on?"),
         ("poem00", "Write me a poem about the sea")]

vf = os.path.join(SC, 'vec.txt')
with open(vf, 'w') as f:
    for cid, q in CASES:
        ids, _ = bpe.build_encoder_input(tok, q, ALL)
        f.write(f"{cid} {len(ids)} {' '.join(map(str, ids))} 1 1\n")

EXE = os.path.join(REPO, 'engine', 'needle_trace')

def run(ablate=None):
    env = dict(os.environ)
    if ablate:
        env['NE_ABLATE'] = ablate
    r = subprocess.run([EXE, NPK, vf, '-'], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"{ablate}: {r.stderr}")
    got = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] == 'got:':
            got[p[0]] = [int(x) for x in p[2:]]
    return got

def classify(base, abl):
    """same | name | args | broken -- worst difference across the battery."""
    def name_of(ids):
        s = ''.join(tok.id_to_piece[i] for i in ids).replace('▁', ' ').strip()
        try:
            j = json.loads(s)
            return (j[0]['name'] if j else '[]', s)
        except Exception:
            return (None, s)
    worst = 'same'
    diffs = {}
    for cid in base:
        if base[cid] == abl.get(cid):
            continue
        bn, bs = name_of(base[cid])
        an, as_ = name_of(abl[cid])
        kind = 'broken' if an is None else 'name' if an != bn else 'args'
        diffs[cid] = {'kind': kind, 'base': bs, 'abl': as_}
        order = ['same', 'args', 'name', 'broken']
        if order.index(kind) > order.index(worst):
            worst = kind
    return worst, diffs

base = run()
print("baseline:", {c: len(v) for c, v in base.items()}, flush=True)

heads = ([("enc", L, h) for L in range(12) for h in range(8)] +
         [("dself", L, h) for L in range(8) for h in range(8)] +
         [("dcross", L, h) for L in range(8) for h in range(8)])
results = {}
for i, (kind, L, h) in enumerate(heads):
    spec = f"{kind}:{L}:{h}"
    worst, diffs = classify(base, run(spec))
    results[spec] = {'worst': worst, 'diffs': diffs}
    print(f"[{i+1}/{len(heads)}] {spec:12s} {worst}"
          + (f"  ({', '.join(sorted(diffs))})" if diffs else ""), flush=True)

json.dump({'base': base, 'results': results}, open(OUTJSON, 'w'), indent=1)
sev = {}
for spec, r in results.items():
    sev.setdefault(r['worst'], []).append(spec)
print("\nsummary:", {k: len(v) for k, v in sev.items()})
