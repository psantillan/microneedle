#!/usr/bin/env python3
"""Margin audit: every decision the int8 build makes by less than a threshold.

The int8 attention configuration is numerically exact in its own arithmetic,
but fp32 libm/codegen jitter (which host and board have never promised to
share -- see the engine header) can flip an argmax whose top-2 logit gap is
small enough. Measured: the one certification divergence sits on a
0.0097-logit gap; its neighbors clear 5. This tool enumerates every
generation step under the threshold so the board-referenced certification
can carry a documented exception list instead of a shrug.

    python3 tools/margin_audit.py weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt
    python3 tools/margin_audit.py weights/needle_runa_g512.npk --battery

Requires: make -C engine needle_i8_trace
"""
import os
import struct
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tokenizer'))
import bpe

THRESHOLD = 1.0   # logits; jitter amplitude measured well under 0.01, x100 safety

def records(path):
    with open(path, 'rb') as f:
        while True:
            h = f.read(24)
            if len(h) < 24:
                return
            tag = h[:8].rstrip(b'\0').decode()
            layer, head, step, n = struct.unpack('<4i', h[8:])
            yield tag, layer, head, step, np.frombuffer(f.read(4 * n), dtype='<f4')

def main():
    npk = sys.argv[1]
    tok = bpe.NeedleBPE()
    out = os.path.join(os.environ.get('NE_TRACE_DIR', '/tmp'), 'margin_audit')
    os.makedirs(out, exist_ok=True)

    if '--battery' in sys.argv:
        sys.path.insert(0, os.path.join(REPO, 'train'))
        from make_dataset import TOOLS
        cases = [("What's the weather in Oslo?", 'wea'), ("When does the sun set in Reykjavik?", 'sun'),
                 ("Tell me about tardigrades", 'lok'), ("Set a timer for 5 minutes", 'tim'),
                 ("Play a chime", 'aud'), ("How long have you been running?", 'brd'),
                 ("What chip are you running on?", 'brd03'), ("Write me a poem about the sea", 'poem')]
        vf = os.path.join(out, 'vec.txt')
        with open(vf, 'w') as f:
            for i, (q, c) in enumerate(cases):
                ids, _ = bpe.build_encoder_input(tok, q, list(TOOLS.values()))
                f.write(f"{c}{i:02d} {len(ids)} {' '.join(map(str, ids))} 1 1\n")
    else:
        vf = sys.argv[2]

    exe = os.path.join(REPO, 'engine', 'needle_i8_trace')
    r = subprocess.run([exe, npk, vf, out], capture_output=True, text=True)
    if r.returncode not in (0, 1):   # main_host-style exit; traces still written
        raise SystemExit(f"trace run failed: {r.stderr}")

    flagged = 0
    for line in open(vf):
        cid = line.split()[0]
        p = os.path.join(out, f'{cid}.bin')
        if not os.path.exists(p):
            continue
        for tag, _l, _h, st, v in records(p):
            if tag != 'logit':
                continue
            top = np.argsort(v)[::-1][:2]
            gap = float(v[top[0]] - v[top[1]])
            if gap < THRESHOLD:
                flagged += 1
                print(f"KNIFE-EDGE {cid} step {st}: gap={gap:.4f}  "
                      f"'{tok.id_to_piece[top[0]]}'({top[0]}) vs '{tok.id_to_piece[top[1]]}'({top[1]})")
    print(f"\n{flagged} decision(s) under {THRESHOLD} logits. Each is a documented "
          f"exception where host/board argmax may legitimately differ; everywhere "
          f"else, disagreement is a defect.")

if __name__ == '__main__':
    main()
