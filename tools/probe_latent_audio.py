#!/usr/bin/env python3
"""Where inside the 12+8 layers an audio query becomes play_tone.

The discipline: print the literal path stage by stage, measure claimed
properties instead of assuming them, and put a number on latent-space
separation with an honest control. Applied to Needle:

  1. encoder depth  -- per-layer pooled states; at which layer do the six
     tools' queries separate? (same-vs-other cosine separation, here
     nearest-centroid over tool classes)
  2. decoder logit lens -- the emitted call is boilerplate until the first
     name piece ("play" vs "get"/"look"/"set"/"board"); rank and margin of
     that piece after every decoder layer at the committing step
  3. cross-attention buckets -- at that step, where the probability mass sits:
     query tokens, the <tools> marker, play_tone's schema entry, other entries
  4. causal probe (claimed, then measured) -- rename
     play_tone in the schema and shuffle the tool order; a copy mechanism
     follows the schema, a memorized route keeps saying play_tone

Uses engine/needle_trace (make -C engine needle_trace), the scalar fp32 path,
whose tokens are fixture-certified identical to the board's.

    python3 tools/probe_latent_audio.py weights/needle_tools_g512.npk
"""
import json
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

DM, VOCAB, DEC_LAYERS, HEADS = 512, 8192, 8, 8
NPK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'weights', 'needle_tools_g512.npk')
OUT = os.environ.get('NE_TRACE_DIR', '/tmp/ne_trace/netrace')
tok = bpe.NeedleBPE()
ALL = list(TOOLS.values())

# ---- battery: in-distribution phrasings, audio from TONE_Q itself ----------
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
NAME_IDS = {  # first piece of each tool name in the emitted call
    'play_tone': 549, 'get_weather': 358, 'get_sun_times': 358,
    'look_up': 6203, 'set_timer': 452, 'board_status': 1073,
}

def cases():
    out = []
    for i, q in enumerate(AUDIO):
        out.append((f"aud{i:02d}", q, ALL))
    for k, qs in OTHER.items():
        for i, q in enumerate(qs):
            out.append((f"{k}{i:02d}", q, ALL))
    out.append(("poem00", "Write me a poem about the sea", ALL))
    # causal probes on the first audio prompt
    renamed = [dict(t, name="make_noise") if t["name"] == "play_tone" else t for t in ALL]
    out.append(("prb_rename", "Play a chime", renamed))
    out.append(("prb_shuffle", "Play a chime", ALL[::-1]))
    out.append(("prb_remove", "Play a chime", [t for t in ALL if t["name"] != "play_tone"]))
    return out

# ---- run the traced engine -------------------------------------------------
def run(case_list):
    os.makedirs(OUT, exist_ok=True)
    vec, meta = [], {}
    for cid, q, tools in case_list:
        ids, tj = bpe.build_encoder_input(tok, q, tools)
        vec.append(f"{cid} {len(ids)} {' '.join(map(str, ids))} 1 1")
        meta[cid] = {'q': q, 'enc': ids, 'tools_json': tj}
    vf = os.path.join(OUT, 'vec.txt')
    open(vf, 'w').write("\n".join(vec) + "\n")
    exe = os.path.join(REPO, 'engine', 'needle_trace')
    r = subprocess.run([exe, NPK, vf, OUT], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"needle_trace failed:\n{r.stdout}\n{r.stderr}")
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] == 'got:':
            meta[p[0]]['got'] = [int(x) for x in p[2:]]
    return meta

def records(path):
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(24)
            if len(hdr) < 24:
                return
            tag = hdr[:8].rstrip(b'\0').decode()
            layer, head, step, n = struct.unpack('<4i', hdr[8:])
            yield tag, layer, head, step, np.frombuffer(f.read(4 * n), dtype='<f4')

def load_case(cid):
    d = {'encq': {}, 'enca': {}, 'enct': {}, 'dcros': {}, 'dself': {},
         'xprob': {}, 'logit': {}}
    for tag, layer, head, step, v in records(os.path.join(OUT, f'{cid}.bin')):
        if tag in ('encq', 'enca', 'enct'):
            d[tag][layer] = v
        elif tag in ('dcros', 'dself'):
            d[tag][(step, layer)] = v
        elif tag == 'xprob':
            d[tag][(step, layer, head)] = v
        elif tag == 'logit':
            d[tag][step] = v
    return d

# ---- offline lens ----------------------------------------------------------
def load_model_bin():
    emb = fnw = None
    for tag, *_rest, v in records(os.path.join(OUT, 'model.bin')):
        if tag == 'dfnw':
            fnw = v
        elif tag == 'embw':
            emb = v.reshape(VOCAB, DM)
    return emb, fnw

def lens(x, emb, fnw):
    h = (1.0 + fnw) * x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6)
    return emb @ h.astype(np.float32)

def name_step(got):
    """Step emitting the first tool-name piece: the token after 'name' '":"'."""
    for i in range(2, len(got)):
        if got[i - 2] == 294 and got[i - 1] == 264:
            return i
    return None

# ---- encoder token buckets -------------------------------------------------
def buckets(enc, tools):
    tp = enc.index(5)
    spans = {'query': list(range(tp)), 'marker': [tp]}
    tail, base = enc[tp + 1:], tp + 1
    starts = {}
    for t in tools:
        seq = tok.encode('"name":"%s"' % t["name"])[3:-1]   # the name pieces
        for i in range(len(tail) - len(seq) + 1):
            if tail[i:i + len(seq)] == seq:
                starts[t["name"]] = (base + i, len(seq))
                break
    order = sorted(starts.items(), key=lambda kv: kv[1][0])
    for j, (nm, (s, ln)) in enumerate(order):
        end = order[j + 1][1][0] if j + 1 < len(order) else len(enc)
        spans[f'name:{nm}'] = list(range(s, s + ln))
        spans[f'entry:{nm}'] = list(range(s + ln, end))
    return spans

def mass(spans, p):
    tot = {}
    for k, idx in spans.items():
        tot[k] = float(p[idx].sum()) if idx else 0.0
    return tot

# ---- main ------------------------------------------------------------------
if __name__ == '__main__':
    meta = run(cases())
    emb, fnw = load_model_bin()

    def toolname(got):
        s = tok.decode(got) if hasattr(tok, 'decode') else None
        if s is None:
            pieces = ''.join(tok.id_to_piece[i] for i in got).replace('▁', ' ')
            s = pieces
        try:
            return json.loads(s.strip())[0]
        except Exception:
            return {'raw': s.strip()}

    print("== generations ==")
    for cid in meta:
        print(f"  {cid:12s} -> {toolname(meta[cid]['got'])}")

    # 1. encoder depth: nearest-centroid over tool classes, leave-one-out
    classes = {'aud': 'play_tone', 'wea': 'get_weather', 'sun': 'get_sun_times',
               'lok': 'look_up', 'tim': 'set_timer', 'brd': 'board_status'}
    ids_by_class = {c: [k for k in meta if k.startswith(c)] for c in classes}
    layers = sorted(load_case('aud00')['encq'].keys())
    print("\n== encoder depth: leave-one-out nearest-centroid on query-mean ==")
    states = {cid: load_case(cid)['encq'] for cl in ids_by_class.values() for cid in cl}
    print("  layer  acc    aud-vs-rest separation (same-cos - cross-cos)")
    for L in layers:
        X = {cid: st[L] / np.linalg.norm(st[L]) for cid, st in states.items()}
        ok = tot = 0
        for c, cl in ids_by_class.items():
            for cid in cl:
                cents = {}
                for c2, cl2 in ids_by_class.items():
                    rest = [X[k] for k in cl2 if k != cid]
                    cents[c2] = np.mean(rest, axis=0)
                pred = max(cents, key=lambda c2: float(X[cid] @ cents[c2]))
                ok += (pred == c); tot += 1
        aud = [X[k] for k in ids_by_class['aud']]
        oth = [X[k] for c2, cl2 in ids_by_class.items() if c2 != 'aud' for k in cl2]
        same = np.mean([a @ b for i, a in enumerate(aud) for b in aud[i + 1:]])
        cross = np.mean([a @ b for a in aud for b in oth])
        print(f"  {L:5d}  {ok/tot:4.2f}   {same:+.4f} - {cross:+.4f} = {same-cross:+.4f}")

    # 2. logit lens at the committing step, audio cases
    print("\n== decoder logit lens at the first-name-piece step (aud00) ==")
    cid = 'aud00'
    d = load_case(cid)
    ns = name_step(meta[cid]['got'])
    emitted = meta[cid]['got'][ns]
    rivals = sorted(set(NAME_IDS.values()) - {emitted})
    print(f"  emits id {emitted} '{tok.id_to_piece[emitted]}' at step {ns}")
    print("  after   rank(play)  logit(play)  best rival        margin")
    for L in range(DEC_LAYERS):
        lg = lens(d['dcros'][(ns, L)], emb, fnw)
        rank = int((lg > lg[emitted]).sum())
        rb = max(rivals, key=lambda i: lg[i])
        print(f"  cross{L}  {rank:9d}  {lg[emitted]:+10.3f}  "
              f"{tok.id_to_piece[rb]:8s}{lg[rb]:+8.3f}  {lg[emitted]-lg[rb]:+8.3f}")
    err = np.abs(lens(d['dcros'][(ns, DEC_LAYERS - 1)], emb, fnw) - d['logit'][ns]).max()
    print(f"  lens check vs engine logits at layer 7: max |diff| = {err:.4e}")

    # 3. cross-attention mass at the committing step
    print("\n== cross-attention mass at the committing step (mean over aud cases) ==")
    keys = ['query', 'marker', 'name:play_tone', 'entry:play_tone']
    acc = {}
    for cid in ids_by_class['aud']:
        d = load_case(cid)
        ns = name_step(meta[cid]['got'])
        sp = buckets(meta[cid]['enc'], ALL)
        for L in range(DEC_LAYERS):
            for h in range(HEADS):
                m = mass(sp, d['xprob'][(ns, L, h)])
                m['other entries'] = 1.0 - sum(m[k] for k in keys)
                acc.setdefault((L, h), []).append(m)
    print("  layer  " + "".join(f"{k:>16s}" for k in keys + ['other entries']) + "   (mean over heads; max head in brackets)")
    for L in range(DEC_LAYERS):
        row = "  " + f"cross{L}"
        for k in keys + ['other entries']:
            per_head = [np.mean([m[k] for m in acc[(L, h)]]) for h in range(HEADS)]
            row += f"  {np.mean(per_head):5.2f} [{np.max(per_head):4.2f}]"
        print(row)

    print("\n== causal probes ==")
    for cid in ('prb_rename', 'prb_shuffle', 'prb_remove'):
        print(f"  {cid:12s} -> {toolname(meta[cid]['got'])}")
