#!/usr/bin/env python3
"""int8 attention vs exact (fp32 scalar) on the 36-case tools battery.

Battery = 29 standard-schema cases (probe set without causal probes) + 7
eval_tools cases. Same construction as tools/eval_warmschema.py /
tools/probe_latent_audio.py / tools/eval_tools.py.

Also computes first-name-piece logit margin under exact vs int8 via the
traced builds (needle_trace, needle_i8_trace).

    python3 tools/measure_attn_i8.py [weights/needle_tools_g512.npk]
"""
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tokenizer'))
sys.path.insert(0, os.path.join(REPO, 'train'))
import bpe
from make_dataset import TOOLS

NPK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'weights', 'needle_tools_g512.npk')
OUT = os.environ.get('NE_I8_DIR', '/tmp/ne_attn_i8')
tok = bpe.NeedleBPE()
ALL = list(TOOLS.values())

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
EVAL7 = [
    ("e7_wea", "What's the weather in Oslo?", "get_weather"),
    ("e7_sun", "When does the sun set in Reykjavik?", "get_sun_times"),
    ("e7_lok", "Tell me about tardigrades", "look_up"),
    ("e7_tim", "Set a timer for 5 minutes", "set_timer"),
    ("e7_aud", "Play a chime", "play_tone"),
    ("e7_brd", "How long have you been running?", "board_status"),
    ("e7_poem", "Write me a poem about the sea", None),
]
CLASS_TOOL = {
    'aud': 'play_tone', 'wea': 'get_weather', 'sun': 'get_sun_times',
    'lok': 'look_up', 'tim': 'set_timer', 'brd': 'board_status', 'poem': None,
}
# first piece of each tool name in the emitted call
NAME_IDS = {
    'play_tone': 549, 'get_weather': 358, 'get_sun_times': 358,
    'look_up': 6203, 'set_timer': 452, 'board_status': 1073,
}
RIVAL_IDS = sorted(set(NAME_IDS.values()))


def battery36():
    out = []
    for i, q in enumerate(AUDIO):
        out.append((f"aud{i:02d}", q, 'play_tone'))
    for k, qs in OTHER.items():
        for i, q in enumerate(qs):
            out.append((f"{k}{i:02d}", q, CLASS_TOOL[k]))
    out.append(("poem00", "Write me a poem about the sea", None))
    for cid, q, want in EVAL7:
        out.append((cid, q, want))
    return out


def toolname(ids):
    s = tok.decode(ids)
    try:
        c = json.loads(s.strip())
        if not c:
            return None, None, s.strip()
        return c[0].get('name'), c[0].get('arguments'), s.strip()
    except Exception:
        return '(unparseable)', None, s.strip()


def name_step(got):
    """Step emitting the first tool-name piece: the token after 'name' '":"'."""
    for i in range(2, len(got)):
        if got[i - 2] == 294 and got[i - 1] == 264:
            return i
    return None


def records(path):
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(24)
            if len(hdr) < 24:
                return
            tag = hdr[:8].rstrip(b'\0').decode()
            layer, head, step, n = struct.unpack('<4i', hdr[8:])
            yield tag, layer, head, step, np.frombuffer(f.read(4 * n), dtype='<f4')


def run_trace(exe, cases, outdir):
    """Trace runner always prints '<id> got: <ids...>' and writes per-step logits."""
    os.makedirs(outdir, exist_ok=True)
    lines, meta = [], {}
    for cid, q, want in cases:
        ids, _ = bpe.build_encoder_input(tok, q, ALL)
        lines.append(f"{cid} {len(ids)} {' '.join(map(str, ids))} 1 1")
        meta[cid] = {'q': q, 'want': want, 'enc': ids}
    vf = os.path.join(outdir, 'vec.txt')
    open(vf, 'w').write("\n".join(lines) + "\n")
    r = subprocess.run([exe, NPK, vf, outdir], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{exe} failed:\n{r.stdout}\n{r.stderr}")
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] == 'got:':
            meta[p[0]]['got'] = [int(x) for x in p[2:]]
    missing = [c[0] for c in cases if 'got' not in meta[c[0]]]
    if missing:
        raise SystemExit(f"{exe} missing cases {missing}:\n{r.stdout[:1200]}")
    return meta


def load_logits(outdir, cid):
    """Return dict step -> logits vector from a case trace file."""
    path = os.path.join(outdir, f'{cid}.bin')
    out = {}
    for tag, layer, head, step, v in records(path):
        if tag == 'logit':
            out[step] = v
    return out


def margin_at_name(got, logits_by_step):
    """Margin of the emitted first-name-piece vs best rival name-piece id."""
    ns = name_step(got)
    if ns is None or ns not in logits_by_step:
        return None, None, None, None
    lg = logits_by_step[ns]
    emitted = got[ns]
    win = float(lg[emitted])
    rivals = [i for i in RIVAL_IDS if i != emitted]
    if not rivals:
        return ns, emitted, win, None
    rb = max(rivals, key=lambda i: float(lg[i]))
    return ns, emitted, win, win - float(lg[rb])


def main():
    cases = battery36()
    assert len(cases) == 36, f"expected 36, got {len(cases)}"

    tr_exact = os.path.join(REPO, 'engine', 'needle_trace')
    tr_i8 = os.path.join(REPO, 'engine', 'needle_i8_trace')
    for p in (tr_exact, tr_i8):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}; build with make -C engine needle_trace needle_i8_trace")

    print(f"== generations: exact (fp32) vs int8 attention ==")
    print(f"   npk={NPK}")
    print(f"   cases={len(cases)} (29 standard-schema + 7 eval_tools)")
    dir_e = os.path.join(OUT, 'trace_exact')
    dir_i = os.path.join(OUT, 'trace_i8')
    print("  tracing exact (fp32 scalar)...")
    exact = run_trace(tr_exact, cases, dir_e)
    print("  tracing int8 (NE_PIE + NE_ATTN_I8 + stub)...")
    i8m = run_trace(tr_i8, cases, dir_i)

    name_ok = id_ok = 0
    want_exact_ok = want_i8_ok = 0
    divs = []
    for cid, q, want in cases:
        ge, gi = exact[cid]['got'], i8m[cid]['got']
        ne, _, se = toolname(ge)
        ni, _, si = toolname(gi)
        same_name = (ne == ni)
        same_ids = (ge == gi)
        name_ok += same_name
        id_ok += same_ids
        want_e = (ne == want) if want is not None else (ne is None)
        want_i = (ni == want) if want is not None else (ni is None)
        want_exact_ok += want_e
        want_i8_ok += want_i
        mark = 'OK ' if same_ids else ('nm' if same_name else 'XX')
        print(f"  {mark} {cid:8s} want={str(want):14s} exact={str(ne):14s} i8={str(ni):14s}"
              f"  want_e={want_e} want_i={want_i}")
        if not same_ids:
            divs.append({
                'cid': cid, 'q': q, 'want': want,
                'exact_name': ne, 'i8_name': ni,
                'exact_ids': ge, 'i8_ids': gi,
                'exact_text': se, 'i8_text': si,
            })

    print(f"\n  tool-name agreement (exact vs int8): {name_ok}/{len(cases)}")
    print(f"  full byte-identical:                 {id_ok}/{len(cases)}")
    print(f"  expected-tool hit exact:             {want_exact_ok}/{len(cases)}")
    print(f"  expected-tool hit int8:              {want_i8_ok}/{len(cases)}")
    if divs:
        print(f"\n== divergent pairs ({len(divs)}) ==")
        for d in divs:
            print(f"  --- {d['cid']}: {d['q']!r}")
            print(f"      exact name={d['exact_name']!r}  ids={d['exact_ids']}")
            print(f"      text: {d['exact_text']}")
            print(f"      i8    name={d['i8_name']!r}  ids={d['i8_ids']}")
            print(f"      text: {d['i8_text']}")
    else:
        print("\n  no divergent pairs")

    # --- margin analysis ---
    print("\n== margin analysis (first-name-piece step) ==")
    shifts = []
    flips = []
    rows = []
    print(f"  {'cid':8s}  {'emit_e':6s} {'emit_i':6s}  "
          f"{'m_exact':>8s} {'m_i8':>8s} {'shift':>8s}")
    for cid, q, want in cases:
        ge = exact[cid]['got']
        gi = i8m[cid]['got']
        lg_e = load_logits(dir_e, cid)
        lg_i = load_logits(dir_i, cid)
        ns_e, em_e, win_e, m_e = margin_at_name(ge, lg_e)
        ns_i, em_i, win_i, m_i = margin_at_name(gi, lg_i)
        if m_e is None and m_i is None:
            print(f"  {cid:8s}  (no name-piece step — no-call or parse miss)")
            continue
        shift = None
        if ns_e is not None and ns_e in lg_e and ns_e in lg_i:
            em = ge[ns_e]
            rivals = [i for i in RIVAL_IDS if i != em]
            me = float(lg_e[ns_e][em]) - max(float(lg_e[ns_e][r]) for r in rivals)
            mi = float(lg_i[ns_e][em]) - max(float(lg_i[ns_e][r]) for r in rivals)
            # Aligned only when both paths share the prefix through the name piece.
            if ge[:ns_e + 1] == gi[:ns_e + 1]:
                shift = mi - me
                shifts.append(shift)
                m_e, m_i = me, mi
                rows.append({'cid': cid, 'm_exact': me, 'm_i8': mi, 'shift': shift,
                             'emitted': int(em)})
        em_e_s = str(em_e) if em_e is not None else '-'
        em_i_s = str(em_i) if em_i is not None else '-'
        sh_s = f"{shift:+8.3f}" if shift is not None else "     n/a"
        me_s = f"{m_e:+8.3f}" if m_e is not None else "     n/a"
        mi_s = f"{m_i:+8.3f}" if m_i is not None else "     n/a"
        print(f"  {cid:8s}  {em_e_s:6s} {em_i_s:6s}  {me_s} {mi_s} {sh_s}")
        if em_e is not None and em_i is not None and em_e != em_i:
            flips.append((cid, em_e, em_i, m_e, m_i))

    if shifts:
        arr = np.array(shifts)
        print(f"\n  margin shift (i8 - exact) over {len(shifts)} aligned name-steps:")
        print(f"    mean={arr.mean():+.4f}  min={arr.min():+.4f}  "
              f"max={arr.max():+.4f}  std={arr.std():+.4f}")
        # cases where exact had positive margin but int8 flipped the sign
        sign_flips = [r for r in rows if r['m_exact'] > 0 and r['m_i8'] < 0]
        print(f"    exact>0 but i8<0: {len(sign_flips)}")
    else:
        print("\n  no aligned name-steps for margin shift")
    if flips:
        print(f"\n  WINNER FLIPS at name-piece ({len(flips)}):")
        for cid, ee, ei, me, mi in flips:
            print(f"    {cid}: exact id {ee} -> i8 id {ei}  "
                  f"(margins exact={me} i8={mi})")
    else:
        print("\n  no winner flips at the first-name-piece step")

    summary = {
        'n': len(cases),
        'name_agree': name_ok,
        'byte_identical': id_ok,
        'want_exact': want_exact_ok,
        'want_i8': want_i8_ok,
        'divergences': divs,
        'margin_shift_mean': float(np.mean(shifts)) if shifts else None,
        'margin_shift_min': float(np.min(shifts)) if shifts else None,
        'margin_shift_max': float(np.max(shifts)) if shifts else None,
        'margin_shift_std': float(np.std(shifts)) if shifts else None,
        'n_shifts': len(shifts),
        'margin_rows': rows,
        'flips': [{'cid': c, 'exact': int(ee), 'i8': int(ei)} for c, ee, ei, _, _ in flips],
    }
    open(os.path.join(OUT, 'summary.json'), 'w').write(json.dumps(summary, indent=2))
    print(f"\n  wrote {OUT}/summary.json")


if __name__ == '__main__':
    main()

