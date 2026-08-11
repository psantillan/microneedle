#!/usr/bin/env python3
"""Product warm-schema F=4 acceptance + economics.

    python3 tools/eval_warmf4.py [weights/needle_tools_g512.npk]

Gates (prints verbatim for docs/notebook/RESULTS-WARMF4.md):
  1. 36-case battery: warm vs exact byte-identical (shared full-6 schema)
  2. Mutation: corrupt one cached K row -> diverge; restore -> identical
  3. Wrong-schema: key mismatch -> warm returns -1 (no silent use)
  4. Fixture parity: per-fixture matching-schema capture (schemas differ)
  5. Cache size + capture cost (from engine stderr / CACHE line)
  6. Tool-retrieval economics: size-per-subset vs full-6 cache scope

Research multi-F curve remains in tools/eval_warmschema.py + docs/notebook/RESULTS-HYBRID.md.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tokenizer'))
sys.path.insert(0, os.path.join(REPO, 'train'))
import bpe
from make_dataset import TOOLS

NPK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, 'weights', 'needle_tools_g512.npk')
OUT = os.environ.get('NE_WARMF4_DIR', '/tmp/ne_warmf4')
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
    ("What's the weather in Oslo?", "get_weather"),
    ("When does the sun set in Reykjavik?", "get_sun_times"),
    ("Tell me about tardigrades", "look_up"),
    ("Set a timer for 5 minutes", "set_timer"),
    ("Play a chime", "play_tone"),
    ("How long have you been running?", "board_status"),
    ("Write me a poem about the sea", None),
]
CLASS_TOOL = {
    'aud': 'play_tone', 'wea': 'get_weather', 'sun': 'get_sun_times',
    'lok': 'look_up', 'tim': 'set_timer', 'brd': 'board_status', 'poem': None,
}


def battery29():
    out = []
    for i, q in enumerate(AUDIO):
        out.append((f"aud{i:02d}", q, 'play_tone'))
    for k, qs in OTHER.items():
        for i, q in enumerate(qs):
            out.append((f"{k}{i:02d}", q, CLASS_TOOL[k]))
    out.append(("poem00", "Write me a poem about the sea", None))
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


def schema_ns(ids):
    try:
        nq = ids.index(5)
    except ValueError:
        return 0
    return len(ids) - nq


def build_vector_file(cases, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    lines = []
    meta = {}
    for cid, q, want in cases:
        ids, _ = bpe.build_encoder_input(tok, q, ALL)
        lines.append(f"{cid} {len(ids)} {' '.join(map(str, ids))} 1 1")
        meta[cid] = {'q': q, 'want': want, 'enc': ids, 'ns': schema_ns(ids)}
    open(path, 'w').write("\n".join(lines) + "\n")
    return meta


def ensure_exe():
    exe = os.path.join(REPO, 'engine', 'needle_warmschema')
    if not os.path.isfile(exe):
        subprocess.check_call(['make', '-C', os.path.join(REPO, 'engine'),
                               'needle_warmschema'])
    return exe


def run_engine(exe, vf, extra=None, npk=None):
    cmd = [exe, npk or NPK, vf] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def parse_battery(stdout):
    meta = {}
    cache = {}
    summary = {}
    for line in stdout.splitlines():
        m = re.match(r'^CACHE ns=(\d+) bytes=(\d+) capture_ms=([0-9.]+)$', line)
        if m:
            cache = {'ns': int(m.group(1)), 'bytes': int(m.group(2)),
                     'capture_ms': float(m.group(3))}
            continue
        m = re.match(r'^(\S+) exact:(.*)$', line)
        if m:
            meta.setdefault(m.group(1), {})['exact'] = [
                int(x) for x in m.group(2).split()]
            continue
        m = re.match(r'^(\S+) warm:(.*)$', line)
        if m:
            meta.setdefault(m.group(1), {})['warm'] = [
                int(x) for x in m.group(2).split()]
            continue
        m = re.match(r'^(\S+) warm: SKIP key_mismatch$', line)
        if m:
            meta.setdefault(m.group(1), {})['skip'] = True
            continue
        m = re.match(
            r'^(\S+) rows: (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)$', line)
        if m:
            meta.setdefault(m.group(1), {})['rows'] = tuple(
                int(m.group(i)) for i in range(2, 8))
            continue
        m = re.match(r'^SUMMARY identical=(\d+) total=(\d+) skipped_key=(\d+)$',
                     line)
        if m:
            summary = {'identical': int(m.group(1)), 'total': int(m.group(2)),
                       'skipped': int(m.group(3))}
    return cache, meta, summary


def score_set(name, cases, meta):
    name_ok_e = name_ok_w = ident = 0
    rows = []
    divs = []
    for cid, q, want in cases:
        e_ids = meta.get(cid, {}).get('exact', [])
        w_ids = meta.get(cid, {}).get('warm', [])
        en, ea, eraw = toolname(e_ids)
        wn, wa, wraw = toolname(w_ids)
        e_hit = (en == want) if want is not None else (en is None)
        w_hit = (wn == want) if want is not None else (wn is None)
        name_ok_e += e_hit
        name_ok_w += w_hit
        same = e_ids == w_ids and not meta.get(cid, {}).get('skip')
        ident += same
        if meta.get(cid, {}).get('rows'):
            rows.append(meta[cid]['rows'])
        mark = 'OK' if same else 'XX'
        print(f"  {mark} {cid:8s} want={str(want):<14} "
              f"exact={str(en):<14} warm={str(wn):<14}  {q[:40]}")
        if not same:
            divs.append({'cid': cid, 'q': q, 'exact_raw': eraw, 'warm_raw': wraw,
                         'exact_ids': e_ids, 'warm_ids': w_ids})
    n = len(cases)
    avg_rl = sum(r[3] for r in rows) / len(rows) if rows else 0
    avg_rf = sum(r[4] for r in rows) / len(rows) if rows else 1
    frac = avg_rl / avg_rf if avg_rf else 0
    print(f"\n  [{name}] tool-name exact {name_ok_e}/{n} warm {name_ok_w}/{n}")
    print(f"  [{name}] byte-identical {ident}/{n}")
    if rows:
        print(f"  [{name}] row-layers mean {avg_rl:.0f}/{avg_rf:.0f} "
              f"= {100*frac:.1f}% ({avg_rf/avg_rl:.2f}x)" if avg_rl else "")
    return {'n': n, 'name_exact': name_ok_e, 'name_warm': name_ok_w,
            'identical': ident, 'divs': divs, 'avg_rl': avg_rl, 'avg_rf': avg_rf,
            'frac': frac}


# ---- tool-retrieval economics (mirror of toolrag scorer, inlined subset) ----

# Lightweight lexical scorer so we do not hard-depend on the toolrag worktree.
_KEYWORDS = {
    'play_tone': ['play', 'sound', 'noise', 'chime', 'alarm', 'success', 'tone',
                  'audio', 'beep'],
    'get_weather': ['weather', 'rain', 'hot', 'cold', 'temperature', 'forecast'],
    'get_sun_times': ['sun', 'sunset', 'sunrise', 'golden', 'hour', 'dawn', 'dusk'],
    'look_up': ['look', 'up', 'about', 'who', 'what', 'tell', 'quasar',
                'tardigrade', 'lovelace', 'antikythera'],
    'set_timer': ['timer', 'countdown', 'minutes', 'minute', 'hour', 'hours'],
    'board_status': ['board', 'running', 'memory', 'chip', 'free', 'status'],
}


def _tokset(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def select_tools_k(query, tools, k=2):
    q = _tokset(query)
    scored = []
    for t in tools:
        name = t['name']
        pieces = set(name.split('_')) | _tokset(t.get('description', '')[:80])
        pieces |= set(_KEYWORDS.get(name, []))
        score = len(q & pieces)
        # bigrams
        ql = re.findall(r"[a-z0-9]+", query.lower())
        for a, b in zip(ql, ql[1:]):
            if a + b in ''.join(pieces) or a + '_' + b in name:
                score += 1
        scored.append((score, t))
    scored.sort(key=lambda x: (-x[0], x[1]['name']))
    # floor + OOD fallback
    if not scored or scored[0][0] < 1:
        return list(tools)  # full catalogue
    top = [t for s, t in scored if s >= 2][:k]
    if len(top) < k:
        for s, t in scored:
            if t not in top:
                top.append(t)
            if len(top) >= k:
                break
    if not top:
        return list(tools)
    return top


def cache_bytes_formula(ns, F=4, dmodel=512, kvdim=256):
    """General formula (fp32): F*ns*kvdim*4*2 + ns*dmodel*4 + ns*4."""
    return (F * ns * kvdim * 4 * 2) + (ns * dmodel * 4) + (ns * 4)


def toolrag_economics(exe):
    print("\n" + "=" * 60)
    print("== tool-retrieval (k=2) cache economics ==")
    print("=" * 60)
    # Full-6 schema size (measured via a capture on a single prompt)
    ids_full, _ = bpe.build_encoder_input(tok, "Play a chime", ALL)
    ns_full = schema_ns(ids_full)
    # Distinct k=2 schemas across the battery
    b29 = battery29()
    e7 = [(f"e7_{i}", q, w) for i, (q, w) in enumerate(EVAL7)]
    all_q = [(cid, q) for cid, q, _ in b29 + e7]
    subsets = {}
    for cid, q in all_q:
        sel = select_tools_k(q, ALL, k=2)
        key = tuple(sorted(t['name'] for t in sel))
        ids, _ = bpe.build_encoder_input(tok, q, sel)
        ns = schema_ns(ids)
        subsets.setdefault(key, {'ns': ns, 'n': 0, 'example': cid})
        subsets[key]['n'] += 1
        # ns should be stable for a given tool set (JSON order may vary)
        subsets[key]['ns'] = ns

    print(f"  full-6 schema ns={ns_full}  formula_bytes={cache_bytes_formula(ns_full)}")
    print(f"  distinct k=2 tool-subsets observed: {len(subsets)}")
    print(f"  {'subset':<50} {'n':>3} {'ns':>4} {'bytes':>10}")
    total_if_all = 0
    for key, info in sorted(subsets.items(), key=lambda x: -x[1]['n']):
        b = cache_bytes_formula(info['ns'])
        total_if_all += b
        print(f"  {','.join(key):<50} {info['n']:3d} {info['ns']:4d} {b:10d}")
    print(f"  sum of one cache per distinct subset: {total_if_all} bytes "
          f"({total_if_all/1024:.1f} KiB)")
    print(f"  single full-6 cache:                 {cache_bytes_formula(ns_full)} bytes "
          f"({cache_bytes_formula(ns_full)/1024:.1f} KiB)")
    print("  v1 scope: cache FULL six-tool schema only (retrieval OFF / fallback).")
    print("  Rationale: k=2 yields many distinct subsets; multi-entry cache would")
    print("  multiply memory. Full-6 still wins for fixed-schema deployments and")
    print("  the OOD full-list fallback path under retrieval.")
    return {
        'ns_full': ns_full,
        'bytes_full': cache_bytes_formula(ns_full),
        'n_subsets': len(subsets),
        'bytes_all_subsets': total_if_all,
        'subsets': {','.join(k): v for k, v in subsets.items()},
    }


def fixture_warm_parity(exe, npk):
    """Each of the 8 parity fixtures has its own tools JSON — no shared schema.
    Per fixture: capture on that prompt, warm-encode same prompt, compare to exact.
    """
    print("\n" + "=" * 60)
    print("== fixture warm parity (per-fixture matching-schema capture) ==")
    print("=" * 60)
    vec = os.path.join(REPO, 'weights', 'vectors_int4p_int8e.txt')
    fixtures = []
    with open(vec) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            cid = parts[0]
            n_enc = int(parts[1])
            enc = list(map(int, parts[2:2 + n_enc]))
            fixtures.append((cid, enc, n_enc))
    schemas = []
    for cid, enc, n_enc in fixtures:
        try:
            nq = enc.index(5)
        except ValueError:
            nq = 0
        schemas.append(tuple(enc[nq:]))
    n_unique = len(set(schemas))
    print(f"  fixtures={len(fixtures)} unique_schema_keys={n_unique}")
    print("  (no single shared schema across the 8 parity fixtures)")
    print(f"  npk={npk}")

    ok = 0
    for cid, enc, n_enc in fixtures:
        path = os.path.join(OUT, f'fix_{cid}.txt')
        open(path, 'w').write(
            f"{cid} {n_enc} {' '.join(map(str, enc))} 1 1\n")
        r = run_engine(exe, path, npk=npk)
        if r.returncode != 0:
            print(f"  XX {cid}: engine failed\n{r.stderr}")
            continue
        _, meta, summary = parse_battery(r.stdout)
        same = summary.get('identical', 0) == 1
        ok += same
        mark = 'OK' if same else 'XX'
        rows = meta.get(cid, {}).get('rows')
        ns = rows[1] if rows else '?'
        print(f"  {mark} {cid}: identical={summary.get('identical')} ns={ns}")
    print(f"\n  fixture warm parity: {ok}/{len(fixtures)}")
    return ok, len(fixtures), n_unique


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"npk={NPK}")
    print(f"NE_WARM_F=4 (product)")
    exe = ensure_exe()

    b29 = battery29()
    e7 = [(f"e7_{i}", q, w) for i, (q, w) in enumerate(EVAL7)]
    seen = set()
    all_cases = []
    for c in b29 + e7:
        if c[1] in seen and c[0].startswith('e7_'):
            all_cases.append(c)
        else:
            seen.add(c[1])
            all_cases.append(c)

    vf = os.path.join(OUT, 'vec36.txt')
    meta0 = build_vector_file(all_cases, vf)

    # ---- Gate: 36-case battery ----
    print("\n" + "=" * 60)
    print("== F=4 product battery (36 cases, full-6 schema cache) ==")
    print("=" * 60)
    t0 = time.time()
    r = run_engine(exe, vf)
    wall = time.time() - t0
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"needle_warmschema failed: {r.returncode}")
    print(r.stderr, file=sys.stderr)
    cache, meta, summary = parse_battery(r.stdout)
    print(f"  CACHE measured: ns={cache.get('ns')} bytes={cache.get('bytes')} "
          f"capture_ms={cache.get('capture_ms')}")
    formula = cache_bytes_formula(cache.get('ns', 0))
    print(f"  formula_bytes={formula}  measured_matches_formula="
          f"{cache.get('bytes') == formula}")
    print(f"  engine wall for 36 exact+warm: {wall:.2f}s")

    s29 = score_set('29-case battery', b29, meta)
    s7 = score_set('7-case eval_tools', e7, meta)
    ti = s29['identical'] + s7['identical']
    tn = s29['n'] + s7['n']
    tw = s29['name_warm'] + s7['name_warm']
    te = s29['name_exact'] + s7['name_exact']
    all_rows = []
    for cid, _, _ in all_cases:
        if meta.get(cid, {}).get('rows'):
            all_rows.append(meta[cid]['rows'])
    avg_rl = sum(r[3] for r in all_rows) / len(all_rows) if all_rows else 0
    avg_rf = sum(r[4] for r in all_rows) / len(all_rows) if all_rows else 1
    print(f"\n  OVERALL F=4: tool-name warm {tw}/{tn} (exact {te}/{tn})  "
          f"byte-identical {ti}/{tn}")
    print(f"  encoder work mean {avg_rl:.0f}/{avg_rf:.0f} = "
          f"{100*avg_rl/avg_rf:.1f}% ({avg_rf/avg_rl:.2f}x saving)")
    if ti != tn:
        print("  *** FAIL: not byte-identical on all cases ***")
        for d in s29['divs'] + s7['divs']:
            print(f"    [{d['cid']}] exact={d['exact_ids'][:8]}... "
                  f"warm={d['warm_ids'][:8]}...")
        raise SystemExit(1)

    # ---- Mutation ----
    print("\n" + "=" * 60)
    print("== mutation test ==")
    print("=" * 60)
    r = run_engine(exe, vf, ['--mutate'])
    print(r.stdout)
    if r.returncode != 0 or 'MUTATE_OK 1' not in r.stdout:
        print(r.stderr)
        raise SystemExit("mutation test FAILED")
    print("  mutation gate: PASS")

    # ---- Wrong schema ----
    print("\n" + "=" * 60)
    print("== wrong-schema negative control ==")
    print("=" * 60)
    r = run_engine(exe, vf, ['--wrong-schema'])
    print(r.stdout)
    print(r.stderr, file=sys.stderr)
    if r.returncode != 0 or 'WRONG_SCHEMA_OK 1' not in r.stdout:
        raise SystemExit("wrong-schema guard FAILED")
    print("  wrong-schema gate: PASS")

    # ---- Fixtures (base v3 weights; each fixture has its own tools JSON) ----
    v3 = os.path.join(REPO, 'weights', 'needle_v3_g512.npk')
    if os.path.isfile(v3):
        f_ok, f_n, f_unique = fixture_warm_parity(exe, v3)
    else:
        print("  (no needle_v3_g512.npk — skip fixture warm parity)")
        f_ok, f_n, f_unique = 0, 0, 0

    econ = toolrag_economics(exe)

    # ---- Verdict block ----
    print("\n" + "=" * 60)
    print("== verdict ==")
    print("=" * 60)
    print(f"  battery byte-identical: {ti}/{tn}")
    print(f"  cache bytes (measured): {cache.get('bytes')}  ns={cache.get('ns')}")
    print(f"  capture_ms (host):      {cache.get('capture_ms')}")
    print(f"  prefill saving:         {avg_rf/avg_rl:.2f}x row-layers")
    print(f"  fixture warm parity:    {f_ok}/{f_n} (unique schemas={f_unique})")
    print(f"  mutation:               PASS")
    print(f"  wrong-schema guard:     PASS")
    print(f"  toolrag subsets@k=2:    {econ['n_subsets']} distinct "
          f"(full-6 v1 scope recommended)")
    print("done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
