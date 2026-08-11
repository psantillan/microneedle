#!/usr/bin/env python3
"""Client-side tool retrieval: prune the tools schema before the encoder sees it.

MicroNeedle's prefill cost is dominated by the tools JSON. This module selects
top-k candidate tools for a query with lexical features only (no model, no
embeddings), then the measurement harness scores the pruned schema through
engine/needle_trace.

The scorer is mirrored in web/tool_retrieval.js — same changed-together contract
as tokenizer/bpe.py ↔ web/bpe.js. Keep the algorithms identical; the JS twin is
asserted from this harness via node.

    python3 tools/tool_retrieval.py [weights/needle_tools_g512.npk]
    python3 tools/tool_retrieval.py --score-only          # no engine
    python3 tools/tool_retrieval.py --mutate-synonyms     # mutation gate
    python3 tools/tool_retrieval.py --twin-check          # JS parity only

Design (TinyAgent-style retrieval, lexical):
  - score = weighted overlap of query tokens against tool name pieces,
    description words, and a per-tool keyword table (data, not code)
  - select top-k by score; always include any tool that clears an absolute floor
  - when every score is near-zero (OOD like the poem), fall back to the full
    catalogue so a retrieval miss cannot silently kill a routable query
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "train"))
import bpe  # noqa: E402
from make_dataset import TOOLS  # noqa: E402

# ---------------------------------------------------------------------------
# Synonym / keyword table — data, not code. New tools carry their own keywords.
# Mirrored in web/tool_retrieval.js (TOOL_KEYWORDS). Change them together.
# ---------------------------------------------------------------------------
TOOL_KEYWORDS: dict[str, list[str]] = {
    "get_weather": [
        "weather", "raining", "rain", "hot", "cold", "temperature", "temp",
        "jacket", "conditions", "forecast", "humid", "wind", "climate",
    ],
    "get_sun_times": [
        "sunset", "sunrise", "golden", "daylight", "dusk", "dawn", "sun",
        "daylight", "dark", "light", "hour",
    ],
    "look_up": [
        "look", "lookup", "facts", "explain", "search", "tell", "who",
        "background", "know", "topic", "info", "information",
    ],
    "set_timer": [
        "timer", "countdown", "minutes", "minute", "remind", "wake",
        "clock", "alarm",  # "alarm" also on play_tone; timer phrases win via "timer"
    ],
    "play_tone": [
        "sound", "noise", "play", "chime", "alarm", "success", "tone",
        "beep", "jingle", "ring",
    ],
    "board_status": [
        "board", "uptime", "memory", "running", "chip", "device", "status",
        "ram", "diagnostics", "healthy", "ip", "psram", "free", "yourself",
        "system", "cpu", "esp",
    ],
}

# Query words ignored for description/name soft matches. Keyword hits still count
# (so "who"/"what" can fire look_up via the keyword table without "what is the
# weather" being decided solely by the stopword).
STOPWORDS = frozenset({
    "a", "an", "the", "me", "you", "i", "we", "my", "your", "it", "its",
    "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
    "in", "on", "at", "for", "of", "to", "and", "or", "but", "with",
    "from", "by", "as", "if", "so", "than", "that", "this", "these",
    "those", "there", "here", "when", "how", "where", "which", "whom",
    "can", "could", "would", "should", "will", "just", "please", "some",
    "any", "into", "out", "up", "down", "over", "under", "about",  # "about"
    # stays a stopword so poem "about the sea" does not pin look_up alone
    "what",  # keyword-table hit still scores; soft desc match does not
})

# Selection knobs — mirrored in web/tool_retrieval.js.
SCORE_FLOOR = 2.0       # always include tools at or above this
NEAR_ZERO = 2.0         # max score below this → full catalogue fallback
NAME_W = 3.0
KEYWORD_W = 2.0
DESC_W = 1.0
BIGRAM_W = 2.5

# ---- battery (29 probe cases + 7 eval_tools cases = 36) --------------------
AUDIO = [
    "Play a chime", "play the alarm sound", "make a success noise",
    "sound the alarm", "Play a success", "make a chime noise",
    "Play an alarm", "sound the chime",
]
OTHER = {
    "wea": ["What's the weather in Oslo?", "weather in San Diego",
            "Is it raining in London?", "How hot is it in Phoenix?"],
    "sun": ["When does the sun set in Reykjavik?", "sunrise time in Tokyo",
            "When is sunset in Lisbon?", "golden hour in Denver"],
    "lok": ["Tell me about tardigrades", "look up the Antikythera mechanism",
            "What is a quasar?", "Who was Ada Lovelace?"],
    "tim": ["Set a timer for 5 minutes", "start a 20 minute timer",
            "countdown 3 minutes", "timer for half an hour"],
    "brd": ["How long have you been running?", "board status",
            "How much memory is free?", "What chip are you running on?"],
}
EVAL7 = [
    ("eval_wea", "What's the weather in Oslo?", "get_weather"),
    ("eval_sun", "When does the sun set in Reykjavik?", "get_sun_times"),
    ("eval_lok", "Tell me about tardigrades", "look_up"),
    ("eval_tim", "Set a timer for 5 minutes", "set_timer"),
    ("eval_aud", "Play a chime", "play_tone"),
    ("eval_brd", "How long have you been running?", "board_status"),
    ("eval_poem", "Write me a poem about the sea", None),
]
CLASS_TOOL = {
    "aud": "play_tone", "wea": "get_weather", "sun": "get_sun_times",
    "lok": "look_up", "tim": "set_timer", "brd": "board_status", "poem": None,
}

ALL_TOOLS = list(TOOLS.values())
tok = bpe.NeedleBPE()


def battery36():
    """29 standard-schema probe cases + 7 eval_tools cases."""
    out = []
    for i, q in enumerate(AUDIO):
        out.append((f"aud{i:02d}", q, "play_tone"))
    for k, qs in OTHER.items():
        for i, q in enumerate(qs):
            out.append((f"{k}{i:02d}", q, CLASS_TOOL[k]))
    out.append(("poem00", "Write me a poem about the sea", None))
    for cid, q, want in EVAL7:
        out.append((cid, q, want))
    return out


# ---------------------------------------------------------------------------
# Scorer — keep in lockstep with web/tool_retrieval.js
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def stem(w: str) -> str:
    """Very light stemmer — same rules as the JS twin."""
    for suf in ("ings", "ing", "tion", "ness", "ment", "ies", "ied",
                "ers", "est", "ed", "es", "ly", "er", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            # do not strip the trailing 's' from short stems that would collide
            # ("as", "is" already stopworded; "status"→"statu" is fine)
            return w[: -len(suf)]
    return w


def _bag(words: list[str]) -> set[str]:
    b: set[str] = set()
    for w in words:
        b.add(w)
        b.add(stem(w))
    return b


def score_tool(query: str, tool: dict, keywords: dict[str, list[str]] | None = None) -> float:
    """Lexical score of one tool against the query. Pure function of strings.

    Description is scored only up to a colon (if any). Example values after a
    colon ("chime, alarm, or success") live in TOOL_KEYWORDS so the synonym
    table is the real lever — mutation of those keywords must be able to fail.
    """
    kw_table = keywords if keywords is not None else TOOL_KEYWORDS
    name = (tool.get("name") or "").replace("_", " ")
    desc_full = tool.get("description") or ""
    desc = desc_full.split(":")[0]
    kws = kw_table.get(tool.get("name") or "", [])

    name_bag = _bag(tokenize(name))
    desc_bag = _bag(tokenize(desc))
    kw_bag = _bag(list(kws))

    q = tokenize(query)
    score = 0.0
    hit = set()  # avoid double-counting the same query token

    for w in q:
        if w in hit:
            continue
        sw = stem(w)
        if w in kw_bag or sw in kw_bag:
            score += KEYWORD_W
            hit.add(w)
        elif w not in STOPWORDS and (w in name_bag or sw in name_bag):
            score += NAME_W
            hit.add(w)
        elif w not in STOPWORDS and (w in desc_bag or sw in desc_bag):
            score += DESC_W
            hit.add(w)

    # bigrams catch "sun set" → sunset, "look up" → lookup
    for i in range(len(q) - 1):
        bi = q[i] + q[i + 1]
        sbi = stem(q[i]) + stem(q[i + 1])
        if bi in kw_bag or sbi in kw_bag or bi in name_bag or bi in desc_bag:
            score += BIGRAM_W
            break  # one bigram bonus is enough

    return score


def score_all(query: str, tools: list[dict] | None = None,
              keywords: dict[str, list[str]] | None = None) -> list[tuple[float, dict]]:
    tools = tools if tools is not None else ALL_TOOLS
    scored = [(score_tool(query, t, keywords), t) for t in tools]
    scored.sort(key=lambda x: (-x[0], x[1].get("name") or ""))
    return scored


def select_tools(query: str, tools: list[dict] | None = None, k: int = 2,
                 floor: float = SCORE_FLOOR, near_zero: float = NEAR_ZERO,
                 keywords: dict[str, list[str]] | None = None) -> list[dict]:
    """Top-k by score, union tools clearing `floor`; full list if all near-zero.

    Returned order follows score (high→low), then name — stable for the twin
    check. Callers that care about catalogue order can re-sort.
    """
    tools = tools if tools is not None else ALL_TOOLS
    scored = score_all(query, tools, keywords)
    if not scored:
        return []
    max_s = scored[0][0]
    if max_s < near_zero:
        # OOD / no signal: never silently drop the catalogue
        return list(tools)

    selected: list[dict] = []
    seen: set[str] = set()
    for s, t in scored:
        name = t.get("name") or ""
        if name in seen:
            continue
        if len(selected) < k or s >= floor:
            selected.append(t)
            seen.add(name)
    return selected


# ---------------------------------------------------------------------------
# Measurement harness
# ---------------------------------------------------------------------------
def toolname(ids: list[int]):
    s = tok.decode(ids)
    try:
        c = json.loads(s.strip())
        if not c:
            return None, None, s.strip()
        return c[0].get("name"), c[0].get("arguments"), s.strip()
    except Exception:
        return "(unparseable)", None, s.strip()


def encode_case(query: str, tools: list[dict]):
    ids, tj = bpe.build_encoder_input(tok, query, tools)
    return ids, tj


def run_engine(npk: str, cases: list[tuple[str, str, list[dict]]], out_dir: str):
    """Build a vector file with the given per-case tool lists and run needle_trace."""
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    meta = {}
    for cid, q, tools in cases:
        ids, tj = encode_case(q, tools)
        lines.append(f"{cid} {len(ids)} {' '.join(map(str, ids))} 1 1")
        meta[cid] = {"q": q, "tools": [t["name"] for t in tools],
                     "n_tools": len(tools), "enc": ids, "n_enc": len(ids),
                     "tools_json": tj}
    vf = os.path.join(out_dir, "vec.txt")
    open(vf, "w").write("\n".join(lines) + "\n")
    exe = os.path.join(REPO, "engine", "needle_trace")
    if not os.path.isfile(exe):
        raise SystemExit(f"missing {exe}; build with: make -C engine needle_trace")
    r = subprocess.run([exe, npk, vf, out_dir], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"needle_trace failed (exit {r.returncode}):\n"
                         f"{r.stdout}\n{r.stderr}")
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] == "got:":
            meta[p[0]]["got"] = [int(x) for x in p[2:]]
    missing = [cid for cid, _, _ in cases if "got" not in meta.get(cid, {})]
    if missing:
        raise SystemExit(f"needle_trace returned no output for {missing}\n"
                         f"{(r.stderr or r.stdout)[:800]}")
    return meta


def retrieval_recall(cases_want, k: int, keywords=None):
    """Per-case: is the correct tool present in the pruned set (or full on OOD)?"""
    hits = []
    for cid, q, want in cases_want:
        selected = select_tools(q, ALL_TOOLS, k=k, keywords=keywords)
        names = [t["name"] for t in selected]
        if want is None:
            # OOD: success if we fell back to full list (model needs the chance
            # to emit []) OR if empty catalogue — we require full list
            ok = len(selected) == len(ALL_TOOLS)
        else:
            ok = want in names
        hits.append((cid, q, want, names, ok, score_all(q, keywords=keywords)))
    return hits


def measure(npk: str, ks=(1, 2, 3), keywords=None, label=""):
    bat = battery36()
    results = {}
    # full-6 baseline first
    configs = [("full6", None)] + [(f"k{k}", k) for k in ks]
    for cfg_name, k in configs:
        case_vecs = []
        for cid, q, want in bat:
            if k is None:
                tools = ALL_TOOLS
            else:
                tools = select_tools(q, ALL_TOOLS, k=k, keywords=keywords)
            case_vecs.append((cid, q, tools))
        out_dir = os.path.join(tempfile.gettempdir(), f"ne_toolrag_{cfg_name}{label}")
        print(f"\n== engine run: {cfg_name} ({len(case_vecs)} cases) → {out_dir} ==")
        meta = run_engine(npk, case_vecs, out_dir)

        name_ok = 0
        n_encs = []
        divs = []
        recall_ok = 0
        per = []
        for cid, q, want in bat:
            tools_used = meta[cid]["tools"]
            n_enc = meta[cid]["n_enc"]
            n_encs.append(n_enc)
            got_name, got_args, raw = toolname(meta[cid]["got"])
            hit = (got_name == want) if want is not None else (got_name is None)
            if hit:
                name_ok += 1
            if want is None:
                r_ok = len(tools_used) == len(ALL_TOOLS)
            else:
                r_ok = want in tools_used
            if r_ok:
                recall_ok += 1
            row = {
                "cid": cid, "q": q, "want": want, "got": got_name,
                "args": got_args, "hit": hit, "recall": r_ok,
                "tools": tools_used, "n_enc": n_enc, "raw": raw,
            }
            per.append(row)
            if not hit or not r_ok:
                divs.append(row)

        results[cfg_name] = {
            "name_acc": name_ok,
            "n": len(bat),
            "recall": recall_ok,
            "mean_enc": sum(n_encs) / len(n_encs),
            "per": per,
            "divs": divs,
        }
        print(f"  recall {recall_ok}/{len(bat)}  "
              f"name-acc {name_ok}/{len(bat)}  "
              f"mean enc tokens {results[cfg_name]['mean_enc']:.1f}")
    return results


def print_table(results):
    base = results["full6"]["mean_enc"]
    print("\n## k-sweep table")
    print(f"{'cfg':<8} {'recall':>10} {'name-acc':>10} {'mean_enc':>10} {'vs full':>10}")
    for cfg, r in results.items():
        ratio = r["mean_enc"] / base if base else 0
        print(f"{cfg:<8} {r['recall']:>4}/{r['n']:<5} {r['name_acc']:>4}/{r['n']:<5} "
              f"{r['mean_enc']:>10.1f} {ratio:>10.3f}")


def print_divergences(results):
    print("\n## divergences (name miss and/or recall miss)")
    for cfg, r in results.items():
        print(f"\n### {cfg}")
        if not r["divs"]:
            print("  (none)")
            continue
        for d in r["divs"]:
            print(f"  {d['cid']:12s} want={d['want']!s:<14} got={d['got']!s:<14} "
                  f"recall={'Y' if d['recall'] else 'N'} hit={'Y' if d['hit'] else 'N'} "
                  f"tools={d['tools']}  q={d['q']!r}")
            if d.get("raw"):
                print(f"               raw={d['raw'][:120]}")


def compare_to_full(results):
    """Cases where pruning changes the end-task answer vs full6."""
    full = {p["cid"]: p for p in results["full6"]["per"]}
    print("\n## pruning vs full6 (answer changes)")
    any_change = False
    for cfg, r in results.items():
        if cfg == "full6":
            continue
        for p in r["per"]:
            f = full[p["cid"]]
            if p["got"] != f["got"]:
                any_change = True
                fixed = (not f["hit"]) and p["hit"]
                broke = f["hit"] and (not p["hit"])
                tag = "FIX" if fixed else ("REGRESS" if broke else "CHANGE")
                print(f"  [{tag}] {cfg} {p['cid']:12s} full={f['got']!s:<14} "
                      f"pruned={p['got']!s:<14} want={p['want']!s}  q={p['q']!r}")
    if not any_change:
        print("  (no answer changes)")


# ---------------------------------------------------------------------------
# Mutation test: remove play_tone sound keywords → aud recall must drop
# ---------------------------------------------------------------------------
def mutate_keywords():
    """Break the synonym table: strip play_tone's sound/noise/play/... keywords."""
    kw = {k: list(v) for k, v in TOOL_KEYWORDS.items()}
    sound = {"sound", "noise", "play", "chime", "alarm", "success", "tone",
             "beep", "jingle", "ring"}
    kw["play_tone"] = [w for w in kw["play_tone"] if w not in sound]
    return kw


def _aud_signal_recall(aud, k, keywords):
    """Recall that does NOT credit OOD full-list fallback.

    In-distribution audio queries must rank play_tone into the selected set with
    a real score signal. Fallback-to-full still keeps play_tone present, so the
    plain presence metric cannot see a synonym-table break — this one can.
    """
    rows = []
    for cid, q, want in aud:
        scored = score_all(q, keywords=keywords)
        max_s = scored[0][0] if scored else 0.0
        selected = select_tools(q, ALL_TOOLS, k=k, keywords=keywords)
        names = [t["name"] for t in selected]
        fell_back = max_s < NEAR_ZERO
        ok = (not fell_back) and (want in names)
        rows.append((cid, q, want, names, ok, fell_back, scored))
    return rows


def mutation_test():
    bat = battery36()
    aud = [(c, q, w) for c, q, w in bat if c.startswith("aud") or c == "eval_aud"]
    print("\n## mutation test: strip play_tone sound keywords")
    print("  (signal-recall: want in top-k with score ≥ NEAR_ZERO; no fallback credit)")
    scores = {}
    for label, kw in (("intact", TOOL_KEYWORDS), ("mutated", mutate_keywords())):
        rows = _aud_signal_recall(aud, k=2, keywords=kw)
        ok = sum(1 for *_, h, __, ___ in rows if h)
        scores[label] = ok
        print(f"  {label:8s} aud signal-recall@2 = {ok}/{len(aud)}")
        if label == "mutated":
            for cid, q, want, names, h, fell, scored in rows:
                if h:
                    continue
                top = [(s, t["name"]) for s, t in scored[:3]]
                why = "fallback" if fell else "wrong-top"
                print(f"    miss ({why}) {cid}: tools={names} top={top} q={q!r}")
        if label == "intact" and ok < len(aud):
            print("  WARNING: intact signal-recall already incomplete")
    if scores["mutated"] >= scores["intact"]:
        print("  FAIL: mutation did not drop aud signal-recall")
        return False
    print(f"  OK: signal-recall dropped {scores['intact']} → {scores['mutated']}")
    return True


# ---------------------------------------------------------------------------
# JS twin parity: same top-k sets on all 36 queries via node
# ---------------------------------------------------------------------------
TWIN_JS = r"""
import { selectTools, scoreAll } from './tool_retrieval.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// argv: cases.json tools.json k   — paths may be absolute
const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const tools = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const k = parseInt(process.argv[4] || '2', 10);
const out = [];
for (const {cid, q} of cases) {
  const sel = selectTools(q, tools, k);
  const scored = scoreAll(q, tools).map(([s, t]) => [s, t.name]);
  out.push({cid, q, names: sel.map(t => t.name), scores: scored});
}
process.stdout.write(JSON.stringify(out));
"""


def twin_check(ks=(1, 2, 3)):
    """Assert JS and Python scorers agree on top-k sets for every battery query.

    Verification method: write a small ESM harness next to web/tool_retrieval.js,
    run it with node for each k, compare selected name lists to select_tools().
    """
    bat = battery36()
    tools_json = json.dumps(ALL_TOOLS)
    cases_json = json.dumps([{"cid": c, "q": q} for c, q, _ in bat])
    web_dir = os.path.join(REPO, "web")
    with tempfile.TemporaryDirectory() as td:
        cases_path = os.path.join(td, "cases.json")
        tools_path = os.path.join(td, "tools.json")
        # script must live under web/ so the relative import resolves
        script_path = os.path.join(web_dir, "_twin_check_toolrag.mjs")
        open(cases_path, "w").write(cases_json)
        open(tools_path, "w").write(tools_json)
        open(script_path, "w").write(TWIN_JS)
        print("\n## JS ↔ Python twin parity")
        print("  method: node web/_twin_check_toolrag.mjs (imports web/tool_retrieval.js)")
        all_ok = True
        try:
            for k in ks:
                r = subprocess.run(
                    ["node", script_path, cases_path, tools_path, str(k)],
                    capture_output=True, text=True, cwd=REPO,
                )
                if r.returncode != 0:
                    print(f"  node failed for k={k}:\n{r.stderr or r.stdout}")
                    return False
                js_out = json.loads(r.stdout)
                js_by = {row["cid"]: row for row in js_out}
                mismatches = []
                for cid, q, _ in bat:
                    py_names = [t["name"] for t in select_tools(q, ALL_TOOLS, k=k)]
                    js_names = js_by[cid]["names"]
                    if py_names != js_names:
                        mismatches.append((cid, q, py_names, js_names))
                if mismatches:
                    all_ok = False
                    print(f"  k={k}: FAIL {len(mismatches)} mismatches")
                    for cid, q, py, js in mismatches[:10]:
                        print(f"    {cid}: py={py} js={js} q={q!r}")
                else:
                    print(f"  k={k}: OK same top-k sets on all {len(bat)} queries")
        finally:
            try:
                os.remove(script_path)
            except OSError:
                pass
        return all_ok


def dump_scores():
    print("## per-query scores (k=2 selection)")
    for cid, q, want in battery36():
        scored = score_all(q)
        sel = select_tools(q, k=2)
        names = [t["name"] for t in sel]
        top = ", ".join(f"{t['name']}={s:.1f}" for s, t in scored[:4])
        mark = "OK" if (want in names if want else len(sel) == 6) else "MISS"
        print(f"  {mark:4s} {cid:12s} want={want!s:<14} sel={names}  [{top}]  {q!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npk", nargs="?",
                    default=os.path.join(REPO, "weights", "needle_tools_g512.npk"))
    ap.add_argument("--score-only", action="store_true",
                    help="print retrieval scores / recall, skip the engine")
    ap.add_argument("--mutate-synonyms", action="store_true",
                    help="run the synonym-table mutation gate and exit")
    ap.add_argument("--twin-check", action="store_true",
                    help="JS/Python twin parity only")
    ap.add_argument("--ks", default="1,2,3", help="comma-separated k values")
    args = ap.parse_args()
    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())

    if args.mutate_synonyms:
        ok = mutation_test()
        sys.exit(0 if ok else 1)
    if args.twin_check:
        ok = twin_check(ks=ks)
        sys.exit(0 if ok else 1)

    print("tools/tool_retrieval.py — client-side tool retrieval harness")
    dump_scores()
    print()
    # recall table without engine
    print("## retrieval recall (no engine)")
    for k in ks:
        hits = retrieval_recall(battery36(), k=k)
        ok = sum(1 for *_, h, __ in hits if h)
        print(f"  recall@{k} = {ok}/{len(hits)}")
        if ok < len(hits):
            for cid, q, want, names, h, _ in hits:
                if not h:
                    print(f"    MISS {cid} want={want} got_tools={names} q={q!r}")

    mut_ok = mutation_test()
    twin_ok = twin_check(ks=ks)

    if args.score_only:
        print(f"\nmutation={'PASS' if mut_ok else 'FAIL'}  twin={'PASS' if twin_ok else 'FAIL'}")
        sys.exit(0 if mut_ok and twin_ok else 1)

    if not os.path.isfile(args.npk):
        raise SystemExit(f"npk not found: {args.npk}")

    results = measure(args.npk, ks=ks)
    print_table(results)
    print_divergences(results)
    compare_to_full(results)

    # summary for docs/notebook/RESULTS-TOOLRAG.md consumers
    summary_path = os.path.join(REPO, "tools", "_toolrag_summary.json")
    serializable = {}
    for cfg, r in results.items():
        serializable[cfg] = {
            "name_acc": r["name_acc"], "n": r["n"], "recall": r["recall"],
            "mean_enc": r["mean_enc"],
            "divs": [{k: d[k] for k in ("cid", "q", "want", "got", "hit", "recall",
                                         "tools", "n_enc", "raw")} for d in r["divs"]],
            "per": [{k: p[k] for k in ("cid", "q", "want", "got", "hit", "recall",
                                        "tools", "n_enc")} for p in r["per"]],
        }
    serializable["_meta"] = {
        "mutation_pass": mut_ok, "twin_pass": twin_ok,
        "full6_mean_enc": results["full6"]["mean_enc"],
    }
    open(summary_path, "w").write(json.dumps(serializable, indent=2))
    print(f"\nwrote {summary_path}")
    print(f"mutation={'PASS' if mut_ok else 'FAIL'}  twin={'PASS' if twin_ok else 'FAIL'}")

    # acceptance: recall@k=2 must be 36/36 or misses shown (shown above)
    r2 = results.get("k2", {})
    if r2.get("recall") != r2.get("n"):
        print(f"\nNOTE: recall@k=2 is {r2.get('recall')}/{r2.get('n')} (misses listed above)")
    sys.exit(0 if mut_ok and twin_ok else 1)


if __name__ == "__main__":
    main()
