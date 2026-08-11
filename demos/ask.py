#!/usr/bin/env python3
"""Ask the model something and run the tool it picks.

Works two ways, same prompt, same tokenizer, same decoding:

  no hardware -- runs the host engine (the fixture-certified reference build):

      python3 demos/ask.py "what's the weather in Oslo?"
      python3 demos/ask.py --npk weights/mine.npk --tools my_tools.json "..."

  against a board -- speaks the two-endpoint API (works on an API-only build,
  CONFIG_NEEDLE_WEB_UI=n):

      python3 demos/ask.py --host <board-ip> "what's the weather in Oslo?"
      python3 demos/ask.py --host <board-ip> --list

The model speaks token ids. Everything else -- tokenizing, choosing what the
tools are, executing the call -- happens here. That split is the whole API:

    POST /api/tokens  {"ids": [...]}  ->  {"gen_ids": [...], "enc_ms", "tok_ms"}
    GET  /api/status                  ->  uptime, memory, boot self-tests
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tokenizer"))
sys.path.insert(0, os.path.join(REPO, "train"))
import bpe                        # noqa: E402
from make_dataset import TOOLS as SPECS   # noqa: E402  the six demo tool specs

MAX_ENC = 384
DEFAULT_NPK = os.path.join(REPO, "weights", "needle_tools_g512.npk")


def _get(url, timeout=20):
    # wttr.in and some others reject the default urllib agent
    req = urllib.request.Request(url, headers={"User-Agent": "microneedle/1.0"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _weather(a):
    place = a.get("location", "")
    if not place:
        return "the call arrived with no location"
    d = _get(f"https://wttr.in/{urllib.parse.quote(place)}?format=j1")
    c, ar = d["current_condition"][0], d["nearest_area"][0]
    where = ", ".join(x for x in (ar["areaName"][0]["value"], ar["region"][0]["value"]) if x)
    return (f"{where}: {c['temp_C']}C, {c['weatherDesc'][0]['value'].strip()}, "
            f"wind {c['windspeedKmph']} km/h, humidity {c['humidity']}%")


def _sun(a):
    place = a.get("location", "")
    if not place:
        return "the call arrived with no location"
    # wttr.in doubles as the geocoder; sunrisesunset.io does the arithmetic
    g = _get(f"https://wttr.in/{urllib.parse.quote(place)}?format=j1")["nearest_area"][0]
    j = _get(f"https://api.sunrisesunset.io/json?lat={g['latitude']}&lng={g['longitude']}")["results"]
    return f"{g['areaName'][0]['value']}: sunrise {j['sunrise']}, sunset {j['sunset']}"


def _wiki(a):
    topic = a.get("topic", "")
    if not topic:
        return "the call arrived with no topic"
    try:
        d = _get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                 + urllib.parse.quote(topic.replace(" ", "_")))
    except urllib.error.HTTPError:
        return f"nothing found for {topic!r}"
    return f"{d['title']}: {d['extract']}"


# What this file can actually DO when the model picks a tool. Specs come from
# train/make_dataset.py so the prompt matches what the model was trained on;
# a tool with no runtime still routes -- the call is printed, nothing runs.
RUNTIMES = {
    "get_weather":  _weather,
    "get_sun_times": _sun,
    "look_up":      _wiki,
    "set_timer":    lambda a: f"timer set for {a.get('minutes')} minutes",
    "play_tone":    lambda a: (print("\a", end="", flush=True) or
                               f"({a.get('sound', 'chime')}!)"),
    "board_status": None,     # filled in when --host is given; needs the board
}


def call_board(host, ids, timeout=240):
    req = urllib.request.Request(
        f"http://{host}/api/tokens",
        data=json.dumps({"ids": ids}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"board refused the request: {e.read().decode()[:200]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach {host}: {e.reason}")


def call_host(npk, ids):
    """Run the host engine (scalar reference build), building it if needed."""
    exe = os.path.join(REPO, "engine", "needle_ask")
    if not os.path.exists(exe):
        r = subprocess.run(["make", "-C", os.path.join(REPO, "engine"), "needle_ask"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"could not build engine/needle_ask:\n{r.stderr[-500:]}")
    if not os.path.exists(npk):
        raise SystemExit(f"no model at {npk}\n"
                         f"Weights are distributed via GitHub Releases -- see WEIGHTS.md "
                         f"(or pass --npk if yours lives elsewhere).")
    t0 = time.time()
    r = subprocess.run([exe, npk], input=" ".join(map(str, ids)),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"engine failed: {r.stderr.strip()[-300:]}")
    gen = [int(x) for x in r.stdout.split()]
    stats = dict(kv.split("=") for kv in r.stderr.split() if "=" in kv)
    return {"gen_ids": gen,
            "enc_ms": float(stats.get("enc_ms", 0)),
            "tok_ms": float(stats.get("tok_ms", 0)),
            "wall_s": time.time() - t0}


def load_tools(arg):
    """--tools is either a JSON file of tool specs or a comma subset of the six."""
    if arg and os.path.exists(arg):
        with open(arg) as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise SystemExit(f"{arg} must contain a JSON list of tool specs")
        return specs
    names = arg.split(",") if arg else list(SPECS)
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise SystemExit(f"unknown tools {unknown}; have {list(SPECS)} "
                         f"(or pass a JSON file of your own specs)")
    return [SPECS[n] for n in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="what to ask")
    ap.add_argument("--host", default=os.environ.get("NEEDLE_HOST"),
                    help="board address (or $NEEDLE_HOST); omit to run the host engine")
    ap.add_argument("--npk", default=DEFAULT_NPK,
                    help="model for the host engine (default: the demo fine-tune)")
    ap.add_argument("--tools", default=None,
                    help="comma subset of the six demo tools, or a JSON file of your own specs")
    ap.add_argument("--list", action="store_true", help="show the board's status and exit")
    ap.add_argument("--ids", action="store_true", help="also print the token ids")
    args = ap.parse_args()

    if args.host:
        RUNTIMES["board_status"] = lambda a: _board_status(args.host)

    if args.list:
        if not args.host:
            raise SystemExit("--list needs --host (it asks the board about itself)")
        s = _get(f"http://{args.host}/api/status", timeout=10)
        print(f"board {s.get('ip', args.host)}  up {s.get('uptime_s', '?')}s  "
              f"psram free {s.get('psram_free_kb', '?')} KB")
        for line in s.get("banner", []):
            print(" ", line)
        return

    if not args.query:
        ap.error("give it something to ask, or use --list")

    query = " ".join(args.query)
    specs = load_tools(args.tools)

    # Same builder as eval_tools / the web client: snake_case tool names and
    # the trained truncation order. Hand-rolling this drifts from the fixtures.
    tok = bpe.NeedleBPE()
    ids, _tools_json = bpe.build_encoder_input(tok, query, specs, MAX_ENC)

    where = args.host if args.host else "host engine"
    print(f"asking {where} with {len(specs)} tools, {len(ids)} tokens ...")
    d = call_board(args.host, ids) if args.host else call_host(args.npk, ids)
    text = tok.decode(d["gen_ids"])

    print(f"  encoder {d['enc_ms']:.0f} ms, {d['tok_ms']:.1f} ms/token")
    print(f"  model   {text}")
    if args.ids:
        print(f"  in      {ids}")
        print(f"  out     {d['gen_ids']}")

    try:
        call = json.loads(text)[0]
    except (json.JSONDecodeError, IndexError, KeyError):
        print("  no tool call in that answer")
        return

    fn = RUNTIMES.get(call.get("name"))
    if not fn:
        print(f"  no runtime here for {call.get('name')!r} -- the call above is the output")
        return
    print(f"  result  {fn(call.get('arguments') or {})}")


def _board_status(host):
    s = _get(f"http://{host}/api/status", timeout=10)
    up = int(s.get("uptime_s", 0))
    return (f"up {up // 3600}h{up % 3600 // 60:02d}m, "
            f"{s.get('psram_free_kb')} KB PSRAM free, "
            f"{s.get('cpu_mhz')} MHz, at {s.get('ip')}")


if __name__ == "__main__":
    main()
