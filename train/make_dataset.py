"""Build a fine-tuning set for the six demo tools.

Why this exists: measured on the board, the stock model routes these six tools
unreliably once more than two are offered. `look_up` lands on `set_timer`,
`board_status` produces no call at all, and `set_timer` sometimes answers
{"minutes": "3 minutes"} instead of an integer. None of that is surprising --
26M parameters, and these tools were never in its training set.

The fix is a fine-tune, and this generates the data for it in the schema
`needle finetune` reads:

    {"query": str, "tools": json-string, "answers": json-string}

Design notes, because they matter more than the code:

- **Tool subsets are sampled, not fixed.** Every example shows the model a
  different 2-4 tool menu including the right answer. Training on a single
  fixed menu teaches position, not selection.
- **Negatives are included.** Roughly one example in eight asks for something
  no offered tool can do, with `answers` of "[]". Without these the model
  learns that a call is always required, which is exactly the failure the demo
  shows on "write me a poem about the sea".
- **Argument types are enforced.** `minutes` is emitted as a JSON number, never
  a string, because that is the observed failure.
- **Phrasings are varied per tool**, including terse ones ("weather, Oslo") and
  indirect ones ("do I need a jacket in Oslo"), so the model keys on intent
  rather than on a template.

    python3 train/make_dataset.py --per-tool 200 --out train/tools.jsonl
"""

import argparse
import json
import random

TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"}},
                       "required": ["location"]},
    },
    "get_sun_times": {
        "name": "get_sun_times",
        "description": "Get sunrise and sunset times for a location",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"}},
                       "required": ["location"]},
    },
    "look_up": {
        "name": "look_up",
        "description": "Look up facts about a topic",
        "parameters": {"type": "object",
                       "properties": {"topic": {"type": "string"}},
                       "required": ["topic"]},
    },
    "set_timer": {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {"type": "object",
                       "properties": {"minutes": {"type": "integer"}},
                       "required": ["minutes"]},
    },
    "play_tone": {
        "name": "play_tone",
        "description": "Play a sound: chime, alarm, or success",
        "parameters": {"type": "object",
                       "properties": {"sound": {"type": "string"}},
                       "required": ["sound"]},
    },
    "board_status": {
        "name": "board_status",
        "description": "Report device uptime and memory",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

PLACES = [
    "San Francisco", "Oslo", "Reykjavik", "Tokyo", "Denver", "San Diego",
    "Lisbon", "Nairobi", "Reykjavík", "Auckland", "Chicago", "Berlin",
    "Cairo", "Hanoi", "Quito", "Perth", "Seattle", "Dublin", "Prague",
    "Bogotá", "Helsinki", "Marrakesh", "Vancouver", "Osaka", "Lima",
]
TOPICS = [
    "the Antikythera mechanism", "the Voyager program", "RISC-V", "tardigrades",
    "the Sargasso Sea", "Grace Hopper", "the Bronze Age collapse", "quicksort",
    "the Trans-Siberian Railway", "photosynthesis", "the Rosetta Stone",
    "neutron stars", "the Silk Road", "Ada Lovelace", "plate tectonics",
    "the Dead Sea", "penicillin", "Machu Picchu", "the Chicxulub crater",
    "lighthouses", "the Kola Superdeep Borehole", "cuttlefish",
]
SOUNDS = ["chime", "alarm", "success"]

WEATHER_Q = [
    "What's the weather in {p}?", "weather in {p}", "How's the weather in {p}?",
    "Is it raining in {p}?", "What's it like in {p} right now?",
    "Do I need a jacket in {p}?", "Current conditions in {p}",
    "How hot is it in {p}?", "Tell me the weather for {p}",
    "give me {p} weather", "Is it cold in {p}?", "what's the temperature in {p}",
]
SUN_Q = [
    "When does the sun set in {p}?", "sunset time in {p}",
    "What time is sunrise in {p}?", "How long is the day in {p}?",
    "When does it get dark in {p}?", "sunrise and sunset for {p}",
    "Is it still light in {p}?", "daylight hours in {p}",
    "when's golden hour in {p}", "What time does the sun come up in {p}?",
]
LOOK_Q = [
    "Tell me about {t}", "What is {t}?", "look up {t}", "Explain {t}",
    "I want to know about {t}", "give me facts on {t}", "What do you know about {t}?",
    "Who was {t}?", "background on {t}", "search for {t}",
]
TIMER_Q = [
    "Set a timer for {n} minutes", "timer {n} minutes", "remind me in {n} minutes",
    "start a {n} minute timer", "countdown {n} minutes", "wake me in {n} minutes",
    "give me {n} minutes on the clock", "set {n} minute timer please",
]
TONE_Q = [
    "Play a {s}", "play the {s} sound", "make a {s} noise", "sound the {s}",
    "give me a {s}", "can you play a {s}", "{s} please", "trigger the {s}",
]
BOARD_Q = [
    "How long have you been running?", "what's your uptime",
    "How much memory do you have left?", "status report", "system status",
    "Are you healthy?", "tell me about yourself", "how are you doing",
    "device status", "what's your IP address", "how much RAM is free",
    "diagnostics",
]
NEGATIVE_Q = [
    "Write me a poem about the sea", "What's the capital of France?",
    "Tell me a joke", "Who won the world cup in 1998?",
    "Translate hello into Japanese", "What's 17 times 23?",
    "Sing me a song", "Book me a flight to Paris",
    "What's your favourite colour?", "Summarise this article",
    "Send an email to my boss", "Order a pizza",
]


def line(query, tools, answers):
    return json.dumps({
        "query": query,
        "tools": json.dumps(tools, separators=(",", ":")),
        "answers": json.dumps(answers, separators=(",", ":")),
    })


def menu(rng, must_include, k=None):
    """A 2-4 tool menu that always contains `must_include`, shuffled."""
    others = [t for t in TOOLS if t != must_include]
    k = k or rng.randint(1, 3)
    chosen = [must_include] + rng.sample(others, min(k, len(others)))
    rng.shuffle(chosen)
    return [TOOLS[c] for c in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tool", type=int, default=200,
                    help="examples per tool (upstream wants >=120)")
    ap.add_argument("--negatives", type=int, default=None,
                    help="no-tool-fits examples (default: per-tool)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="train/tools.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []

    def emit(n, tool, make):
        for _ in range(n):
            q, a = make()
            out.append(line(q, menu(rng, tool), a))

    emit(args.per_tool, "get_weather", lambda: (
        rng.choice(WEATHER_Q).format(p=(p := rng.choice(PLACES))),
        [{"name": "get_weather", "arguments": {"location": p}}]))

    emit(args.per_tool, "get_sun_times", lambda: (
        rng.choice(SUN_Q).format(p=(p := rng.choice(PLACES))),
        [{"name": "get_sun_times", "arguments": {"location": p}}]))

    emit(args.per_tool, "look_up", lambda: (
        rng.choice(LOOK_Q).format(t=(t := rng.choice(TOPICS))),
        [{"name": "look_up", "arguments": {"topic": t}}]))

    # minutes must be a NUMBER; emitting "3 minutes" is the observed bug
    emit(args.per_tool, "set_timer", lambda: (
        rng.choice(TIMER_Q).format(n=(n := rng.choice([1, 2, 3, 5, 10, 15, 20, 25, 30, 45, 60]))),
        [{"name": "set_timer", "arguments": {"minutes": n}}]))

    emit(args.per_tool, "play_tone", lambda: (
        rng.choice(TONE_Q).format(s=(s_ := rng.choice(SOUNDS))),
        [{"name": "play_tone", "arguments": {"sound": s_}}]))

    emit(args.per_tool, "board_status", lambda: (
        rng.choice(BOARD_Q),
        [{"name": "board_status", "arguments": {}}]))

    n_neg = args.negatives if args.negatives is not None else args.per_tool
    for _ in range(n_neg):
        q = rng.choice(NEGATIVE_Q)
        tools = [TOOLS[t] for t in rng.sample(list(TOOLS), rng.randint(2, 4))]
        out.append(line(q, tools, []))

    rng.shuffle(out)
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")

    per = args.per_tool
    print(f"{args.out}: {len(out)} examples "
          f"({per} x 6 tools + {n_neg} negatives)")
    print("train with:")
    print(f"  cd $NEEDLE_HOME && python -m needle.cli finetune "
          f"{args.out} --epochs 3")


if __name__ == "__main__":
    main()
