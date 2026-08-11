# demos

Ask the model something and watch it pick a tool. `ask.py` runs the same
scalar engine build the parity fixtures certify. No hardware needed; use
`--host` against a live board.

## No hardware

```
$ python3 demos/ask.py "What's the weather in Oslo?"
asking host engine with 6 tools, 253 tokens ...
  encoder 4180 ms, 19.2 ms/token
  model   [{"name":"get_weather","arguments":{"location":"Oslo"}}]
  result  Oslo, Oslo: 18C, Sunny, wind 14 km/h, humidity 43%

$ python3 demos/ask.py "Play a chime"
  model   [{"name":"play_tone","arguments":{"sound":"chime"}}]
  result  (chime!)

$ python3 demos/ask.py "Write me a poem about the sea"
  model   []
  no tool call in that answer
```

Needs `weights/needle_tools_g512.npk` (see `WEIGHTS.md`; override with
`--npk`). The engine binary builds itself on first run.

Bring your own tools. The model routes by reading the schema at inference
time:

```
$ python3 demos/ask.py --tools my_tools.json "email bob@example.com"
  model   [{"name":"send_email","arguments":{"to":"bob@example.com"}}]
  no runtime here for 'send_email' -- the call above is the output
```

(`my_tools.json` is a JSON list of tool specs: name, description, JSON-schema
parameters. `--tools get_weather,play_tone` offers a subset of the six built-ins.)

## Against a board

```
python3 demos/ask.py --host <board-ip> "what's the weather in Oslo?"
python3 demos/ask.py --host <board-ip> --list          # status and boot self-tests
```

`ask.py` is the whole integration in one file: build the prompt, POST token
ids, decode the reply, run the tool the model named. Works against an API-only
build (`CONFIG_NEEDLE_WEB_UI=n`); the board exposes two endpoints:

```
POST /api/tokens   {"ids": [...]}   ->  {"gen_ids": [...], "enc_ms", "tok_ms"}
GET  /api/status                    ->  uptime, memory, boot self-tests
```

Tool specs come from `train/make_dataset.py` (what the model was trained on);
the Python functions that execute them are the `RUNTIMES` dict at the top of
`ask.py`. Add yours there. Routing is most reliable for tools the model was
fine-tuned on. See `docs/finetuning.md`.

For the browser equivalent, `web/needle.js` is the same idea in JavaScript.
