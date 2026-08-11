# RESULTS: grammar-forced decode steps skip the logits projection

Branch: `grammar-bake`

## What changed

On rigid tool-call JSON positions the next token is fully determined by the
surface grammar. The decoder layers still run (KV cache and residual stream
byte-identical), but the final rmsnorm + 8192×512 logits GEMV + vocab argmax
are skipped and the forced token is returned.

### Engine (`engine/needle_engine.c`, `engine/needle_engine.h`)

- `ne_step_f(ctx, token, forced)` — same as `ne_step` when `forced < 0`; when
  `forced >= 0`, layers run and `forced` is returned without the logits stage.
- `ne_step` remains the public entry point; it calls `ne_step_f(..., -1)`.
- Static `ne_forced_next(prev, n_out, n_265)` implements a conservative DFA.
- `ne_generate_cb` applies the DFA when `ne_grammar_force != 0` (default 1).
- `ne_grammar_skips` counts skipped logits steps (for measurement).

### DFA (three forces, not four)

| Previous token | Condition | Forced next |
|---|---|---|
| 356 `▁[{"` | always | 294 `name` |
| 294 `name` | only when it sits at fixed index 2 (`n_out == 3`) | 264 `":"` |
| 265 `","` | only the first 265 in the sequence | 393 `arguments` |

**Why not 393 → 282:** empty-argument tools emit `630` (`":{}}]'`) after
`arguments`, not `282` (`":{"`). Forcing 282 fails `single_tool_no_args` in
the parity set (gate 1). The brief listed that edge; the fixtures overrule it.
Three forced steps per tool call (~18 skips across the 6 tool cases of the
eval set; the poem case emits no call and skips nothing).

### Host bench

`engine/bench_grammar.c` (+ `make -C engine bench_grammar`) times N reps of
the eval_tools vector set with `ne_grammar_force` 0 then 1.

Firmware callers that use raw `ne_step` are unchanged; they can opt into
`ne_step_f` later. No firmware sources were modified.

---

## Gate 1 — parity (8/8 byte-identical)

```
$ make -C engine needle_host && ./engine/needle_host weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt
weather_sf             PASS  enc=48   ref=20  got=20      920 ms
single_tool_no_args    PASS  enc=22   ref=10  got=10      419 ms
two_tools_pick_second  PASS  enc=87   ref=18  got=18     1360 ms
numeric_arg            PASS  enc=42   ref=13  got=13      702 ms
long_tools_context     PASS  enc=271  ref=22  got=22     4275 ms
no_matching_tool       PASS  enc=45   ref=15  got=15      801 ms
ups_status             PASS  enc=63   ref=19  got=19     1106 ms
unicode_and_punct      PASS  enc=70   ref=30  got=30     1394 ms

8/8 cases match the JAX oracle token-for-token
```

---

## Gate 2 — eval_tools (identical to pre-change baseline)

**Baseline (before change):** 7/7 correct.

**After change:**

```
$ python3 tools/eval_tools.py weights/needle_tools_g512.npk
  OK  What's the weather in Oslo?          -> get_weather    {"location": "Oslo"}
  OK  When does the sun set in Reykjavik?  -> get_sun_times  {"location": "Reykjavik"}
  OK  Tell me about tardigrades            -> look_up        {"topic": "tardigrades"}
  OK  Set a timer for 5 minutes            -> set_timer      {"minutes": 5}
  OK  Play a chime                         -> play_tone      {"sound": "chime"}
  OK  How long have you been running?      -> board_status   
  OK  Write me a poem about the sea        -> None           

  7/7 correct with ALL SIX tools offered (250 token prompt)
```

(Brief said 6/7; this weight/tool set scores 7/7 both before and after.)

---

## Gate 3 — mutation test (must be able to fail)

Temporarily changed `ne_forced_next` so `356 → 999` instead of `294`, rebuilt,
re-ran gate 1:

```
=== MUTATED RUN (356 -> 999) ===
weather_sf             FAIL  ... first divergence at index 2 (ref=294 got=999)
single_tool_no_args    FAIL  ... first divergence at index 2 (ref=294 got=999)
two_tools_pick_second  FAIL  ... first divergence at index 2 (ref=294 got=999)
numeric_arg            FAIL  ... first divergence at index 2 (ref=294 got=999)
long_tools_context     FAIL  ... first divergence at index 2 (ref=294 got=999)
no_matching_tool       FAIL  ... first divergence at index 2 (ref=294 got=999)
ups_status             FAIL  ... first divergence at index 2 (ref=294 got=999)
unicode_and_punct      FAIL  ... first divergence at index 2 (ref=294 got=999)

0/8 cases match the JAX oracle token-for-token
mutated_exit=1
```

Reverted the one-line mutation, rebuilt, gate 1 returned to **8/8 PASS**.
The DFA path is live and gate 1 catches a wrong force.

Also confirmed earlier: forcing `393 → 282` (as the brief suggested) yields
`single_tool_no_args` FAIL (`ref=630 got=282`) — that is why the shipped DFA
omits that edge.

---

## Gate 4 — pie-check

```
$ make -C engine pie-check
cc -O2 -std=c99 -Wall -Wextra -DNE_PIE -c needle_engine.c -o /dev/null
```

Clean (no warnings). `make -C engine needle_trace` also builds clean.

---

## Gate 5 — measurement (100× the 7 eval generations)

Host CPU time via `engine/bench_grammar` on `weights/needle_tools_g512.npk`
with the same 7 encoder inputs as `eval_tools.py`. Mode 0 = no skip,
mode 1 = grammar force on.

```
$ ./engine/bench_grammar weights/needle_tools_g512.npk /tmp/ne_tools_vec.txt 100
grammar_force=0  cases=7  reps=100  tokens=9300  skips=0     cpu_s=2598.412
grammar_force=1  cases=7  reps=100  tokens=9300  skips=1800  cpu_s=2605.341
```

| Mode | grammar_force | tokens | steps skipped | CPU time (s) |
|---|---|---|---|---|
| without skip | 0 | 9300 | 0 | 2598.412 |
| with skip | 1 | 9300 | 1800 | 2605.341 |
| **delta** | | 0 | **+1800 skips** | **+6.9 s (~0.3%, noise)** |

Per single pass of the 7 cases: **93 tokens**, **18 forced skips**
(3 per tool call × 6 tools; poem yields no call → 0 skips). Over 100 reps:
9300 tokens, 1800 skips (~19% of generated tokens).

Host end-to-end CPU is encode-dominated (250-token tool prompt), so the
logits skip is lost in run-to-run noise at this scale. A shorter 2-rep
probe on the same harness showed a clearer decode-side signal:

```
grammar_force=0  cases=7  reps=2  tokens=186  skips=0   cpu_s=55.714
grammar_force=1  cases=7  reps=2  tokens=186  skips=36  cpu_s=51.047
```

(~8% CPU reduction on that shorter sample). Board decode-side gain should
be larger as a fraction of per-token time: the 8192-row logits GEMV is the
largest per-step matrix in decode (~40 ms on device for that GEMV alone).

---

# RESULTS: warm-schema prefill experiment

Host-only experiment on branch `schema-cache`. Exact engine path untouched;
`NE_TRACE` and `NE_WARMSCHEMA` are opt-in Makefile targets.

Weights: `weights/needle_tools_g512.npk` (fine-tune; routing battery).
Parity weights: `weights/needle_v3_g512.npk` (base; 8/8 fixtures).

---

## Acceptance gates

| Gate | Result |
|------|--------|
| `make -C engine needle_host && ./engine/needle_host weights/needle_v3_g512.npk weights/vectors_int4p_int8e.txt` | **8/8** |
| `make -C engine pie-check` | clean |
| Exact `eval_tools.py` (no warm) | **7/7** |

---

## Phase 1 — schema drift vs depth

Script: `tools/measure_schema_drift.py`  
Driver: `make -C engine needle_trace` (new `encx` records: full per-position encoder state).  
Battery: 29 standard-schema queries (probe set without causal schema rewrites).  
Schema length: 243 tokens after `<tools>` (id 5).

### Cross-query cosine of bucket-mean schema states (mean / min over pairs)

Depth = offset past the `<tools>` marker (0 = marker itself).

| layer | 0–15 | 16–63 | 64–127 | 128+ | marker alone |
|------:|-----:|------:|-------:|-----:|-------------:|
| 0 (post-embed) | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |
| 1 | 0.9989/0.9978 | 0.9999/0.9999 | 1.0000/0.9999 | 0.9999/0.9999 | 0.8986/0.7945 |
| 2 | 0.9988/0.9977 | 0.9999/0.9999 | 1.0000/0.9999 | 0.9999/0.9999 | 0.9225/0.8299 |
| 3 | 0.9989/0.9979 | 0.9998/0.9996 | 0.9997/0.9994 | 0.9998/0.9995 | 0.9436/0.8767 |
| 4 | 0.9986/0.9971 | 0.9996/0.9989 | 0.9996/0.9990 | 0.9995/0.9987 | 0.9379/0.8740 |
| 5 | 0.9984/0.9959 | 0.9995/0.9985 | 0.9996/0.9990 | 0.9994/0.9986 | 0.9435/0.8838 |
| 6 | 0.9984/0.9962 | 0.9995/0.9986 | 0.9996/0.9990 | 0.9995/0.9987 | 0.9570/0.9075 |
| 7 | 0.9956/0.9852 | 0.9982/0.9936 | 0.9987/0.9962 | 0.9989/0.9974 | 0.9665/0.9216 |
| 8 | 0.9932/0.9725 | 0.9973/0.9905 | 0.9982/0.9944 | 0.9984/0.9959 | 0.9720/0.9291 |
| 9 | 0.9809/0.8971 | 0.9888/0.9538 | 0.9948/0.9793 | 0.9952/0.9808 | 0.9643/0.9090 |
| 10 | 0.9657/0.8395 | 0.9668/0.8823 | 0.9857/0.9405 | 0.9881/0.9465 | 0.9598/0.9038 |
| 11 | 0.9747/0.8811 | 0.9724/0.9107 | 0.9862/0.9471 | 0.9888/0.9598 | 0.9740/0.9265 |
| 12 | 0.9650/0.8355 | 0.9600/0.8703 | 0.9791/0.9107 | 0.9829/0.9379 | 0.9600/0.8971 |
| **13 (final norm)** | **0.9206/0.6540** | **0.9091/0.7225** | **0.9506/0.7989** | **0.9521/0.8442** | **0.9534/0.8814** |

### Phase 1 reading

- **Early layers (1–6):** deep schema (16+) is essentially constant across queries
  (mean cosine ≥ 0.999). The marker is the noisy edge case (mean ~0.90–0.96,
  min down to 0.79), matching the prior 0.90–0.97 measurement.
- **Late layers (9–13):** query bleed reaches deep into the JSON. At the final
  norm, even the 128+ bucket is only ~0.95 mean / 0.84 min — **not** 0.999.
  Freezing final-layer schema state is therefore freezing a query-dependent
  representation, not a constant one.
- Per-position cosines at layer 13 oscillate (some depths ~0.87 min 0.44, some
  ~0.999). There is no long tail of near-identical deep tokens at the output.

Reproduce:

```bash
make -C engine needle_trace
python3 tools/measure_schema_drift.py weights/needle_tools_g512.npk
```

---

## Phase 2 — warm-schema prototype

Build: `make -C engine needle_warmschema` (`-DNE_WARMSCHEMA`).  
Script: `tools/eval_warmschema.py`.  
Reference capture: first case, `"Play a chime"` (same standard schema).  
Approximation: recompute only query rows through the encoder; schema residual
and per-layer K/V stay at the reference cache. Query rows attend over
`[fresh query K/V | cached schema K/V]`. Final norm on query only; schema
uses cached post-final-norm rows. Cross-K/V + decode unchanged.

### Accuracy

| Set | tool-name exact | tool-name warm | byte-identical warm vs exact |
|-----|----------------:|---------------:|-----------------------------:|
| 29-case standard-schema battery | 28/29 | **9/29** | **7/29** |
| 7-case `eval_tools` | 7/7 | **2/7** | **2/7** |
| Combined | 35/36 | **11/36** | **9/36** |

Warm tool-name hits are almost entirely: the reference query itself, other
`play_tone` phrasings that also want `sound=chime`/`success`, the no-call poem,
and one case where exact already mis-routes to `play_tone` (`brd03`).

### FLOP / row-count reduction (per prefill)

| | mean nq (recomputed) | ns (frozen) | ntot | recompute fraction |
|--|---------------------:|------------:|-----:|-------------------:|
| 29-case battery | 6.7 | 243 | 249.7 | **2.7%** |
| 7-case eval_tools | 8.4 | 243 | 251.4 | **3.4%** |

Encoder row-count reduction ≈ **30–38×** on these prompts (query-only
recompute). That is the cheap part; the accuracy cost is not.

### Divergent cases (verbatim pairs)

Warm collapses almost every non-reference route to the reference call
`play_tone(sound=chime)`. Selected pairs:

**Argument stuck on reference (tool name still correct):**

```
[aud01] play the alarm sound
  exact: [{"name":"play_tone","arguments":{"sound":"alarm"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]

[aud03] sound the alarm
  exact: [{"name":"play_tone","arguments":{"sound":"alarm"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]

[aud06] Play an alarm
  exact: [{"name":"play_tone","arguments":{"sound":"alarm"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]
```

**Routing collapse to reference tool:**

```
[wea00] What's the weather in Oslo?
  exact: [{"name":"get_weather","arguments":{"location":"Oslo"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]

[sun00] When does the sun set in Reykjavik?
  exact: [{"name":"get_sun_times","arguments":{"location":"Reykjavik"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]

[lok00] Tell me about tardigrades
  exact: [{"name":"look_up","arguments":{"topic":"tardigrades"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]

[tim01] start a 20 minute timer
  exact: [{"name":"set_timer","arguments":{"minutes":20}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]

[brd00] How long have you been running?
  exact: [{"name":"board_status","arguments":{}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"chime"}}]
```

**Degenerate empty / truncated:**

```
[tim00] Set a timer for 5 minutes
  exact: [{"name":"set_timer","arguments":{"minutes":5}}]
  warm:  []

[lok01] look up the Antikythera mechanism
  exact: [{"name":"look_up","arguments":{"topic":"the Antikythera mechanism"}}]
  warm:  [{"name":"play_tone","arguments":{"sound":"the Antiky"}}]
```

Full dump: re-run `python3 tools/eval_warmschema.py weights/needle_tools_g512.npk`.

Reproduce:

```bash
make -C engine needle_warmschema
python3 tools/eval_warmschema.py weights/needle_tools_g512.npk
```

---

## Verdict

**Do not put this warm-schema approximation on the board.** Phase 1 shows why
it looked tempting: through mid-encoder, deep schema states are near-constant
across queries while the marker is not. Phase 2 shows why that is not enough:
by the final norm the query has bled through the whole schema (mean cosine
~0.91–0.95, not 0.999), and freezing reference schema K/V + residual destroys
tool routing — warm accuracy falls from 28/29 to 9/29 on the standard battery
and from 7/7 to 2/7 on `eval_tools`, with non-reference tools collapsing to
the capture call `play_tone(sound=chime)`. The ~35× encoder row reduction is
real and cheap to prototype, but the model uses bidirectional query→schema
mixing for routing; an exact or near-exact schema cache is not available at
the layers the decoder reads. A board implementation of this freeze is not
worth it. Any follow-up would need a different approximation (e.g. partial
schema recompute, layer-wise hybrid, or a deliberately query-independent
schema encoder), not this prototype as-is.
