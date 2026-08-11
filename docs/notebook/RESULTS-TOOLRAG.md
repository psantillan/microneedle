# Results: client-side tool retrieval

Prompt-side pruning of the tools JSON before `bpe.build_encoder_input` /
`buildEncoderInput`. No engine changes, tokenizer untouched.

## What shipped

| Path | Role |
|------|------|
| `tools/tool_retrieval.py` | Reference scorer + measurement harness (battery, mutation, twin, engine) |
| `web/tool_retrieval.js` | JS twin of the scorer (changed-together with the Python file) |
| `web/index.html` | Visible **tool retrieval** toggle (default **ON**, k=2) + live “tools sent” readout |

**Scorer (lexical only):** weighted overlap of query tokens against tool name
pieces, description head (text before a colon), and a per-tool **keyword
table** (data, not code). Bigrams catch `sun set` → `sunset` and `look up`.

**Selection:** top-k by score; always include any tool that clears
`SCORE_FLOOR=2`; if every score is `< NEAR_ZERO=2` (OOD, e.g. the poem), fall
back to the full six-tool catalogue so retrieval never silently turns a
routable query into a miss.

## Battery

36 cases = 29 standard-schema cases from `tools/probe_latent_audio.py`
(8 audio + 4×5 other + poem) + 7 from `tools/eval_tools.py`.

Engine: `engine/needle_trace` (built with `make -C engine needle_trace`) on
`weights/needle_tools_g512.npk`, one vector file per config with the **pruned**
schema encoded exactly as `bpe.build_encoder_input` does.

### k-sweep

| cfg | retrieval recall | end-task name accuracy | mean encoder tokens | vs full-6 |
|-----|------------------|------------------------|---------------------|-----------|
| full6 | **36/36** | **35/36** | 250.0 | 1.000 |
| k=1 | **36/36** | **36/36** | 76.9 | **0.308** |
| k=2 | **36/36** | **36/36** | 99.0 | **0.396** |
| k=3 | **36/36** | **36/36** | 135.4 | **0.542** |

Prefill scales with encoder length. At k=2 the mean prompt is ~40% of the
full-schema length (~2.5× fewer tokens). On the board, full-schema prefill is
~5 s for this catalogue; a ~0.4× token ratio is the expected prefill saving
(wall clock is roughly linear in token count for this encoder).

### Divergences (verbatim)

**full6 only miss** — already known, not tuned around:

```
brd03  want=board_status  got=play_tone
  q='What chip are you running on?'
  raw=[{"name":"play_tone","arguments":{"sound":"chime"}}]
  tools=all six
```

**Pruning fixes that misroute** (distractor-removal):

```
[FIX] k1/k2/k3  brd03
  full=play_tone  pruned=board_status  want=board_status
  q='What chip are you running on?'
```

With k∈{1,2,3} the selected set is `{board_status, …}` and no longer offers
`play_tone` as a distractor; the model routes correctly. No other answer
changes vs full6 on the 36-case battery.

All k=1/2/3 configs: **no name misses, no recall misses.**

### Recall by class (presence of correct tool in pruned schema)

| class | n | k=1 | k=2 | k=3 | full6 |
|-------|---|-----|-----|-----|-------|
| aud (play_tone) | 8 | 8/8 | 8/8 | 8/8 | 8/8 |
| wea | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| sun | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| lok | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| tim | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| brd | 4 | 4/4 | 4/4 | 4/4 | 4/4 |
| poem (OOD → full list) | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| eval_* (7) | 7 | 7/7 | 7/7 | 7/7 | 7/7 |
| **total** | **36** | **36/36** | **36/36** | **36/36** | **36/36** |

Note: `lok02` (“What is a quasar?”) scores near-zero and takes the OOD full-list
fallback; that still counts as recall success and keeps end-task accuracy.

## Mutation test (synonym table can fail)

Break `play_tone` sound keywords (`sound/noise/play/chime/alarm/success/…`),
re-score audio cases with **signal-recall** (want in top-k with a real score;
OOD fallback does **not** count — otherwise full-list fallback would hide the
break):

| keywords | aud signal-recall@2 |
|----------|---------------------|
| intact | **9/9** |
| mutated (sound keywords stripped) | **6/9** |

Misses after mutation (all fell back to the full catalogue for lack of signal):

- `aud02` “make a success noise”
- `aud05` “make a chime noise”
- `aud07` “sound the chime”

Restored the table after the run. Gate: `python3 tools/tool_retrieval.py --mutate-synonyms`.

## JS ↔ Python twin parity

Method: harness writes `web/_twin_check_toolrag.mjs` next to the JS module,
runs `node` for k∈{1,2,3}, compares selected name lists to
`select_tools()` in Python on all 36 queries, deletes the temp script.

```
k=1: OK same top-k sets on all 36 queries
k=2: OK same top-k sets on all 36 queries
k=3: OK same top-k sets on all 36 queries
```

Gate: `python3 tools/tool_retrieval.py --twin-check`.

## Acceptance gates

| Gate | Status |
|------|--------|
| No engine diff (`git diff engine/` empty); tokenizer untouched | **PASS** |
| Battery table; recall@k=2 is 36/36 | **PASS** |
| Mutation drops aud signal-recall (9→6), then restore | **PASS** |
| JS and Python scorers agree on all 36 queries | **PASS** |

## Verdict

- **Recommended k: 2.** Recall and name accuracy are perfect at k=1 already on
  this battery, but k=2 keeps a second candidate when scores are close
  (e.g. `sun set` vs `set_timer`, alarm/timer overlap) and still cuts mean
  encoder tokens to **~40%** of full-6 (99 vs 250).
- **Demo default: ON** at k=2. The battery says so: better end-task accuracy
  than full-6 (fixes `brd03`) and a large prefill saving. Toggle remains
  visible; k is selectable 1/2/3; the UI shows which tools were sent (or
  “OOD fallback · sent all 6”).

### Reproduce

```bash
make -C engine needle_trace
python3 tools/tool_retrieval.py weights/needle_tools_g512.npk
# faster gates:
python3 tools/tool_retrieval.py --score-only
python3 tools/tool_retrieval.py --mutate-synonyms
python3 tools/tool_retrieval.py --twin-check
```
