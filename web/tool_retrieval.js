/* Client-side tool retrieval — prune the tools schema before buildEncoderInput.
 *
 * Direct port of tools/tool_retrieval.py. Same changed-together contract as
 * tokenizer/bpe.py ↔ web/bpe.js: the measurement harness asserts identical
 * top-k sets on every battery query via node (tools/tool_retrieval.py
 * --twin-check). Keep the algorithms and the TOOL_KEYWORDS table in lockstep.
 *
 * Lexical features only (no model, no embeddings):
 *   score = weighted overlap of query tokens vs tool name + description +
 *           per-tool keyword table (data, not code — new tools carry keywords)
 *   select top-k; always include tools that clear SCORE_FLOOR; when every
 *   score is near-zero fall back to the full catalogue so OOD queries like
 *   the poem never silently become retrieval misses.
 */

export const TOOL_KEYWORDS = {
  get_weather: [
    "weather", "raining", "rain", "hot", "cold", "temperature", "temp",
    "jacket", "conditions", "forecast", "humid", "wind", "climate",
  ],
  get_sun_times: [
    "sunset", "sunrise", "golden", "daylight", "dusk", "dawn", "sun",
    "daylight", "dark", "light", "hour",
  ],
  look_up: [
    "look", "lookup", "facts", "explain", "search", "tell", "who",
    "background", "know", "topic", "info", "information",
  ],
  set_timer: [
    "timer", "countdown", "minutes", "minute", "remind", "wake",
    "clock", "alarm",
  ],
  play_tone: [
    "sound", "noise", "play", "chime", "alarm", "success", "tone",
    "beep", "jingle", "ring",
  ],
  board_status: [
    "board", "uptime", "memory", "running", "chip", "device", "status",
    "ram", "diagnostics", "healthy", "ip", "psram", "free", "yourself",
    "system", "cpu", "esp",
  ],
};

export const STOPWORDS = new Set([
  "a", "an", "the", "me", "you", "i", "we", "my", "your", "it", "its",
  "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
  "in", "on", "at", "for", "of", "to", "and", "or", "but", "with",
  "from", "by", "as", "if", "so", "than", "that", "this", "these",
  "those", "there", "here", "when", "how", "where", "which", "whom",
  "can", "could", "would", "should", "will", "just", "please", "some",
  "any", "into", "out", "up", "down", "over", "under", "about",
  "what",
]);

export const SCORE_FLOOR = 2.0;
export const NEAR_ZERO = 2.0;
export const NAME_W = 3.0;
export const KEYWORD_W = 2.0;
export const DESC_W = 1.0;
export const BIGRAM_W = 2.5;

const WORD_RE = /[a-z0-9]+/g;

export function tokenize(text) {
  const m = String(text).toLowerCase().match(WORD_RE);
  return m || [];
}

export function stem(w) {
  const sufs = ["ings", "ing", "tion", "ness", "ment", "ies", "ied",
                "ers", "est", "ed", "es", "ly", "er", "s"];
  for (const suf of sufs) {
    if (w.length > suf.length + 2 && w.endsWith(suf)) return w.slice(0, -suf.length);
  }
  return w;
}

function bag(words) {
  const b = new Set();
  for (const w of words) {
    b.add(w);
    b.add(stem(w));
  }
  return b;
}

/**
 * Lexical score of one tool against the query. Pure function of strings.
 * keywords: optional override table (mutation tests); defaults to TOOL_KEYWORDS.
 */
export function scoreTool(query, tool, keywords) {
  /* Description is scored only up to a colon — example values after it
   * ("chime, alarm, or success") live in TOOL_KEYWORDS so the synonym
   * table is the real lever. Keep in lockstep with tools/tool_retrieval.py. */
  const kwTable = keywords || TOOL_KEYWORDS;
  const name = String(tool.name || "").replace(/_/g, " ");
  const descFull = String(tool.description || "");
  const desc = descFull.split(":")[0];
  const kws = kwTable[tool.name] || [];

  const nameBag = bag(tokenize(name));
  const descBag = bag(tokenize(desc));
  const kwBag = bag(kws);

  const q = tokenize(query);
  let score = 0;
  const hit = new Set();

  for (const w of q) {
    if (hit.has(w)) continue;
    const sw = stem(w);
    if (kwBag.has(w) || kwBag.has(sw)) {
      score += KEYWORD_W;
      hit.add(w);
    } else if (!STOPWORDS.has(w) && (nameBag.has(w) || nameBag.has(sw))) {
      score += NAME_W;
      hit.add(w);
    } else if (!STOPWORDS.has(w) && (descBag.has(w) || descBag.has(sw))) {
      score += DESC_W;
      hit.add(w);
    }
  }

  for (let i = 0; i < q.length - 1; i++) {
    const bi = q[i] + q[i + 1];
    const sbi = stem(q[i]) + stem(q[i + 1]);
    if (kwBag.has(bi) || kwBag.has(sbi) || nameBag.has(bi) || descBag.has(bi)) {
      score += BIGRAM_W;
      break;
    }
  }
  return score;
}

/** @returns {[number, object][]} sorted high→low, then name */
export function scoreAll(query, tools, keywords) {
  const scored = tools.map(t => [scoreTool(query, t, keywords), t]);
  scored.sort((a, b) => (b[0] - a[0]) || String(a[1].name).localeCompare(String(b[1].name)));
  return scored;
}

/**
 * Top-k by score, union tools clearing floor; full list if all near-zero.
 */
export function selectTools(query, tools, k = 2, floor = SCORE_FLOOR,
                            nearZero = NEAR_ZERO, keywords) {
  if (!tools || !tools.length) return [];
  const scored = scoreAll(query, tools, keywords);
  const maxS = scored[0][0];
  if (maxS < nearZero) return tools.slice();

  const selected = [];
  const seen = new Set();
  for (const [s, t] of scored) {
    const name = t.name || "";
    if (seen.has(name)) continue;
    if (selected.length < k || s >= floor) {
      selected.push(t);
      seen.add(name);
    }
  }
  return selected;
}
