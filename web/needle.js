/* Needle client — talk to a Needle model running on an ESP32-P4.
 *
 * The board speaks token ids and nothing else. This module owns everything
 * around that: the SentencePiece BPE, the prompt format the model was trained
 * on, and a registry for actually executing the tool calls it emits.
 *
 * Quick start (served from the board, so the defaults just work):
 *
 *   import { Needle } from "./needle.js";
 *   const n = await Needle.connect();
 *
 *   const tools = [{
 *     name: "get_weather",
 *     description: "Get current weather for a location",
 *     parameters: { type: "object",
 *                   properties: { location: { type: "string" } },
 *                   required: ["location"] }
 *   }];
 *
 *   const r = await n.call("What's the weather in Oslo?", tools);
 *   r.call        // { name: "get_weather", arguments: { location: "Oslo" } }
 *   r.isToolCall  // false when the model answered with JSON that is not a call
 *   r.stats       // { encMs, tokMs, promptTokens, generatedTokens }
 *
 * Executing tools:
 *
 *   n.register("get_weather", async ({ location }) => fetchWeather(location));
 *   const out = await n.run(r.call);      // null if nothing is registered
 *
 * Tokenizing on its own — useful when preparing training data, since this is
 * the same code path the model sees at inference:
 *
 *   n.encode("hello world")        // [ids...]
 *   n.decode([4279, 743])          // "What is"
 *   n.prompt(query, tools).ids     // the full encoder input, <tools> and all
 */

import { NeedleBPE, buildEncoderInput, toSnakeCase } from "./bpe.js";

export class Needle {
  constructor(tok, base) {
    this.tok = tok;
    this.base = base;
    this.tools = new Map();
    this.maxEnc = 384;
  }

  /* base defaults to wherever this page came from, i.e. the board itself. */
  static async connect({ base = "", vocabUrl = null } = {}) {
    const url = vocabUrl || `${base}/vocab.txt`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`vocabulary fetch failed: ${res.status}`);
    return new Needle(new NeedleBPE(await res.text()), base);
  }

  // ---- tokenizer -------------------------------------------------------
  encode(text) { return this.tok.encode(text); }
  decode(ids) { return this.tok.decode(ids); }
  get vocabSize() { return this.tok.vocabSize; }

  /* Build the encoder input the model was trained on:
   * [query tokens] <tools> [tools-JSON tokens], truncated to maxEnc.
   * Tool names are snake_cased, which is what the model answers with. */
  prompt(query, tools = []) {
    const { ids, toolsJson } = buildEncoderInput(this.tok, query, tools, this.maxEnc);
    const names = {};
    if (Array.isArray(tools))
      for (const t of tools)
        if (t && typeof t === "object" && "name" in t) names[toSnakeCase(t.name)] = t.name;
    return { ids, toolsJson, names };
  }

  // ---- board -----------------------------------------------------------
  async status() {
    const r = await fetch(`${this.base}/api/status`);
    if (!r.ok) throw new Error(`status ${r.status}`);
    return r.json();
  }

  /* Token ids in, token ids out. The only call that touches the hardware. */
  async raw(ids) {
    const r = await fetch(`${this.base}/api/tokens`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || `board returned ${r.status}`);
    return d;
  }

  /* Question in, tool call out. */
  async call(query, tools = []) {
    if (!query || !query.trim()) throw new Error("empty query");
    const t0 = performance.now();
    const { ids, names } = this.prompt(query, tools);
    const tokenizeMs = performance.now() - t0;

    const t1 = performance.now();
    const d = await this.raw(ids);
    const boardMs = performance.now() - t1;

    const t2 = performance.now();
    const text = this.decode(d.gen_ids);
    let parsed = null, jsonOk = false;
    try { parsed = JSON.parse(text); jsonOk = true; } catch { /* model emitted non-JSON */ }
    const detokenizeMs = performance.now() - t2;

    const first = jsonOk && Array.isArray(parsed) ? parsed[0] : null;
    /* Well-formed JSON is not the same thing as a usable tool call: the model
     * can emit [] or a bare string and still parse. Branch on isToolCall
     * before touching .call; jsonOk only says the text parsed, and `valid` is
     * an alias of it. */
    const isToolCall = !!(first && typeof first === "object" && typeof first.name === "string");
    return {
      call: first,
      isToolCall,
      originalName: first && names[first.name] ? names[first.name] : (first && first.name),
      text, jsonOk, valid: jsonOk, parsed, ids, genIds: d.gen_ids,
      stats: {
        promptTokens: ids.length, generatedTokens: d.gen_ids.length,
        encMs: d.enc_ms, tokMs: d.tok_ms,
        tokenizeMs, boardMs, detokenizeMs,
      },
    };
  }

  // ---- tool runtime ----------------------------------------------------
  /* Keys are snake_cased on the way in and on every lookup: the model answers
   * in the snake form, but callers register with the name they wrote in the
   * tool catalogue. */
  register(name, fn) { this.tools.set(toSnakeCase(name), fn); return this; }
  has(name) { return typeof name === "string" && this.tools.has(toSnakeCase(name)); }

  /* Execute a call the model produced. Returns null when nothing is
   * registered for it, so a caller can tell "no runtime" from "ran and
   * returned nothing". */
  async run(call) {
    if (!call || typeof call.name !== "string") return null;
    const fn = this.tools.get(toSnakeCase(call.name));
    if (!fn) return null;
    return await fn(call.arguments || {}, call);
  }
}

export { NeedleBPE, buildEncoderInput, toSnakeCase };
