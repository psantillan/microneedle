/* SentencePiece BPE for Needle, in the browser.
 *
 * A direct port of tokenizer/bpe.py. Everything except the matrix multiplies
 * runs here: the page tokenizes your query and tool catalogue, sends only
 * token ids to the board, and detokenizes the ids that come back. The server
 * is a serial cable with an HTTP connector on it.
 *
 * The merge loop must match the Python exactly or the board is handed a
 * different prompt than the oracle was scored on. Two details carry that:
 *   - ties in merge score break by LOWEST left index, which is what Python's
 *     heapq does with the (-score, i, j) tuple;
 *   - byte-fallback pieces (<0xNN>) never merge into a longer piece.
 * web/verify_bpe.mjs checks this file against the oracle's encoder ids for
 * all eight fixtures.
 */

const SPACE = "▁";

export class NeedleBPE {
  /* vocabText: one "<id>\t<piece>\t<score>" or "<piece>\t<score>" per line. */
  constructor(vocabText) {
    this.idToPiece = [];
    this.pieceToId = new Map();
    this.scores = [];

    for (const raw of vocabText.split("\n")) {
      const line = raw.replace(/\r$/, "");
      if (!line) continue;
      const parts = line.split("\t");
      let piece, score;
      if (parts.length >= 3) {
        const cut = line.indexOf("\t");
        const rest = line.slice(cut + 1);
        const last = rest.lastIndexOf("\t");
        piece = rest.slice(0, last);
        score = rest.slice(last + 1);
      } else {
        const last = line.lastIndexOf("\t");
        piece = line.slice(0, last);
        score = line.slice(last + 1);
      }
      this.pieceToId.set(piece, this.idToPiece.length);
      this.idToPiece.push(piece);
      this.scores.push(parseFloat(score));
    }

    this.vocabSize = this.idToPiece.length;
    this.padId = this.pieceToId.get("<pad>");
    this.eosId = this.pieceToId.get("</s>");
    this.bosId = this.pieceToId.get("<s>");
    this.unkId = this.pieceToId.get("<unk>");
    this.toolCallId = this.pieceToId.get("<tool_call>");
    this.toolsId = this.pieceToId.get("<tools>");

    this.byteIds = [];
    for (let b = 0; b < 256; b++) {
      this.byteIds.push(this.pieceToId.get(`<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`));
    }
    this.byteIdSet = new Set(this.byteIds);
    this.specialIds = new Set([this.padId, this.eosId, this.bosId,
                               this.unkId, this.toolCallId, this.toolsId]);
  }

  /* sp_remove_extra_whitespaces + add_dummy_prefix + escape_whitespaces */
  _preprocess(text) {
    const t = text.split(/\s+/).filter(Boolean).join(" ");
    if (!t) return "";
    return SPACE + t.replaceAll(" ", SPACE);
  }

  encode(text) {
    const s = this._preprocess(text);
    if (!s) return [];

    // one symbol per character; characters outside the vocab fall back to
    // their UTF-8 bytes as <0xNN> pieces
    const syms = [];
    const enc = new TextEncoder();
    for (const ch of s) {
      if (this.pieceToId.has(ch)) {
        syms.push(ch);
      } else {
        for (const b of enc.encode(ch)) {
          syms.push(`<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`);
        }
      }
    }

    const n = syms.length;
    const prev = new Int32Array(n);
    const next = new Int32Array(n);
    const alive = new Uint8Array(n).fill(1);
    for (let i = 0; i < n; i++) { prev[i] = i - 1; next[i] = i + 1; }
    next[n - 1] = -1;

    const pairScore = (i, j) => {
      const a = syms[i], b = syms[j];
      if (a === null || b === null) return null;
      if (a.startsWith("<0x") || b.startsWith("<0x")) return null;
      const merged = a + b;
      const pid = this.pieceToId.get(merged);
      if (pid === undefined) return null;
      return { score: this.scores[pid], merged };
    };

    // binary heap ordered by (-score, i, j) -- the same total order Python's
    // heapq imposes, so identical inputs produce identical merge sequences
    const heap = [];
    const less = (x, y) => (x[0] !== y[0] ? x[0] < y[0]
                          : x[1] !== y[1] ? x[1] < y[1]
                          : x[2] < y[2]);
    const push = (item) => {
      heap.push(item);
      let c = heap.length - 1;
      while (c > 0) {
        const p = (c - 1) >> 1;
        if (less(heap[c], heap[p])) { const t = heap[c]; heap[c] = heap[p]; heap[p] = t; c = p; }
        else break;
      }
    };
    const pop = () => {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length) {
        heap[0] = last;
        let p = 0;
        for (;;) {
          const l = 2 * p + 1, r = l + 1;
          let m = p;
          if (l < heap.length && less(heap[l], heap[m])) m = l;
          if (r < heap.length && less(heap[r], heap[m])) m = r;
          if (m === p) break;
          const t = heap[p]; heap[p] = heap[m]; heap[m] = t; p = m;
        }
      }
      return top;
    };

    for (let i = 0; i < n - 1; i++) {
      const r = pairScore(i, i + 1);
      if (r) push([-r.score, i, i + 1, r.merged]);
    }

    while (heap.length) {
      const [, i, j, merged] = pop();
      if (!alive[i] || !alive[j] || next[i] !== j) continue;   // stale
      if (syms[i] + syms[j] !== merged) continue;
      syms[i] = merged;
      alive[j] = 0;
      const k = next[j];
      next[i] = k;
      if (k !== -1) prev[k] = i;
      for (const [a, b] of [[prev[i], i], [i, next[i]]]) {
        if (a !== -1 && b !== -1 && alive[a] && alive[b]) {
          const r = pairScore(a, b);
          if (r) push([-r.score, a, b, r.merged]);
        }
      }
    }

    const out = [];
    for (let i = 0; i !== -1; i = next[i]) {
      const id = this.pieceToId.get(syms[i]);
      out.push(id === undefined ? this.unkId : id);
    }
    return out;
  }

  decode(ids) {
    const pieces = [];
    let buf = [];
    const flush = () => {
      if (buf.length) {
        pieces.push(new TextDecoder("utf-8").decode(new Uint8Array(buf)));
        buf = [];
      }
    };
    for (const i of ids) {
      const p = this.idToPiece[i];
      // a board that has drifted out of sync can emit an id past the vocabulary;
      // drop it rather than splicing "undefined" into the JSON the caller parses
      if (p === undefined) { flush(); continue; }
      if (this.byteIdSet.has(i)) { buf.push(parseInt(p.slice(3, 5), 16)); continue; }
      flush();
      if (this.specialIds.has(i)) continue;
      pieces.push(p);
    }
    flush();
    return pieces.join("").replaceAll(SPACE, " ").replace(/^ +/, "");
  }
}

/* Mirror of needle/dataset/tokenizer.py to_snake_case. Tool names are
 * snake_cased before tokenization, and the model answers in the snake form. */
export function toSnakeCase(name) {
  let s = name.replace(/[^a-zA-Z0-9_]+/g, "_");
  s = s.replace(/([a-z0-9])([A-Z])/g, "$1_$2");
  s = s.replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2");
  s = s.replace(/_+/g, "_");
  return s.toLowerCase().replace(/^_+|_+$/g, "");
}

/* [query tokens] <tools> [tools-JSON tokens], truncated exactly as
 * needle/model/run.py::_build_encoder_input does. */
export function buildEncoderInput(tok, query, toolsValue, maxEnc = 384) {
  const tools = Array.isArray(toolsValue)
    ? toolsValue.map(t => (t && typeof t === "object" && "name" in t)
        ? { ...t, name: toSnakeCase(t.name) } : t)
    : toolsValue;
  const toolsJson = JSON.stringify(tools);
  const q = tok.encode(query).slice(0, maxEnc - 2);
  const t = tok.encode(toolsJson).slice(0, maxEnc - q.length - 1);
  return { ids: q.concat([tok.toolsId], t), toolsJson };
}
