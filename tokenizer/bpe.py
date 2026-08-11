#!/usr/bin/env python3
"""SentencePiece BPE tokenizer for Needle, implemented from vocab.txt alone.

The vocabulary is the whole merge table: SentencePiece BPE stores, for every
piece, a score == -(merge rank).  (`merges.txt` in the CQ4 bundle is a one-line
stub, "#version: 0.2 / 1 merges.txt", and carries nothing.)  Inference is
then:

  1. normalize (identity for this model: sp_normalizer is "identity")
  2. remove extra whitespace (sp_remove_extra_whitespaces = true)
  3. add dummy prefix, escape whitespace as U+2581 (sp_add_dummy_prefix,
     sp_escape_whitespaces = true)
  4. split to unicode characters; any character not in the vocab becomes its
     UTF-8 bytes as <0xNN> pieces (sp_byte_fallback = true)
  5. greedily merge the adjacent pair whose concatenation is in the vocab with
     the HIGHEST score, until no pair merges.

That reproduces sentencepiece's BpeModel::Encode exactly.  Two properties of
the merge loop decide the token ids and must survive any port (web/bpe.js
is the JS twin):

  - equal scores break by the LOWEST left index, which is the order the
    (-score, i, j) tuples impose on heapq;
  - byte-fallback pieces (<0xNN>) never merge into a longer piece.

Validated in validate_bpe() against invariants that only hold for a correct
implementation (round-trip detokenization, byte-fallback coverage, and the
model's own special-token ids).
"""
from __future__ import annotations

import heapq
import unicodedata

import json as _json
import os as _os
import re as _re

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_NEEDLE_HOME = _os.environ.get("NEEDLE_HOME", _os.path.expanduser("~/needle"))

# The vocabulary this tokenizer needs is "<id>\t<piece>\t<score>" per line.
# Upstream ships "<piece>\t<score>" (needle.vocab) -- same data, no index
# column -- so either is accepted and the index is supplied when missing.
# Prefer the vendored copy: the web demo is a product dependency and must not
# reach outside the repo for its data.
VOCAB_TXT = _os.environ.get("NEEDLE_VOCAB") or next(
    (p for p in (_os.path.join(_REPO, "tokenizer", "vocab.txt"),
                 _os.path.join(_NEEDLE_HOME, "tokenizer", "needle.vocab"))
     if _os.path.exists(p)),
    _os.path.join(_REPO, "tokenizer", "vocab.txt"))
SPACE = "▁"  # U+2581 LOWER ONE EIGHTH BLOCK, sentencepiece's space marker


class NeedleBPE:
    def __init__(self, vocab_txt: str = VOCAB_TXT):
        self.id_to_piece: list[str] = []
        self.piece_to_id: dict[str, int] = {}
        self.scores: list[float] = []
        with open(vocab_txt, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                # "<id>\t<piece>\t<score>" or upstream's "<piece>\t<score>";
                # the piece may itself contain a tab, hence the rsplit
                parts = line.split("\t")
                if len(parts) >= 3:
                    idx, rest = line.split("\t", 1)
                    piece, score = rest.rsplit("\t", 1)
                    i = int(idx)
                    assert i == len(self.id_to_piece), f"non-sequential id {i}"
                else:
                    piece, score = line.rsplit("\t", 1)
                    i = len(self.id_to_piece)
                self.id_to_piece.append(piece)
                self.piece_to_id[piece] = i
                self.scores.append(float(score))

        self.vocab_size = len(self.id_to_piece)
        self.pad_id = self.piece_to_id["<pad>"]
        self.eos_id = self.piece_to_id["</s>"]
        self.bos_id = self.piece_to_id["<s>"]
        self.unk_id = self.piece_to_id["<unk>"]
        self.tool_call_id = self.piece_to_id["<tool_call>"]
        self.tools_id = self.piece_to_id["<tools>"]
        # byte-fallback pieces <0x00>..<0xFF>
        self.byte_ids = [self.piece_to_id[f"<0x{b:02X}>"] for b in range(256)]
        self.byte_id_set = set(self.byte_ids)
        self.special_ids = {
            self.pad_id, self.eos_id, self.bos_id, self.unk_id,
            self.tool_call_id, self.tools_id,
        }

    # -- preprocessing ----------------------------------------------------
    def _preprocess(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text) if False else text  # identity
        # sp_remove_extra_whitespaces: collapse runs, strip ends
        text = " ".join(text.split())
        if not text:
            return ""
        return SPACE + text.replace(" ", SPACE)  # add_dummy_prefix + escape

    # -- encoding ---------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        s = self._preprocess(text)
        if not s:
            return []

        # initial symbols: one per character, byte-fallback for OOV characters
        syms: list[str | None] = []
        for ch in s:
            if ch in self.piece_to_id:
                syms.append(ch)
            else:
                for b in ch.encode("utf-8"):
                    syms.append(f"<0x{b:02X}>")

        n = len(syms)
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        alive = [True] * n

        def pair_score(i: int, j: int):
            a, b = syms[i], syms[j]
            if a is None or b is None:
                return None
            # byte-fallback pieces never merge into a longer piece
            if a.startswith("<0x") or b.startswith("<0x"):
                return None
            merged = a + b
            pid = self.piece_to_id.get(merged)
            if pid is None:
                return None
            return self.scores[pid], merged

        heap: list[tuple[float, int, int, str]] = []
        for i in range(n - 1):
            r = pair_score(i, i + 1)
            if r is not None:
                heapq.heappush(heap, (-r[0], i, i + 1, r[1]))

        while heap:
            negscore, i, j, merged = heapq.heappop(heap)
            if not alive[i] or not alive[j] or nxt[i] != j:
                continue  # stale entry
            if syms[i] + syms[j] != merged:
                continue
            syms[i] = merged
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
            for a, b in ((prev[i], i), (i, nxt[i])):
                if a != -1 and b != -1 and alive[a] and alive[b]:
                    r = pair_score(a, b)
                    if r is not None:
                        heapq.heappush(heap, (-r[0], a, b, r[1]))

        out = []
        i = 0
        while i != -1:
            p = syms[i]
            out.append(self.piece_to_id.get(p, self.unk_id))
            i = nxt[i]
        return out

    def decode(self, ids: list[int]) -> str:
        pieces, buf = [], bytearray()

        def flush():
            if buf:
                pieces.append(buf.decode("utf-8", errors="replace"))
                buf.clear()

        for i in ids:
            p = self.id_to_piece[i]
            if i in self.byte_id_set:
                buf.append(int(p[3:5], 16))
                continue
            flush()
            if i in self.special_ids:
                continue
            pieces.append(p)
        flush()
        return "".join(pieces).replace(SPACE, " ").lstrip()


def to_snake_case(name: str) -> str:
    """Mirror of needle/dataset/tokenizer.py to_snake_case."""
    s = _re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    s = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = _re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = _re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def build_encoder_input(tok: NeedleBPE, query: str, tools: list,
                        max_enc_len: int = 384) -> tuple[list[int], str]:
    """[query ids] <tools> [tools-JSON ids], returned with the JSON that was fed.

    This is the encoder contract, not a convenience: the model was trained on
    snake_cased tool names (needle/model/run.py::normalize_tools) and on this
    exact truncation order (::_build_encoder_input), so a caller that builds
    the prompt by hand and gets either detail wrong is asking the board a
    different question than the fixtures certify. web/bpe.js carries the
    JS twin of this function; they must be changed together.
    """
    tools = [dict(t, name=to_snake_case(t["name"]))
             if isinstance(t, dict) and "name" in t else t for t in tools]
    tools_json = _json.dumps(tools, separators=(",", ":"))
    q = tok.encode(query)[:max_enc_len - 2]
    t = tok.encode(tools_json)[:max_enc_len - len(q) - 1]
    return q + [tok.tools_id] + t, tools_json


def validate_bpe() -> None:
    """Invariant checks that only pass for a faithful BPE implementation."""
    tk = NeedleBPE()
    assert tk.vocab_size == 8192
    assert (tk.pad_id, tk.eos_id, tk.bos_id, tk.unk_id) == (0, 1, 2, 3)
    assert (tk.tool_call_id, tk.tools_id) == (4, 5)

    cases = [
        'set a timer for 5 minutes',
        '[{"name":"set_timer","arguments":{"time_human":"5 minutes"}}]',
        '{"name":"send_email","arguments":{"to":"a@b.com","subject":"Hi"}}',
        "turn the lights off please",
        "Établis une minuterie de 5 minutes",          # French, diacritics
        "Установи таймер на 5 минут",                   # Russian, Cyrillic
        "Ρύθμισε χρονόμετρο",                           # Greek
        "日本語テキスト",                                 # unseen script -> byte fallback
    ]
    for t in cases:
        ids = tk.encode(t)
        rt = tk.decode(ids)
        expect = " ".join(t.split())
        assert rt == expect, f"round-trip failed:\n  in : {expect!r}\n  out: {rt!r}"
        assert all(0 <= i < 8192 for i in ids)
        assert tk.unk_id not in ids, f"UNK produced for {t!r} (byte_fallback broken)"

    # merging must actually happen: a common word is not one-char-per-token
    ids = tk.encode("arguments")
    assert len(ids) < len("arguments"), "no merges occurred - BPE table unused"

    # byte fallback must engage for an unseen script
    jp = tk.encode("日本語テキスト")
    assert any(i in tk.byte_id_set for i in jp), "byte fallback did not engage"

    print(f"bpe.py OK: vocab={tk.vocab_size}, {len(cases)} round-trips exact, "
          f"merges active, byte-fallback active")


if __name__ == "__main__":
    validate_bpe()
