/* Gate for the browser tokenizer.
 *
 * The page tokenizes in JavaScript now, so the JS must produce byte-for-byte
 * the same encoder ids the JAX oracle was scored against. If it drifts by a
 * single token the board is answering a different question than the fixtures
 * certify, and every accuracy claim in the README stops applying.
 *
 *   node web/verify_bpe.mjs      # expects 8/8, exits non-zero otherwise
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { NeedleBPE, buildEncoderInput } from "./bpe.js";

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const tok = new NeedleBPE(readFileSync(join(REPO, "tokenizer", "vocab.txt"), "utf8"));

const fixtures = JSON.parse(readFileSync(join(REPO, "test", "parity_set.json"), "utf8")).cases;
const oracle = JSON.parse(readFileSync(join(REPO, "weights", "reference_int4p_int8e.json"), "utf8")).cases;
const expected = new Map(oracle.map(c => [c.id, c.enc_ids]));

let pass = 0;
for (const c of fixtures) {
  const tools = typeof c.tools === "string" ? JSON.parse(c.tools) : c.tools;
  const { ids } = buildEncoderInput(tok, c.query, tools);
  const want = expected.get(c.id);
  const same = want && ids.length === want.length && ids.every((v, i) => v === want[i]);
  pass += same ? 1 : 0;
  console.log(`${c.id.padEnd(22)} ${same ? "MATCH" : "DIFF "}  got=${ids.length} want=${want ? want.length : "?"}`);
  if (!same && want) {
    let i = 0;
    while (i < Math.min(ids.length, want.length) && ids[i] === want[i]) i++;
    console.log(`    diverges at ${i}: got ${ids.slice(i, i + 6)} want ${want.slice(i, i + 6)}`);
  }
}

// round-trip: decoding the oracle's generated ids must give back valid output
let rt = 0;
for (const c of oracle) {
  const text = tok.decode(c.gen_ids);
  try { JSON.parse(text); rt++; } catch { console.log(`decode not JSON for ${c.id}: ${text}`); }
}

console.log(`\n${pass}/${fixtures.length} tokenizations match the oracle`);
console.log(`${rt}/${oracle.length} detokenized outputs parse as JSON`);
process.exit(pass === fixtures.length && rt === oracle.length ? 0 : 1);
