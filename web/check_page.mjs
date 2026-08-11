/* Catch the class of bug that shipped twice: an identifier used by the page
 * script but never defined in it. Cheap static check, no browser needed. */
import { readFileSync } from "node:fs";
const html = readFileSync(process.argv[2] || "web/index.html", "utf8");
const raw = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1];
// strip comments only: prose in a comment is not a call, but stripping string
// literals too proved fragile enough to make the check silently pass
const src = raw.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:"'`])\/\/[^\n]*/g, "$1 ");

const declared = new Set([...src.matchAll(/\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)/g)].map(m => m[1]));
// object-literal and class method shorthand:  name(args) {
for (const m of src.matchAll(/^\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{/gm)) declared.add(m[1]);
// destructured bindings:  const { a, b } = ...   and   ({ a, b }) =>
for (const m of src.matchAll(/\{([^{}]*)\}\s*(?:=|=>|\))/g))
  m[1].split(",").forEach(n => { const t = n.split(":").pop().trim().replace(/\s*=.*$/, ""); if (/^[A-Za-z_$][\w$]*$/.test(t)) declared.add(t); });
for (const m of src.matchAll(/\bimport\s*\{([^}]+)\}/g))
  m[1].split(",").forEach(n => declared.add(n.trim().split(/\s+as\s+/).pop()));
const GLOBALS = new Set(["window","document","fetch","console","JSON","Math","Number","String","Object","Array",
  "Promise","Date","setInterval","clearInterval","setTimeout","requestAnimationFrame","cancelAnimationFrame",
  "performance","navigator","getComputedStyle","encodeURIComponent","encodeURI","parseInt","parseFloat",
  "AudioContext","webkitAudioContext","Int32Array","Uint8Array","TextEncoder","TextDecoder","Error","Set","Map","RegExp","isNaN"]);
const used = new Set([...src.matchAll(/(?<![.\w$])([A-Za-z_$][\w$]*)\s*\(/g)].map(m => m[1]));
const missing = [...used].filter(n => !declared.has(n) && !GLOBALS.has(n) &&
  !/^(if|for|while|switch|catch|return|typeof|function|new|await|of|in|do|else|async|var|let|const|throw|delete|void|yield|case)$/.test(n));
if (missing.length) { console.error("UNDEFINED in page script:", missing.join(", ")); process.exit(1); }
console.log(`page script: ${declared.size} declarations, no undefined calls`);
