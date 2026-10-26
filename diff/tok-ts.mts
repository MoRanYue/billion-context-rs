// Emits the corpus and the reference kernel's counts for it. The Rust side
// reads this file and must reproduce every number exactly.
//
//   node diff/tok-ts.mts diff/tok-ts.json
//
// The corpus is built here (not in Rust) so both sides count byte-identical
// strings; the boundary block below walks the exact edges of the kernel's CJK
// regex, which is where a hand-ported character class drifts first.
import fs from "node:fs";
import { defaultCountTokens } from "file:///D:/GITHUB/billion-context/kernel/src/tokenize.ts";

const corpus: string[] = [
  "",
  "a",
  "abcd",
  "abcde",
  "hello world",
  "The quick brown fox jumps over the lazy dog.",
  "这是一段中文散文，用来测试中日韩字符的计数。",
  "日本語のテキスト、カタカナとひらがな。",
  "한국어 텍스트입니다.",
  "中英 mixed 混排 with code `let x = 1;`",
  // Non-BMP: two UTF-16 units each, and NOT inside the CJK ranges.
  "😀",
  "😀😀😀😀",
  "👨‍👩‍👧‍👦",
  "𠀀",
  // A combining mark is two code points but one grapheme.
  "e\u0301",
  "tab\there\nnewline\r\n",
  "     ",
  "\n\n\n\n",
  '{"a":1,"b":[2,3],"c":"中文"}',
  'fn main() { println!("hi"); }',
  "a".repeat(1000),
  "中".repeat(1000),
  "x".repeat(999),
  "😀".repeat(500),
  "The 中文 mixed テキスト 한글 and emoji 😀 end.",
];

// Walk the edges of /[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]/: one code point
// on each side of every boundary, singly and repeated.
for (const cp of [
  0x303f, 0x3040, 0x3041, 0x30fe, 0x30ff, 0x3100, 0x4dff, 0x4e00, 0x9ffe,
  0x9fff, 0xa000, 0xabff, 0xac00, 0xd7ae, 0xd7af, 0xd7b0,
]) {
  const ch = String.fromCodePoint(cp);
  corpus.push(ch, ch.repeat(8), ch + "abc" + ch);
}

const counts = corpus.map((t) => defaultCountTokens(t));

fs.writeFileSync(
  process.argv[2],
  JSON.stringify({ corpus, counts }, null, 2),
);
console.log("wrote", process.argv[2], "samples:", corpus.length);
