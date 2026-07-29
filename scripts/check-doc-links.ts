#!/usr/bin/env bun
/** Check relative links in Git-tracked Markdown files. */
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const tracked = Bun.spawnSync(["git", "ls-files", "*.md"], { cwd: root, stdout: "pipe", stderr: "inherit" });
if (tracked.exitCode !== 0) process.exit(tracked.exitCode);
const files = new TextDecoder().decode(tracked.stdout).trim().split("\n").filter(Boolean);
let broken = 0;
for (const file of files) {
  const text = await Bun.file(resolve(root, file)).text();
  for (const match of text.matchAll(/\[[^\]]*\]\(([^)]+)\)/g)) {
    const target = match[1].split("#", 1)[0];
    if (!target || /^(https?:|mailto:|attachment:)/.test(target)) continue;
    const path = resolve(root, dirname(file), decodeURIComponent(target));
    if (!existsSync(path)) {
      console.error(`${file}: missing ${match[1]}`);
      broken++;
    }
  }
}
console.log(`Checked ${files.length} tracked Markdown files; broken links: ${broken}`);
process.exit(broken === 0 ? 0 : 1);
