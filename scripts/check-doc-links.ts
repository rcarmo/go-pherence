#!/usr/bin/env bun
/** Check local Markdown links, including new (not ignored) worktree files. */
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";

export function localTargets(text: string): string[] {
  // Code examples are not links. Keep inline code inside link labels intact.
  const prose = text.replace(/^\s*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\s*\1\s*$/gm, "");
  const targets: string[] = [];
  for (const re of [
    /!?\[[^\]\n]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+["'][^\n]*?["'])?\s*\)/g,
    /^\s*\[[^\]\n]+\]:\s*(<[^>]+>|\S+)/gm,
    /(?:src|href)=["']([^"']+)["']/g,
  ]) {
    for (const match of prose.matchAll(re)) {
      const target = match[1].replace(/^<|>$/g, "").split(/[?#]/, 1)[0];
      if (target && !/^(?:[a-z][a-z0-9+.-]*:|\/)/i.test(target)) targets.push(decodeURIComponent(target));
    }
  }
  return targets;
}

if (import.meta.main) {
  const root = resolve(import.meta.dir, "..");
  const tracked = Bun.spawnSync(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.md"], { cwd: root, stdout: "pipe", stderr: "inherit" });
  if (tracked.exitCode !== 0) process.exit(tracked.exitCode);
  // Deleted tracked files can still be listed before the rename is staged.
  const files = [...new Set(tracked.stdout.toString().split("\0").filter(file => file && existsSync(resolve(root, file))))];
  let broken = 0;
  for (const file of files) {
    for (const target of localTargets(await Bun.file(resolve(root, file)).text())) {
      if (!existsSync(resolve(root, dirname(file), target))) {
        console.error(`${file}: missing ${target}`);
        broken++;
      }
    }
  }
  console.log(`Checked ${files.length} Markdown files (tracked + new); broken links: ${broken}`);
  process.exit(broken === 0 ? 0 : 1);
}
