#!/usr/bin/env bun
/** Explicit, revision-pinned issue #2 asset downloads. Default is a dry run. */
import { createHash } from "node:crypto";
import { createReadStream, existsSync, lstatSync, mkdirSync, realpathSync, renameSync, rmSync, statfsSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { readdirSync } from "node:fs";

import { parseArgs } from "node:util";

type SourceFile = { path: string; size: number; sha256: string | null; git_blob: string };
type Source = { repository: string; revision: string; files: SourceFile[] };

// Account for complete assets and resumable partials, not just weight suffixes.
// The budget covers the experiment's model directories, not datasets/references.
export function plannedModelBytes(root: string, sources: Source[], tasks: {path: string; file: SourceFile}[]): number {
  const walk = (path: string): number => {
    if (!existsSync(path)) return 0;
    const st = lstatSync(path);
    if (st.isSymbolicLink()) throw new Error("symlink in model asset directory");
    if (st.isFile()) return st.size;
    if (!st.isDirectory()) throw new Error("unsupported model asset file");
    return readdirSync(path).reduce((n, name) => n + walk(resolve(path, name)), 0);
  };
  let bytes = sources.reduce((n,s) => n + walk(resolve(root, s.repository.replace("/", "--"))), 0);
  for (const t of tasks) {
    // Existing destination is verified, never replaced. Oversized partials must
    // not yield negative additions that conceal another checkpoint's footprint.
    if (existsSync(t.path)) continue;
    const partial = existsSync(t.path + ".part") ? lstatSync(t.path + ".part").size : 0;
    if (partial > t.file.size) throw new Error("partial exceeds pinned asset size");
    bytes += t.file.size - partial;
  }
  return bytes;
}

export function validatePinnedSource(source: Source) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(source.repository) || !/^[0-9a-f]{40}$/.test(source.revision)) throw new Error("full pinned repository/revision required");
  for (const file of source.files) {
    if (!file.path || file.path.startsWith("/") || file.path.includes("\\") || file.path.split("/").some(x => x === ".." || x === "." || !x)) throw new Error("unsafe asset path");
    if (!Number.isSafeInteger(file.size) || file.size < 0) throw new Error("invalid pinned size");
    if (file.sha256 !== null && !/^[0-9a-f]{64}$/.test(file.sha256) || !/^[0-9a-f]{40}$/.test(file.git_blob)) throw new Error("invalid pinned content identity");
  }
}

export async function verifyPinnedFile(path: string, file: SourceFile) {
  if (statSync(path).size !== file.size) throw new Error(`size mismatch: ${path}`);
  const sha = createHash("sha256");
  const git = createHash("sha1").update(`blob ${file.size}\0`);
  for await (const chunk of createReadStream(path)) { sha.update(chunk); git.update(chunk); }
  const hash = sha.digest("hex");
  if (file.sha256 ? hash !== file.sha256 : git.digest("hex") !== file.git_blob) throw new Error(`checksum mismatch: ${path}`);
  return hash;
}

if (import.meta.main) {
  const { values } = parseArgs({ args: Bun.argv.slice(2), options: {
    kind: { type: "string", default: "datasets" },
    output: { type: "string", default: "checkpoints/jevlike-qwen3" },
    download: { type: "boolean", default: false },
  }, strict: true });
  const root = resolve(import.meta.dir, "..");
  const manifest = await Bun.file(resolve(root, "docs/experiments/jevlike-qwen3/sources.json")).json();
  if (!['model', 'instruction-model', 'datasets'].includes(values.kind!)) throw new Error("--kind must be model, instruction-model or datasets");
  const modelDownload=values.kind!=="datasets";
  const sources: Source[] = values.kind === "model" ? [manifest.model] : values.kind === "instruction-model" ? [manifest.instruction_model] : manifest.datasets;
  const output = resolve(root, values.output!);
  if(modelDownload && output !== resolve(root, "checkpoints/jevlike-qwen3")) throw new Error("model downloads require the shared experiment budget root");
  if (output === root || output.startsWith(resolve(root, "model") + "/") || output === resolve(root, "model")) throw new Error("assets cannot be written into model source");
  let bytes = 0;
  const tasks: { path: string; url: string; file: SourceFile }[] = [];
  for (const source of sources) {
    validatePinnedSource(source);
    for (const file of source.files) {
      bytes += file.size;
      const path = resolve(output, source.repository.replace("/", "--"), source.revision, file.path);
      const prefix = modelDownload ? "" : "datasets/";
      const url = `https://huggingface.co/${prefix}${source.repository}/resolve/${source.revision}/${file.path}`;
      tasks.push({path, url, file});
    }
  }
  if (bytes > 12 * 2**30) throw new Error("manifest exceeds 12 GiB asset budget");
  console.log(JSON.stringify({ kind: values.kind, download: values.download, bytes, output, files: tasks.length }));
  if (!values.download) {
    for (const task of tasks) console.log(`${task.file.size}\t${task.url}\t${task.path}`);
    process.exit(0);
  }
  mkdirSync(output, { recursive: true });
  // Serialise budget checks and downloads. On SIGKILL, preserve the lock and
  // partials; inspect the process before explicitly removing a stale lock.
  const lock=resolve(output,".fetch-lock");
  mkdirSync(lock);
  try {
  if(modelDownload){
    if(realpathSync(output)!==resolve(realpathSync(root),"checkpoints/jevlike-qwen3"))throw new Error("symlinked budget root");
    const planned=plannedModelBytes(output,[manifest.model,manifest.instruction_model],tasks);
    if(planned>12*2**30)throw new Error("combined model assets exceed 12 GiB; preserve results then remove old reproducible weights before switching checkpoints");
  }
  const free = statfsSync(output); // reserve all missing bytes plus the safety floor
  const missing = tasks.reduce((n, t) => n + (existsSync(t.path) ? 0 : t.file.size), 0);
  if (free.bavail * free.bsize - missing < 30 * 2**30) throw new Error("download would leave less than 30 GiB free");
  const evidence = [];
  for (const task of tasks) {
    if (!existsSync(task.path)) {
      mkdirSync(dirname(task.path), { recursive: true });
      const part = task.path + ".part";
      const curl = Bun.spawn(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30", "--max-time", "3600", "--continue-at", "-", "--output", part, task.url], { stdout: "inherit", stderr: "inherit" });
      if (await curl.exited !== 0) throw new Error(`download failed; partial retained: ${part}`);
      await verifyPinnedFile(part, task.file);
      renameSync(part, task.path);
    }
    const sha256 = await verifyPinnedFile(task.path, task.file);
    evidence.push({ path: task.path, bytes: task.file.size, sha256, url: task.url });
    console.log(`verified ${task.path}`);
  }
  const record = resolve(output, `${values.kind}-verified.json`);
  await Bun.write(record + ".part", JSON.stringify({ version: 1, sources: sources.map(s => ({repository:s.repository, revision:s.revision})), files: evidence }, null, 2) + "\n");
  renameSync(record + ".part", record);
  } finally { rmSync(lock,{recursive:true}); }
}
