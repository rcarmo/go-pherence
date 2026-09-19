import { expect, test } from "bun:test";
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
function run(args: string[], cwd = root, env = process.env) {
  const result = Bun.spawnSync(args, { cwd, env, stdout: "pipe", stderr: "pipe" });
  expect(result.exitCode, result.stderr.toString()).toBe(0);
  return result.stdout.toString();
}

test("source is singular and checkpoint roots are ignored", () => {
  expect(run(["git", "ls-files", "--", "models", "checkpoints"]).trim()).toBe("");
  const paths = run(["git", "ls-files", "--", "model"]);
  for (const family of ["bert", "whisper", "speaker", "omnivoice"]) {
    expect(paths).toContain(`model/${family}/`);
  }
  for (const path of ["checkpoints/example/model.safetensors", "models/legacy/model.safetensors"]) {
    expect(run(["git", "check-ignore", path]).trim()).toBe(path);
  }
  const goFiles = run(["git", "ls-files", "-z", "*.go"]).split("\0").filter(Boolean);
  for (const file of goFiles) {
    expect(readFileSync(join(root, file), "utf8"), file).not.toContain("github.com/rcarmo/go-pherence/models/");
  }
});

test("Make defaults to checkpoints and honours explicit/legacy overrides", () => {
  const env = { ...process.env };
  delete env.MODELS_DIR;
  delete env.CHECKPOINTS_DIR;
  const dry = (...vars: string[]) => run(["make", "-n", "models-download-small", ...vars], root, env);
  expect(dry()).toContain("--checkpoints-dir checkpoints --group small");
  expect(dry("MODELS_DIR=/legacy/weights")).toContain("--checkpoints-dir /legacy/weights");
  expect(dry("MODELS_DIR=/legacy/weights", "CHECKPOINTS_DIR=/new/weights")).toContain("--checkpoints-dir /new/weights");
});

test("Python asset helpers use checkpoints by default and preserve path overrides", () => {
  const dir = mkdtempSync(join(tmpdir(), "go-pherence-layout-"));
  try {
    // A local mock prevents downloads and captures the actual destination argument.
    writeFileSync(join(dir, "huggingface_hub.py"), 'from pathlib import Path\ndef snapshot_download(**kw):\n    Path(kw["local_dir"]).mkdir(parents=True, exist_ok=True)\n    print("MOCK_DEST=" + kw["local_dir"])\n');
    const env = { ...process.env, PYTHONPATH: dir, PYTHONDONTWRITEBYTECODE: "1" };
    const download = (...args: string[]) => run(["python3", join(root, "scripts/download_models.py"), "--only", "qwen3-0.6b-mlx4", ...args], dir, env);
    expect(download()).toContain("MOCK_DEST=checkpoints/qwen3-0.6b-mlx4");
    expect(readFileSync(join(dir, "checkpoints/qwen3-0.6b-mlx4/.huggingface_model"), "utf8")).toContain("mlx-community/");
    expect(download("--models-dir", "legacy")).toContain("MOCK_DEST=legacy/qwen3-0.6b-mlx4");
    expect(download("--checkpoints-dir", "custom")).toContain("MOCK_DEST=custom/qwen3-0.6b-mlx4");
    // Run discovery in a disposable repository, with a fake Go executable.
    mkdirSync(join(dir, "scripts"));
    writeFileSync(join(dir, "scripts/minicpmv_assets_check.py"), readFileSync(join(root, "scripts/minicpmv_assets_check.py")));
    mkdirSync(join(dir, "bin"));
    writeFileSync(join(dir, "bin/go"), "#!/bin/sh\nexit 0\n");
    chmodSync(join(dir, "bin/go"), 0o755);
    for (const name of ["checkpoints", "legacy", "custom"]) {
      mkdirSync(join(dir, name, "minicpm-v-fixture"), { recursive: true });
      writeFileSync(join(dir, name, "minicpm-v-fixture/config.json"), "{}");
    }
    const inspect = (...args: string[]) => run(["python3", join(dir, "scripts/minicpmv_assets_check.py"), ...args], dir, { ...env, PATH: join(dir, "bin") + ":" + env.PATH });
    expect(inspect()).toContain("checkpoints/minicpm-v-fixture");
    expect(inspect("--models-dir", "legacy")).toContain("legacy/minicpm-v-fixture");
    expect(inspect("--checkpoints-dir", "custom")).toContain("custom/minicpm-v-fixture");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
