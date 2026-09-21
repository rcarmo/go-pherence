#!/usr/bin/env bun
/** Freeze a two-stage native-port comparison without reading the spent test set. */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

const hash = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");
const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    dataset: { type: "string", default: "checkpoints/jevlike-qwen3/pilot-v4" },
    spent: { type: "string", default: "checkpoints/jevlike-qwen3/direct-pilot-v1/requests.jsonl" },
    output: { type: "string", default: "checkpoints/jev-port-bakeoff/v1" },
    "screen-per-task": { type: "string", default: "8" },
    "screen-controls-per-task": { type: "string", default: "1" },
    "final-evidence-controls-per-task": { type: "string", default: "12" },
  },
});
const dataset = resolve(values.dataset!);
const output = resolve(values.output!);
const screenN = Number(values["screen-per-task"]);
const screenControlN = Number(values["screen-controls-per-task"]);
const finalEvidenceN = Number(values["final-evidence-controls-per-task"]);
if (!Number.isInteger(screenN) || screenN < 2 || screenN > 32 || !Number.isInteger(screenControlN) || screenControlN < 1 || screenControlN > screenN || !Number.isInteger(finalEvidenceN) || finalEvidenceN < 1 || finalEvidenceN > 32) throw new Error("invalid cohort bounds");
if (existsSync(output)) throw new Error("output exists");

const manifestBytes = readFileSync(dataset + "/manifest.json");
const manifest = JSON.parse(manifestBytes.toString());
const artifact = (name: string) => {
  const a = manifest.artifacts.find((x: any) => x.path === name);
  if (!a) throw new Error("missing artifact " + name);
  const bytes = readFileSync(dataset + "/" + name);
  if (hash(bytes) !== a.sha256) throw new Error("artifact identity changed: " + name);
  return { bytes, spec: a };
};
const validation = artifact("validation.jsonl");
const provenance = artifact("provenance.jsonl");
const choices = validation.bytes.toString().trim().split("\n").map(JSON.parse);
const meta = new Map<number, any>();
for (const p of provenance.bytes.toString().trim().split("\n").map(JSON.parse)) if (p.partition === "validation" && !meta.has(p.output_row)) meta.set(p.output_row, p);

const spentBytes = readFileSync(resolve(values.spent!));
const spent = new Set<string>();
for (const row of spentBytes.toString().trim().split("\n").map(JSON.parse)) if (row.variant === "normal") spent.add(row.task + "\0" + row.id);
if (spent.size !== 60) throw new Error(`expected 60 spent validation originals, got ${spent.size}`);

const groups = new Map<string, any[]>();
for (const [row, p] of meta) {
  if (!Number.isInteger(row) || row < 1 || row > choices.length || !manifest.inputs[p.input]) throw new Error("invalid provenance row");
  const input = manifest.inputs[p.input];
  const task = input.source + (input.config ? "/" + input.config : "");
  if (spent.has(task + "\0" + p.source_id)) continue;
  const list = groups.get(task) ?? [];
  list.push({ p, ex: choices[row - 1] });
  groups.set(task, list);
}
if (groups.size !== 5) throw new Error(`expected five tasks, got ${groups.size}`);
for (const list of groups.values()) list.sort((a, b) => hash("jev-port-bakeoff-v1:" + a.p.source_id).localeCompare(hash("jev-port-bakeoff-v1:" + b.p.source_id)));

function adapt(task: string, x: any) {
  const context = x.ex.context as string;
  let evidence = "", question = context;
  if (task.startsWith("multi_nli")) {
    const start = context.indexOf("Premise: ") + 9, end = context.indexOf("\nHypothesis: ");
    if (start < 9 || end < start) throw new Error("NLI context contract");
    evidence = context.slice(start, end);
    question = "Given the evidence, classify this hypothesis: " + context.slice(end + 13).replace(/\nAnswer:$/, " ").trim();
  } else if (task.startsWith("clinc_oos")) {
    evidence = context.split("Utterance: ")[1]?.replace(/\nIntent:$/, "");
    if (evidence === undefined) throw new Error("CLINC context contract");
    question = "Which intent best matches the utterance?";
  } else question = context.replace(/^Task:[^\n]*\nQuestion: /, "").replace(/\nAnswer:$/, "");
  const candidates = x.ex.options.map((text: string) => ({ id: hash(text).slice(0, 16), text }));
  return { id: x.p.source_id, task, gold_id: candidates[x.ex.label].id, request: { evidence, question, candidates, temperature: 1 } };
}
function swappedEvidence(task: string, other: any) {
  const context = other.ex.context as string;
  if (task.startsWith("multi_nli")) return context.split("Premise: ")[1].split("\nHypothesis: ")[0];
  if (task.startsWith("clinc_oos")) return context.split("Utterance: ")[1].replace(/\nIntent:$/, "");
  throw new Error("evidence control on unsupported task");
}
function controls(task: string, selected: any[], limit: number) {
  const rows: any[] = [];
  for (let i = 0; i < Math.min(limit, selected.length); i++) {
    const base = adapt(task, selected[i]);
    rows.push({ ...base, variant: "reverse-options", request: { ...base.request, candidates: [...base.request.candidates].reverse() } });
    if (base.request.evidence) {
      let replacement = base.request.evidence;
      for (let offset = 1; offset < selected.length && replacement === base.request.evidence; offset++) replacement = swappedEvidence(task, selected[(i + offset) % selected.length]);
      if (replacement === base.request.evidence) throw new Error("no distinct shuffled evidence");
      rows.push({ ...base, variant: "shuffled-evidence", request: { ...base.request, evidence: replacement } });
      rows.push({ ...base, variant: "no-evidence", request: { ...base.request, evidence: "No evidence supplied." } });
    }
  }
  return rows;
}

const screening: any[] = [], finalist: any[] = [], counts: any[] = [];
for (const [task, available] of [...groups].sort()) {
  if (available.length <= screenN) throw new Error("not enough untouched rows for " + task);
  const screen = available.slice(0, screenN), final = available.slice(screenN);
  for (const x of screen) screening.push({ ...adapt(task, x), variant: "normal" });
  screening.push(...controls(task, screen, screenControlN));
  for (const x of final) finalist.push({ ...adapt(task, x), variant: "normal" });
  finalist.push(...controls(task, final, final.length));
  if (task.startsWith("multi_nli") || task.startsWith("clinc_oos")) {
    // Keep every reverse row, but bound the additional evidence ablations.
    const keep = new Set(final.slice(0, finalEvidenceN).map(x => x.p.source_id));
    for (let i = finalist.length - 1; i >= 0; i--) if ((finalist[i].variant === "shuffled-evidence" || finalist[i].variant === "no-evidence") && finalist[i].task === task && !keep.has(finalist[i].id)) finalist.splice(i, 1);
  }
  counts.push({ task, available_after_spent: available.length, screening_originals: screen.length, finalist_originals: final.length });
}
function writeCohort(name: string, rows: any[]) {
  const payload = rows.map(JSON.stringify).join("\n") + "\n";
  writeFileSync(output + "/" + name + ".jsonl", payload);
  return { path: name + ".jsonl", rows: rows.length, originals: new Set(rows.map(x => x.task + "\0" + x.id)).size, sha256: hash(payload) };
}
mkdirSync(output, { recursive: true });
const screeningArtifact = writeCohort("screening", screening);
const finalistArtifact = writeCohort("finalist", finalist);
const selection = {
  version: 1,
  scope: "jev-port-bakeoff-v1",
  partition: "validation",
  source_manifest_sha256: hash(manifestBytes),
  partition_sha256: validation.spec.sha256,
  provenance_sha256: provenance.spec.sha256,
  spent_requests_sha256: hash(spentBytes),
  spent_originals: spent.size,
  cohorts: { screening: screeningArtifact, finalist: finalistArtifact },
  counts,
  candidates: ["openjev", "decider", "laya", "nimble"],
  policies: [
    "never read or score the spent 1,440-row test partition",
    "exclude every original used by direct-pilot-v1",
    "screen by lowest sha256(jev-port-bakeoff-v1:source_id) per source/config",
    "apply the 48 GiB RSS and 60 second/request resource gate before scoring; Nimble is inadmissible from published 59 GiB and roughly four minute measurements",
    "score OpenJEV, Decider and Laya on screening",
    "promote the two highest-accuracy complete practical candidates; ties resolve by fewer order changes then lower decision time",
    "select the winner only on the untouched finalist cohort: accuracy, then fewer order changes, then evidence sensitivity, then decision time",
    "candidate text and stable IDs are unchanged across variants",
    "no calibration, prompt tuning, training or threshold fitting on either cohort",
  ],
};
writeFileSync(output + "/selection.json", JSON.stringify(selection, null, 2) + "\n");
console.log(JSON.stringify({ output, screening: screeningArtifact, finalist: finalistArtifact, counts }, null, 2));
