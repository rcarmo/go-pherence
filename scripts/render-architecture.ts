#!/usr/bin/env bun
/**
 * render-architecture.ts — Generate the go-pherence architecture SVG diagram.
 *
 * Usage: bun run scripts/render-architecture.ts [--output docs/architecture.svg]
 *
 * @module render-architecture
 * @kind entrypoint
 */

import { writeFileSync } from "fs";
import { resolve } from "path";

type BoxKind = "neutral" | "accent" | "success" | "warning" | "danger";

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  sub?: string;
  kind: BoxKind;
  optional?: boolean;
}

interface Zone {
  y: number;
  h: number;
  label: string;
  blurb: string;
  boxes: Box[];
}

interface Link {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  kind?: "main" | "optional";
}

const esc = (value: string) => value
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;");

const W = 960;
const H = 648;
const PAD = 28;
const INNER_W = W - PAD * 2;

const STYLE = `
    svg {
      --ink: #1a2a40;
      --surface: rgba(232, 239, 246, 0.78);
      --surface-strong: rgba(232, 239, 246, 0.94);
      --surface-soft: rgba(232, 239, 246, 0.58);
      --line: rgba(26, 42, 64, 0.22);
      --muted: rgba(26, 42, 64, 0.72);
      --accent: #2b6cb0;
      --accent-fill: rgba(43, 108, 176, 0.12);
      --danger: #c05050;
      --danger-fill: rgba(192, 80, 80, 0.12);
      --success: #2a7a3a;
      --success-fill: rgba(42, 122, 58, 0.12);
      --warning: #c87020;
      --warning-fill: rgba(200, 112, 32, 0.12);
      color-scheme: light dark;
      background: transparent;
    }
    @media (prefers-color-scheme: dark) {
      svg {
        --ink: #e8d8d0;
        --surface: rgba(42, 30, 24, 0.78);
        --surface-strong: rgba(42, 30, 24, 0.94);
        --surface-soft: rgba(42, 30, 24, 0.58);
        --line: rgba(232, 216, 208, 0.24);
        --muted: rgba(232, 216, 208, 0.78);
        --accent: #50b0a0;
        --accent-fill: rgba(80, 176, 160, 0.16);
        --danger: #e08050;
        --danger-fill: rgba(224, 128, 80, 0.16);
        --success: #60c870;
        --success-fill: rgba(96, 200, 112, 0.16);
        --warning: #e0a840;
        --warning-fill: rgba(224, 168, 64, 0.18);
      }
    }
    text {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
      fill: var(--ink);
    }
    .title {
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    .subtitle {
      font-size: 12px;
      fill: var(--muted);
    }
    .zone {
      fill: var(--surface);
      stroke: var(--line);
      stroke-width: 1.2;
    }
    .zone-label {
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      fill: var(--muted);
    }
    .zone-blurb {
      font-size: 11px;
      fill: var(--muted);
    }
    .card {
      fill: var(--surface-strong);
      stroke: var(--line);
      stroke-width: 1.2;
    }
    .card-neutral { fill: var(--surface-strong); }
    .card-accent { fill: var(--accent-fill); stroke: var(--accent); }
    .card-success { fill: var(--success-fill); stroke: var(--success); }
    .card-warning { fill: var(--warning-fill); stroke: var(--warning); }
    .card-danger { fill: var(--danger-fill); stroke: var(--danger); }
    .card-optional {
      stroke-dasharray: 6 4;
    }
    .box-label {
      font-size: 13px;
      font-weight: 700;
    }
    .box-sub {
      font-size: 11px;
      fill: var(--muted);
    }
    .tag {
      font-size: 11px;
      font-weight: 700;
      fill: var(--warning);
    }
    .link-main {
      stroke: var(--accent);
      stroke-width: 1.8;
      fill: none;
      marker-end: url(#arrow-main);
    }
    .link-optional {
      stroke: var(--warning);
      stroke-width: 1.6;
      fill: none;
      stroke-dasharray: 5 4;
      marker-end: url(#arrow-optional);
    }
    .scope-note {
      font-size: 11px;
      fill: var(--muted);
    }
  `;

const zones: Zone[] = [
  {
    y: 76,
    h: 92,
    label: "CLI surfaces",
    blurb: "Current entry points into the LLM and model-loading path.",
    boxes: [
      { x: 24, y: 50, w: 270, h: 30, label: "cmd/llm/llmgen", sub: "prompt, sampling, and GPU flags", kind: "accent" },
      { x: 318, y: 50, w: 270, h: 30, label: "cmd/llm/* + cmd/qwen/*", sub: "specbench, speccheck, qwen36run", kind: "neutral" },
      { x: 612, y: 50, w: 270, h: 30, label: "cmd/models/*", sub: "inspectors and focused smoke tools", kind: "neutral" },
    ],
  },
  {
    y: 190,
    h: 98,
    label: "Loaders and assets",
    blurb: "Weight discovery, metadata inspection, and format-specific tensor readers.",
    boxes: [
      { x: 24, y: 54, w: 270, h: 30, label: "loader/weights", sub: "directory selection, sidecars, and mmap inputs", kind: "warning" },
      { x: 318, y: 54, w: 270, h: 30, label: "loader/safetensors", sub: "shards, metadata, compressed tensor families", kind: "warning" },
      { x: 612, y: 54, w: 270, h: 30, label: "loader/gguf", sub: "GGUF tensors, quant blocks, tokenizer sidecars", kind: "warning" },
    ],
  },
  {
    y: 310,
    h: 114,
    label: "Execution core",
    blurb: "Model surfaces, shared tensor/runtime plumbing, and non-LLM native models.",
    boxes: [
      { x: 24, y: 58, w: 270, h: 36, label: "model/", sub: "GGUF and native LLM decode, MoE, GPU handoff", kind: "success" },
      { x: 318, y: 58, w: 270, h: 36, label: "tensor/ + runtime/*", sub: "graph, KV, memory, and quantized execution surfaces", kind: "accent" },
      { x: 612, y: 58, w: 270, h: 36, label: "models/", sub: "audio, speaker, and native checkpoint integrations", kind: "success" },
    ],
  },
  {
    y: 446,
    h: 164,
    label: "Backend dispatch",
    blurb: "Default CPU and NVIDIA paths, with explicit optional portability tracks.",
    boxes: [
      { x: 24, y: 56, w: 270, h: 32, label: "backends/simd", sub: "scalar reference plus AVX2 / NEON / RVV dispatch", kind: "accent" },
      { x: 318, y: 56, w: 270, h: 32, label: "backends/nvidia/runtime", sub: "device init, buffers, placement, and launch plumbing", kind: "success" },
      { x: 612, y: 56, w: 270, h: 32, label: "NVIDIA + CUDA PTX assets", sub: "backends/nvidia/ptx + backends/cuda/ptx", kind: "success" },
      { x: 24, y: 106, w: 270, h: 32, label: "backends/vulkan", sub: "optional portability track for explicit wrapper use", kind: "warning", optional: true },
      { x: 318, y: 106, w: 270, h: 32, label: "backends/spacemit", sub: "optional K3 / RVV / IME2 board-specific path", kind: "warning", optional: true },
    ],
  },
];

const links: Link[] = [
  { x1: 187, y1: 156, x2: 187, y2: 244 },
  { x1: 481, y1: 156, x2: 481, y2: 244 },
  { x1: 775, y1: 156, x2: 775, y2: 244 },
  { x1: 187, y1: 274, x2: 187, y2: 368 },
  { x1: 481, y1: 274, x2: 481, y2: 368 },
  { x1: 775, y1: 274, x2: 775, y2: 368 },
  { x1: 294, y1: 386, x2: 318, y2: 386 },
  { x1: 588, y1: 386, x2: 612, y2: 386 },
  { x1: 187, y1: 404, x2: 187, y2: 502 },
  { x1: 481, y1: 404, x2: 481, y2: 502 },
  { x1: 775, y1: 404, x2: 775, y2: 502 },
  { x1: 588, y1: 518, x2: 612, y2: 518 },
  { x1: 159, y1: 534, x2: 159, y2: 552, kind: "optional" },
  { x1: 453, y1: 534, x2: 453, y2: 552, kind: "optional" },
];

function renderBox(box: Box): string {
  const cx = box.x + box.w / 2;
  const labelY = box.sub ? box.y + 15 : box.y + 21;
  const sub = box.sub
    ? `<tspan x="${cx}" dy="13" class="box-sub">${esc(box.sub)}</tspan>`
    : "";
  const optionalTag = box.optional
    ? `    <text x="${box.x + box.w - 12}" y="${box.y + 16}" text-anchor="end" class="tag">optional</text>\n`
    : "";

  return [
    `    <rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" rx="10" class="card card-${box.kind}${box.optional ? " card-optional" : ""}"/>`,
    optionalTag.trimEnd(),
    `    <text x="${cx}" y="${labelY}" text-anchor="middle" class="box-label">`,
    `      <tspan x="${cx}" dy="0">${esc(box.label)}</tspan>${sub}`,
    `    </text>`,
  ].filter(Boolean).join("\n") + "\n";
}

let svg = `<!-- Generated by scripts/render-architecture.ts; do not edit by hand. -->\n`;
svg += `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="arch-title arch-desc">\n`;
svg += `  <title id="arch-title">Core inference path</title>\n`;
svg += `  <desc id="arch-desc">A scoped go-pherence architecture diagram showing current CLI entry points, loaders, core model surfaces, default SIMD and NVIDIA backends, and optional Vulkan and SpacemiT branches.</desc>\n`;
svg += `  <style>${STYLE}  </style>\n`;
svg += `  <defs>\n`;
svg += `    <marker id="arrow-main" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n`;
svg += `      <path d="M0,0 L9,4.5 L0,9 z" fill="var(--accent)"/>\n`;
svg += `    </marker>\n`;
svg += `    <marker id="arrow-optional" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="strokeWidth">\n`;
svg += `      <path d="M0,0 L9,4.5 L0,9 z" fill="var(--warning)"/>\n`;
svg += `    </marker>\n`;
svg += `  </defs>\n`;
svg += `  <text x="${PAD}" y="32" class="title">Core inference path</text>\n`;
svg += `  <text x="${PAD}" y="50" class="subtitle">go-pherence architecture · loaders, model surfaces, and backend dispatch without repository-wide counts</text>\n`;

for (const zone of zones) {
  svg += `  <g transform="translate(${PAD},${zone.y})">\n`;
  svg += `    <rect x="0" y="0" width="${INNER_W}" height="${zone.h}" rx="16" class="zone"/>\n`;
  svg += `    <text x="20" y="26" class="zone-label">${esc(zone.label)}</text>\n`;
  svg += `    <text x="20" y="44" class="zone-blurb">${esc(zone.blurb)}</text>\n`;
  for (const box of zone.boxes) {
    svg += renderBox(box);
  }
  svg += `  </g>\n`;
}

for (const link of links) {
  svg += `  <line x1="${link.x1 + PAD}" y1="${link.y1}" x2="${link.x2 + PAD}" y2="${link.y2}" class="${link.kind === "optional" ? "link-optional" : "link-main"}"/>\n`;
}

svg += `  <text x="${PAD}" y="634" class="scope-note">Default LLM execution stays on the SIMD and NVIDIA paths; Vulkan and SpacemiT remain explicit optional branches.</text>\n`;
svg += `</svg>\n`;

const outPath = process.argv.includes("--output")
  ? process.argv[process.argv.indexOf("--output") + 1]
  : resolve(import.meta.dir, "../docs/architecture.svg");

writeFileSync(outPath, svg);
console.log(`Wrote ${outPath} (${svg.length} bytes)`);
