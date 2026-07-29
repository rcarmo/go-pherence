#!/usr/bin/env bun
/**
 * render-test-matrix.ts — Generate the go-pherence test matrix SVG.
 *
 * Usage: bun run scripts/render-test-matrix.ts [--output docs/test-matrix.svg]
 *
 * @module render-test-matrix
 * @kind entrypoint
 */

import { writeFileSync } from "fs";
import { resolve } from "path";

type Status = "pass" | "gated" | "partial" | "na";

interface Row {
  label: string;
  ref: Status;
  simd: Status;
  nvidia: Status;
  optional: Status;
  evidence: string;
}

interface Section {
  title: string;
  rows: Row[];
}

const esc = (value: string) => value
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;");

const W = 1100;
const PAD = 24;
const TOP = 130;
const HEADER_H = 30;
const SECTION_H = 26;
const ROW_H = 32;
const FOOTER_H = 34;
const BADGE_W = 78;
const BADGE_H = 20;

const COL = {
  label: 24,
  ref: 430,
  simd: 522,
  nvidia: 614,
  optional: 714,
  evidence: 816,
};

const STYLE = `
    svg {
      --ink: #1a2a40;
      --surface: rgba(232, 239, 246, 0.78);
      --surface-strong: rgba(232, 239, 246, 0.94);
      --surface-soft: rgba(232, 239, 246, 0.62);
      --line: rgba(26, 42, 64, 0.22);
      --muted: rgba(26, 42, 64, 0.72);
      --accent: #2b6cb0;
      --accent-fill: rgba(43, 108, 176, 0.14);
      --danger: #c05050;
      --danger-fill: rgba(192, 80, 80, 0.14);
      --success: #2a7a3a;
      --success-fill: rgba(42, 122, 58, 0.14);
      --warning: #c87020;
      --warning-fill: rgba(200, 112, 32, 0.16);
      --neutral-fill: rgba(26, 42, 64, 0.08);
      color-scheme: light dark;
      background: transparent;
    }
    @media (prefers-color-scheme: dark) {
      svg {
        --ink: #e8d8d0;
        --surface: rgba(42, 30, 24, 0.78);
        --surface-strong: rgba(42, 30, 24, 0.94);
        --surface-soft: rgba(42, 30, 24, 0.62);
        --line: rgba(232, 216, 208, 0.24);
        --muted: rgba(232, 216, 208, 0.78);
        --accent: #50b0a0;
        --accent-fill: rgba(80, 176, 160, 0.18);
        --danger: #e08050;
        --danger-fill: rgba(224, 128, 80, 0.18);
        --success: #60c870;
        --success-fill: rgba(96, 200, 112, 0.18);
        --warning: #e0a840;
        --warning-fill: rgba(224, 168, 64, 0.2);
        --neutral-fill: rgba(232, 216, 208, 0.1);
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
    .legend-label,
    .section-title,
    .header-text,
    .footer-text {
      font-size: 11px;
      font-weight: 700;
    }
    .legend-note,
    .cell,
    .evidence,
    .section-note,
    .badge-text {
      font-size: 11px;
    }
    .legend-note,
    .section-note,
    .evidence,
    .footer-text {
      fill: var(--muted);
    }
    .panel,
    .header,
    .section,
    .row,
    .footer {
      stroke: var(--line);
      stroke-width: 1.1;
    }
    .panel,
    .row,
    .footer {
      fill: var(--surface);
    }
    .header {
      fill: var(--accent-fill);
    }
    .section {
      fill: var(--surface-strong);
    }
    .row-alt {
      fill: var(--surface-soft);
    }
    .badge-pass {
      fill: var(--success-fill);
      stroke: var(--success);
    }
    .badge-gated {
      fill: var(--accent-fill);
      stroke: var(--accent);
      stroke-dasharray: 4 3;
    }
    .badge-partial {
      fill: var(--warning-fill);
      stroke: var(--warning);
    }
    .badge-na {
      fill: var(--neutral-fill);
      stroke: var(--line);
    }
    .badge {
      stroke-width: 1.1;
    }
  `;

const sections: Section[] = [
  {
    title: "tensor/ — reference semantics",
    rows: [
      { label: "Realize, shape, and graph invariants", ref: "pass", simd: "na", nvidia: "na", optional: "na", evidence: "Golden + NumPy parity; <1e-5 diff" },
      { label: "Fusion and rewrite reference behavior", ref: "pass", simd: "na", nvidia: "na", optional: "na", evidence: "Reference owner for later backend compares" },
    ],
  },
  {
    title: "loader/weights + loader/safetensors + loader/gguf",
    rows: [
      { label: "Directory selection, mmap, and shard metadata", ref: "pass", simd: "na", nvidia: "na", optional: "na", evidence: "Weights and safetensors contract checks" },
      { label: "GGUF tensors, quant blocks, and sidecar loading", ref: "pass", simd: "na", nvidia: "na", optional: "na", evidence: "GGUF fixtures and model-load smokes" },
    ],
  },
  {
    title: "backends/simd/runtime + backends/simd/kernels",
    rows: [
      { label: "RMSNorm, RoPE, SiLU, and GELU wrappers", ref: "pass", simd: "pass", nvidia: "na", optional: "na", evidence: "Scalar owner + AVX2/NEON parity" },
      { label: "GEMV, BF16, and quant helper dispatch", ref: "pass", simd: "pass", nvidia: "na", optional: "na", evidence: "Runtime dispatch and quant fixtures" },
    ],
  },
  {
    title: "backends/nvidia/runtime + backends/nvidia/ptx",
    rows: [
      { label: "DevBuf, uploads, placement, and module load", ref: "na", simd: "na", nvidia: "gated", optional: "na", evidence: "Availability-gated device tests" },
      { label: "PTX kernels for GEMM, GEMV, norm, and elementwise", ref: "pass", simd: "na", nvidia: "gated", optional: "na", evidence: "Compared against CPU references" },
    ],
  },
  {
    title: "backends/nvidia/ioctl",
    rows: [
      { label: "Discovery, UUID, channel, and host-memory paths", ref: "na", simd: "na", nvidia: "gated", optional: "na", evidence: "Raw ioctl smoke and boundary tests" },
      { label: "VA space and UVM edge cases", ref: "na", simd: "na", nvidia: "partial", optional: "na", evidence: "Explicit warning path retained" },
    ],
  },
  {
    title: "backends/vulkan — optional portability track",
    rows: [
      { label: "Dispatch wrappers and embedded SPIR-V plumbing", ref: "na", simd: "na", nvidia: "na", optional: "gated", evidence: "Boundary tests plus gated parity checks" },
      { label: "RMSNorm, RoPEPartial, GELU, and attention score", ref: "pass", simd: "na", nvidia: "na", optional: "gated", evidence: "Explicit wrapper path, not default LLM backend" },
    ],
  },
  {
    title: "backends/spacemit — optional K3 path",
    rows: [
      { label: "RVV, IME2, AICPU, and board wiring", ref: "na", simd: "na", nvidia: "na", optional: "gated", evidence: "Platform-gated validation only" },
      { label: "Inference and memory-surface integration", ref: "pass", simd: "na", nvidia: "na", optional: "gated", evidence: "Board-specific path stays explicit" },
    ],
  },
  {
    title: "model/ + models/ — integration smokes",
    rows: [
      { label: "CPU decode, GGUF, and loader handoff", ref: "pass", simd: "pass", nvidia: "na", optional: "na", evidence: "End-to-end reference path" },
      { label: "NVIDIA handoff and optional backend integration", ref: "pass", simd: "pass", nvidia: "gated", optional: "gated", evidence: "Opt-in compares against CPU path" },
    ],
  },
];

const STATUS_LABEL: Record<Status, string> = {
  pass: "✓ pass",
  gated: "◌ gated",
  partial: "△ partial",
  na: "— n/a",
};

function renderBadge(x: number, y: number, status: Status): string {
  return [
    `    <rect x="${x}" y="${y}" width="${BADGE_W}" height="${BADGE_H}" rx="10" class="badge badge-${status}"/>`,
    `    <text x="${x + BADGE_W / 2}" y="${y + 13}" text-anchor="middle" class="badge-text">${esc(STATUS_LABEL[status])}</text>`,
  ].join("\n");
}

let totalH = TOP + HEADER_H + 8 + FOOTER_H;
for (const section of sections) {
  totalH += SECTION_H + section.rows.length * ROW_H + 12;
}

let svg = `<!-- Generated by scripts/render-test-matrix.ts; do not edit by hand. -->\n`;
svg += `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${totalH}" role="img" aria-labelledby="matrix-title matrix-desc">\n`;
svg += `  <title id="matrix-title">Focused backend validation matrix</title>\n`;
svg += `  <desc id="matrix-desc">A scoped go-pherence validation matrix showing current package labels for tensor references, loaders, SIMD, NVIDIA, Vulkan, SpacemiT, and integration smokes. Statuses include text badges so they do not rely on color alone.</desc>\n`;
svg += `  <style>${STYLE}  </style>\n`;
svg += `  <text x="${PAD}" y="32" class="title">Focused backend validation matrix</text>\n`;
svg += `  <text x="${PAD}" y="50" class="subtitle">Scoped parity and smoke coverage for the active backend-related packages; not a repository-wide test count.</text>\n`;
svg += `  <g role="group" aria-label="Status legend">\n`;
svg += `    <text x="${PAD}" y="74" class="legend-label">Status legend</text>\n`;
svg += renderBadge(PAD, 82, "pass") + "\n";
svg += renderBadge(PAD + 94, 82, "gated") + "\n";
svg += renderBadge(PAD + 188, 82, "partial") + "\n";
svg += renderBadge(PAD + 294, 82, "na") + "\n";
svg += `    <text x="${PAD + 390}" y="95" class="legend-note">Gated = runtime, device, or platform dependent; partial = intentionally limited or still-warning surfaces.</text>\n`;
svg += `  </g>\n`;

svg += `  <rect x="${PAD}" y="${TOP}" width="${W - PAD * 2}" height="${HEADER_H}" rx="12" class="header"/>\n`;
svg += `  <text x="${COL.label}" y="${TOP + 19}" class="header-text">Package scope / check</text>\n`;
svg += `  <text x="${COL.ref}" y="${TOP + 19}" class="header-text">Reference</text>\n`;
svg += `  <text x="${COL.simd}" y="${TOP + 19}" class="header-text">SIMD</text>\n`;
svg += `  <text x="${COL.nvidia}" y="${TOP + 19}" class="header-text">NVIDIA</text>\n`;
svg += `  <text x="${COL.optional}" y="${TOP + 19}" class="header-text">Optional</text>\n`;
svg += `  <text x="${COL.evidence}" y="${TOP + 19}" class="header-text">Evidence / notes</text>\n`;

let y = TOP + HEADER_H + 8;
let rowIndex = 0;
for (const section of sections) {
  svg += `  <rect x="${PAD}" y="${y}" width="${W - PAD * 2}" height="${SECTION_H}" rx="10" class="section"/>\n`;
  svg += `  <text x="${COL.label}" y="${y + 17}" class="section-title">${esc(section.title)}</text>\n`;
  y += SECTION_H;

  for (const row of section.rows) {
    svg += `  <rect x="${PAD}" y="${y}" width="${W - PAD * 2}" height="${ROW_H}" rx="8" class="${rowIndex % 2 === 0 ? "row" : "row row-alt"}"/>\n`;
    svg += `  <text x="${COL.label}" y="${y + 20}" class="cell">${esc(row.label)}</text>\n`;
    svg += renderBadge(COL.ref, y + 6, row.ref) + "\n";
    svg += renderBadge(COL.simd, y + 6, row.simd) + "\n";
    svg += renderBadge(COL.nvidia, y + 6, row.nvidia) + "\n";
    svg += renderBadge(COL.optional, y + 6, row.optional) + "\n";
    svg += `  <text x="${COL.evidence}" y="${y + 20}" class="evidence">${esc(row.evidence)}</text>\n`;
    y += ROW_H;
    rowIndex += 1;
  }

  y += 12;
}

svg += `  <rect x="${PAD}" y="${y}" width="${W - PAD * 2}" height="${FOOTER_H}" rx="12" class="footer"/>\n`;
svg += `  <text x="${COL.label}" y="${y + 21}" class="footer-text">Scope note: this matrix tracks backend validation focus and availability gates, not repository-wide package, test, or kernel totals.</text>\n`;
svg += `</svg>\n`;

const outPath = process.argv.includes("--output")
  ? process.argv[process.argv.indexOf("--output") + 1]
  : resolve(import.meta.dir, "../docs/test-matrix.svg");

writeFileSync(outPath, svg);
console.log(`Wrote ${outPath} (${svg.length} bytes)`);
