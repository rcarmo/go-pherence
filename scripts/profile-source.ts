/**
 * Lexical allocation/SIMD inventory of tracked Go source (not a hotness proof).
 * Run: bun scripts/profile-source.ts > inventory.json
 * No weights, runtime inputs or held-out records are read. Hardware is not used.
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";

// Blank comments and literals while preserving offsets/line numbers. This is a
// lexical scan, not Go escape analysis: make/append sites need measured profiles.
export function codeOnly(source: string): string {
  return source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\/|`[^`]*`|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g,
    token => token.replace(/[^\n]/g, " "));
}

export function scanSource(path: string, source: string) {
  const code = codeOnly(source);
  const tests = [...code.matchAll(/\bfunc\s+(Test\w+)\s*\(/g)].length;
  const benchmarks = [...code.matchAll(/\bfunc\s+(Benchmark\w+)\s*\(/g)].length;
  const generated = /^\/\/ Code generated .*DO NOT EDIT\./m.test(source);
  const sites: { line: number; kind: string; code: string }[] = [];
  let loops = 0;
  let simdCalls = 0;
  if (!path.endsWith("_test.go") && !generated) {
    code.split("\n").forEach((line, index) => {
      loops += [...line.matchAll(/\bfor\b/g)].length;
      simdCalls += [...line.matchAll(/\b(?:simd|kernels)\.[A-Za-z_]\w*\s*\(/g)].length;
      for (const [kind, re] of [
        ["allocation-candidate", /\b(?:make|new|append)\s*\(/],
        ["scalar-math", /\bmath\.(?:Exp|Exp2|Log|Log2|Sqrt|Pow|Sin|Cos|Tanh)\s*\(/],
        ["sort", /\b(?:sort|slices)\.(?:Sort\w*|Stable|Slice\w*)\s*\(/],
        ["numeric-loop-candidate", /\b(?:sum|acc|ss|dot|mean|variance)\w*\s*(?:\+=|=\s*\w+\s*\+)/i],
      ] as const) {
        if (re.test(line)) sites.push({ line: index + 1, kind, code: line.trim() });
      }
    });
  }
  return { path, generated, tests, benchmarks, loops, simdCalls, sites };
}

export function inventory(files: { path: string; source: string }[]) {
  const scanned = files.filter(f => f.path.endsWith(".go"))
    .map(f => scanSource(f.path, f.source));
  const packages = new Map<string, { path: string; sourceFiles: number; testFiles: number; tests: number; benchmarks: number; allocationSites: number; scalarMathSites: number; numericLoopSites: number; simdCalls: number }>();
  for (const f of scanned) {
    const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : ".";
    const p = packages.get(dir) ?? { path: dir, sourceFiles: 0, testFiles: 0, tests: 0, benchmarks: 0, allocationSites: 0, scalarMathSites: 0, numericLoopSites: 0, simdCalls: 0 };
    if (f.path.endsWith("_test.go")) p.testFiles++; else p.sourceFiles++;
    p.tests += f.tests; p.benchmarks += f.benchmarks; p.simdCalls += f.simdCalls;
    p.allocationSites += f.sites.filter(s => s.kind === "allocation-candidate").length;
    p.scalarMathSites += f.sites.filter(s => s.kind === "scalar-math").length;
    p.numericLoopSites += f.sites.filter(s => s.kind === "numeric-loop-candidate").length;
    packages.set(dir, p);
  }
  return {
    scope: "Tracked Go files across all build tags; comments/literals excluded. Lexical candidates, not measured allocations, escape analysis or SIMD eligibility. Import aliases and noncanonical reduction variable names can be missed. Tests/benchmarks are declarations, not execution coverage.",
    files: scanned,
    packages: [...packages.values()].sort((a, b) => a.path.localeCompare(b.path)),
  };
}

if (import.meta.main) {
  const paths = execFileSync("git", ["ls-files", "-z", "--", "*.go"], { encoding: "utf8" }).split("\0").filter(Boolean);
  console.log(JSON.stringify(inventory(paths.map(path => ({ path, source: readFileSync(path, "utf8") }))), null, 2));
}
