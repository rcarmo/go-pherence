# Needle bounded tool-call generation — 2026-09-20

## Scope

This adds a Go schema compiler, byte-oriented prefix recognizer, cached greedy token masking, prompt renderer and `cmd/needle -mode tools`. It generates call data and never executes tools. The [usage guide](../guides/needle-tool-calls.md) defines the accepted schema subset, limits and error behavior.

The prompt layout follows Needle 3 at `fc5bae0f9b6138828fe7589f6b531fb9a26968de`. Upstream's actual grammar engine is distributed as a native library; this implementation does not claim engine equivalence. No native tool executor was downloaded or run. Needle 2's existing numerical fixtures still pass, but its tool-use behavior is not qualified by this work.

## Correctness and admission

Tests cover:

- Partial prefixes, exact call names/argument keys, scalar types, sorted required properties, arrays, depth and call limits, and incomplete JSON.
- Unsupported keywords, references, optional properties, malformed metadata/bounds and incompatible enum members; rejection occurs rather than silently discarding constraints.
- Conservative program-expansion admission before regex simplification/compilation.
- Valid multibyte UTF-8 split across tokens, valid surrogate pairs and rejection of unpaired generated surrogates.
- Immutable source states, reused-scratch versus whole-prefix acceptance, and raw tokenizer bytes for every byte value. Special/control tokens cannot masquerade as ordinary JSON text.
- Reserved prompt markers in query/system text and decoded schema values. Duplicate schema keys, including escaped spellings, are rejected: otherwise a last-key-wins decoder could miss a marker still present in the prompt's raw schema JSON.
- Cancellation, invalid budgets, missing tokenizer markers and exhausted output tokens. API exhaustion returns an error, `Complete=false` and no completed `Calls`; CLI failures emit no result JSON.
- CLI rejection of incompatible or ignored controls, unknown schema-envelope fields and trailing input.

A final five-second grammar fuzz request ran for approximately six seconds and completed 227,833 executions without invalid-JSON acceptance. This is a short fuzz smoke check, not exhaustive schema verification. Earlier smoke fuzzing completed 284,119 executions. Independent delegate review attempts timed out or aborted without a review; **there is no independent review coverage**.

## Released model, Intel and native ARM

The local Needle 3 archive is pinned at `b274efcb211a9eef48c9a88da4b43bd569696a39` (35,335,380 archive bytes; 484,087,640 bytes of decoded weight materialization). With the single boolean `light` schema and “Turn the light on.”, decoded full-depth generation returned:

```json
[{"name":"light","arguments":{"on":true}}]
```

The eleven generated IDs were `507,379,300,1808,301,612,327,290,288,500,572`. Output metadata reports `complete:true`, `executed:false`, `calibrated:false`. The opt-in test also sets the token budget to one and verifies failure with no completed calls.

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 \
  GO_PHERENCE_NEEDLE_TOOLS_MODEL="$PWD/checkpoints/needle3/needle3.cact" \
  go test ./model/needle -run '^TestReleasedToolCall$' -count=1 -timeout=120s -v
```

On Intel i7-12700, the final opt-in test passed in 3.60 seconds; it includes loading, complete generation and the exhaustion case. An earlier CLI process measurement was 2.08 seconds wall time and 1,092,480 KiB maximum RSS. That process includes cold loading/materialization and is not a steady-state generation or memory baseline.

On `agent@orangepi6plus.local` (CIX P1), the final native run used cross-built ARM64 test executables with `GOMAXPROCS=2 nice -n 10`. It passed **104 model tests/subtests, 48 loader tests/subtests and eight CLI tests**. The released-model complete/exhaustion test passed in **5.43 seconds**. These were actual native executions, not merely cross-compilation. Earlier pre-admission-tightening runs passed 101/48/7 and 5.41 seconds; the final counts supersede them.

One deterministic tool smoke prompt is not a tool-use quality benchmark, calibration exercise or proof that arbitrary schemas will finish inside their budgets. Packed tool generation, sliced-depth quality and Needle 2 tool-use behavior were not included in the released-model checks.

## Grammar allocations

Intel microbenchmarks cover only one short boolean-call grammar and seven fragments; they exclude token ranking, model inference, loading and compilation:

| Recognizer path | Time | Bytes/op | Allocs/op |
|---|---:|---:|---:|
| Initial per-byte traversal scratch | 5.84 µs | 14,176 | 331 |
| Final public `ToolState.Advance` | 2.32 µs | 10,152 | 43 |
| Final reused generation scratch | 0.87 µs | 296 | 15 |

Reused scratch is private to a generation. Grammar programs and source states remain immutable. The earlier reused-scratch measurement was 408 bytes/15 allocations; removing unused state fields reduced the final figure to 296 bytes. These are not model-generation speedup claims.

```sh
go test ./model/needle -run '^$' -bench '^BenchmarkToolGrammar' \
  -benchmem -benchtime=300ms
```

## Repository and preservation gates

Final code gates passed:

- `GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s`: **114 packages passed, 54 without tests**.
- `go vet ./...` and `go build ./...`.
- Affected Needle/loader/CLI/SIMD tests with `GODEBUG=cpu.all=off` (CPU-feature-disabled fallback check, not an instruction-level claim that every assembly path is disabled).
- `GOOS=linux GOARCH=riscv64 CGO_ENABLED=0 go build ./...`: cross-compilation only.
- Final native ARM runs above, including the released-model check.

`make docs-check` also passed: diagram and model-layout checks, link-checker tests, **362 Markdown files with zero broken links**, and the docs package tests. `git diff --check` was clean before commit.

All **16 freeze-manifest entries**, both preserved evaluation binaries and **676 record hashes** matched their saved values. The evaluation remains frozen at 676/1,440. No GPU checks or use, service changes, evaluation resumption, tolerance changes or tool execution were performed.

## Remaining scope

This is not full JSON Schema, an agent runtime, server/UI tool integration, confidence calibration, a packed-only loader or global-context eviction. The existing Needle gaps remain, including engram-bearing width parity, Needle 2 extensions and full SIMD coverage. Wider model work remains queued after Needle; this milestone does not complete that queue.
