# Gemma4 vLLM-leverage programme

Cumulative status as of 2026-08-07. The measurement target is the official Gemma4 E4B QAT Q4_0 GGUF at revision `4b4a2c1d`, SHA-256 `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`. CPU measurements use an Intel i7-12700 with AVX2; the bounded GPU graph experiment used an NVIDIA RTX 3060 12 GiB. Individual reports contain commands, workload details, and measurement limitations.

| Phase | Baseline → candidate | Primary result | Memory/parity gate | Disposition |
|---|---|---|---|---|
| Resumable session contract | legacy generation → `c1896760`, `2a7b0875` | request-owned prefill/decode/checkpoint lifecycle | exact first/two-token and restore fixtures | retain SIMD CPU session; scalar/NVIDIA stateful ownership pending |
| Streaming server | serialized generation → `78ea24b1` | immediate SSE token emission and cancellation between steps | deterministic release; legacy path unchanged | retain; shared model execution remains serialized |
| Serving benchmark harness | ad-hoc timing → `018e7f63` | fixed and Gamma/Poisson arrivals with TTFT/ITL/TPOT/E2E/goodput | bounded queue/cancellation fixtures | retain harness; live workload matrix pending |
| Prompt cache | recompute → `351c3295`, `ecb1188e`, `fb78a141` | full hit 17.8–20.6s → 0.40–0.46ms; partial block-2 20.6s → 5.16s | exact logits, race/eviction; 2.32–5.07 MiB cache | retain, block size configurable |
| Decode-first scheduler | monolithic prefill → `36f959c4`, `54a0ba93`, `a93cb862`, `94b64b41` | quantum-1 short TTFT 27.28s → 19.82s; total +5.7% | exact KV/positions and fairness | retain quantum 1 as latency control; reject 2/4 |
| CUDA Graph segment | eager kernels → `ba2042d9` | RTX 3060 replay 9.94–10.01µs vs eager 12.50–13.42µs; capture 13.14–14.05µs | exact fixed-shape output and teardown | retain prototype only; reject full capture while host interactions remain |
| Static decode batch | sequential tail → `ef8298d1`, `c6c98914` | B2/B4/B8 total +18.7/+12.7/+7.0%; B1 −7.6% | exact independent sessions; ITL 6.3–21.1s | experimental only; reject serving promotion |
| AVX2/Q4 projection reuse | repeated GEMV → `bfbe45a8`, `9738c125` | B8 medians improve at E4B widths 512/2048/2560/10240; allocations ~128–133 → 24–25 | exact dequant oracle, race, fallback, arm64/riscv64 builds | retain B8; reject B1/B2/B4 |
| Q6_K LM head | repeated GEMV → `7c2e1a2d` | real B8 decode 9.798s → 8.905s | exact logits/tokens; B2/B4 regress | retain B8 only |
| MTP verifier tests | invalid K=V post-RoPE oracle → `af2302f7` | broad model blocker removed | expected K now includes K-only RoPE | retain |
| KV utilization | uninstrumented → `06e08df2`, `c159b9b9` | 1/2/4/8 resident sessions scale exactly to 0.344/0.688/1.376/2.753 MiB reserved | per-layer used/reserved/slack; restore/race gates | paging not justified; retain linear KV and snapshots |

## Rejected or blocked work

- Full Gemma4 NVIDIA stateful batching is blocked because `GPUModel` owns one mutable KV/cache set; implementing it without request-owned device KV would violate session isolation.
- Full CUDA Graph capture is blocked by host interaction, allocations, CPU fallbacks, KV shadow copies, and logits download.
- CPU paged KV, prefix pages, recompute preemption, disaggregated execution, external KV services, and distributed orchestration are rejected until local measurements show a material memory or concurrency bottleneck.
- Static batching remains non-default because aggregate gains do not compensate for multi-second per-request ITL.
- `go test ./...` is not a clean programme gate because unrelated SpacemiT host-build/API failures, DiffusionGemma command API drift, and `backends/spacemit/inference` Q4_K tolerance failure remain. Focused model/loader, race, and cross-build gates are recorded per candidate.

## Detailed reports

- [Baseline asset](gemma4-baseline-assets.json)
- [Prompt-cache measurements](gemma4-prompt-cache-results.md)
- [Scheduler interference](gemma4-scheduler-interference.md)
- [Static decode batching](gemma4-static-decode-batch.md)
- [AVX2 hot-path, projection, LM-head, and KV audit](gemma4-avx2-hotpath-audit.md)
