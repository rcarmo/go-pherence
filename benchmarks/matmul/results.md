# Matmul optimisation results

## Baseline e05d04e0

Environment: Intel Core i7-12700 exposed as 6 KVM vCPUs, Go 1.26.3, Linux 6.8, NVIDIA RTX 3060 12GB with driver 580.173.02. CPU governor controls are not exposed. `perf` hardware counters are unavailable because `perf_event_paranoid=4`.

Kernel raw data is in `baseline-e05d04e0/`: 21 benchmark cases completed across `GOMAXPROCS=1,2,6`; SpacemiT was explicitly skipped because this host is amd64. Go CPU and memory profiles for dense M=227 prefill are in `profiles-e05d04e0/`.

### End-to-end MOSS JFK

All six runs produced SHA-256 `9a47c0f25721a7deddbdfb7efe651e2ee86f63a37776ca7319517d7bfed44928`.

| Path | Runs | Median wall | Range |
|---|---:|---:|---:|
| CPU/SIMD | 3 | 35.425s | 35.369-41.145s |
| Automatic NVIDIA PTX | 3 | 12.990s | 12.988-13.108s |

The first CPU run is a cold-cache outlier; it remains in the raw record. GPU median speedup over CPU median is 2.73x.

## Cumulative changes

| Family | Baseline commit | Candidate | Hardware | Median before | Median after | Speedup | Memory delta | Parity | Disposition | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| Baseline | e05d04e0 | - | i7-12700 / RTX 3060 | - | - | - | - | exact MOSS hash | retained | Frozen protocol and raw benchmark set. |
| Dense AVX2 NT 1x4 + K128/N64 blocks | e05d04e0 | 00b4b7f2 | i7-12700, 1 core | M=32: 1.568ms; M=227: 11.154ms | M=32: 1.035ms; M=227: 7.805ms | 1.52x; 1.43x | none | blocked-kernel tails pass; exact MOSS hash | retained | Decode M=1 remains on the original kernel; pre-block-tuning MOSS CPU median was 36.031s versus 35.425s baseline (host variance/no gain). |
| Dense AVX2 NN 2x32 | 00b4b7f2 | cc1f3207 | i7-12700, 1 core | M=32: 2.230ms; M=227: 15.876ms; Whisper-like: 4.084ms | M=32: 1.162ms; M=227: 8.748ms; Whisper-like: 2.669ms | 1.92x; 1.81x; 1.53x | none | NN parity/tails pass | retained | M=1 keeps the original row kernel; expansion/contraction shapes improve by roughly 1.1-1.8x despite host variance. |
| Shape-aware dense dispatcher | cc1f3207 | c49cdd5e | i7-12700, 6 cores | MOSS CPU median 35.425s | 27.136s single measured run | 1.31x | none | exact MOSS hash; focused BERT/Whisper/tensor/model gates | retained | Migrates contiguous linear NT and large NN call sites while leaving decode and incompatible strides serial. |
| BF16 x4 output rows | c49cdd5e | 91a03532 | i7-12700, 1 core | F32 activation K1024: 1.830us; BF16 activation: 2.031us | 0.147us; 0.152us | 12.4x; 13.4x | none | x4/tail parity within BF16 accumulation tolerance | retained | K4096 improves from 7.37-8.06us to about 0.58us; scalar fallbacks cross-build on arm64/riscv64. |
| GPTQ symmetric Q4 portable MxN | 91a03532 | bd200952 | i7-12700, 1 core | B4: 9.68ms; B8: 19.29ms | B4: 5.92ms; B8: 11.66ms | 1.64x; 1.65x | none | exact scalar parity including batch/output tails | retained | Shares packed-weight/group traversal across four activation rows; decode and asymmetric paths unchanged; RVV accelerator preserved. |
| MLX affine Q4 MxN | bd200952 | a57baa2d | i7-12700, 1 core | B4: 7.02ms; B8: 14.23ms | B4: 3.52ms; B8: 7.04ms | 1.99x; 2.02x | 0 B common case; heap only above 256 groups | exact repeated-GEMV parity and tails | retained | Four activation rows share nibble unpack and affine metadata; batch=1 and generic bits unchanged. |
| NVFP4 portable MxN | a57baa2d | fd7592a2 | i7-12700, 1 core | B4: 18.89ms; B8: 37.47ms | B4: 7.11ms; B8: 13.74ms | 2.66x; 2.73x | 0 B | exact bitwise repeated-row parity and tails | retained | Four activation rows share FP4/F8 scale decode; decode and riscv dequant-once path unchanged. |
| FP8 E4M3 x4 batch rows | fd7592a2 | 081402e5 | i7-12700, 1 core | B4: 1.84ms; B8: 3.68ms | B4: 1.69ms; B8: 2.09ms | 1.09x; 1.76x | Buf API 0 B; convenience API 6KiB | exact batch/dynamic parity | retained | LUT gather remains 6-7x faster than arithmetic scalar decode for single-row decode; bounded output threading starts at 1024 rows. |
| GGUF Q2_K/Q3_K backend batch x4 | 081402e5 | 31d25d59 | i7-12700, 1 core | Q2 repeated: 33.64ms; Q3: 41.02ms | Q2: 3.61ms; Q3: 4.48ms | 9.32x; 9.16x | 12KiB vs 96KiB repeated | dequant-row oracle parity | retained | Adds missing backend-owned formats. Faster dequant+x4 candidates for Q5/Q8/Q6 were not enabled because real-model golden tests require their existing quant-dot reduction order. |
| GGUF MTP batch integration | 31d25d59 | 4e766504 | i7-12700 | per-row GGUF calls | unified ProjectBatchF32To | format-dependent | backend scratch only | focused MTP GGUF row/batch parity | retained | Q/K/V/O/MLP verifier projections now use backend batch dispatch; unsupported formats retain per-row fallback; DiffusionGemma specialised paths unchanged. |
| F32 x4 output-row GEMV | 4e766504 | daaf0835 | i7-12700, 1 core | rows128/K1024: 6.10us; rows2048/K4096: 1.163ms | 4.08us; 0.973ms | 1.49x; 1.20x | none | scalar-oracle tolerance and tails | retained | Shape guard keeps rows<256 with K>2048 on legacy Sdot after a measured 3% regression at rows128/K4096. BF16 x4 already retained; packed quant x4 remains format-specific. |
| Whisper packed head/query batches | daaf0835 | 25b37a30 | i7-12700, 4 workers | seq375: 6.90ms; seq1500: 92.37ms | seq375/Q96: 4.97ms; seq1500/Q64: 73.21ms | 1.39x; 1.26x | seq375 +1.5MiB; seq1500 -21.1MiB | bit-exact synthetic attention; exact MOSS hash | retained | Packs all heads once and blocks queries without changing per-row softmax; MOSS CPU 28.554s single run, within host variance of dispatcher result. |
| Non-DiffusionGemma MLX MoE paired gate/up | 25b37a30 | 7dfa34cc | i7-12700, 2 workers | 0.930ms; 13 allocs | 0.937ms; 8 allocs | 0.99x | similar bytes, 5 fewer allocs | exact legacy MoE parity | retained for allocation reduction | Single-token decode has no cross-token expert batching opportunity; paired API shares Q4 group sums and reduces scheduling/scratch allocations without a throughput claim. |
| NVIDIA GPTQ Q4 warp-tiled GEMM | 7dfa34cc | 18d80763 | RTX 3060 | repeated GEMV B4: 311us; B8: 618us | tiled B4: 79.7us; B8: 150.4us | 3.90x; 4.11x | 584 B/launch vs 1.9-3.7KiB | live GPU parity maxDiff=0 | retained | Four warps compute four output columns per block with shuffle reductions and no shared-memory barriers. |
| NVIDIA F32 SGEMM reg2/skinny candidates | 18d80763 | cb1c1a7b | RTX 3060 | oracle live parity max 1.67e-6 | skinny candidate max error 0.217 at M1 | rejected | n/a | candidate parity failed | rejected | Candidate entries remain loadable for debugging, but dispatch stays on the verified 16x16 oracle. |
| NVIDIA FP8 direct warp-tiled GEMM | cb1c1a7b | 9fafe44a | RTX 3060 | one-output direct/shared reduction | four warp-owned outputs/block | structural gain | no F32 weight staging | live max error 2.89e-6 | retained | Existing fused QKV/SwiGLU/DiT helpers automatically consume the improved direct kernel; SGEMM staging remains opt-in. NVFP4 native candidate deferred to avoid duplicating unverified scale logic in this slice. |
| NVIDIA Whisper online-softmax attention | 9fafe44a | 8b91ba3d | RTX 3060 | shared scores seq375: 3.41ms; seq1500: 85.3ms | recompute online: 27.2ms; 424.9ms | 0.13x; 0.20x | removes ~8KiB shared score array/block | live parity passed | rejected | Three-pass score recomputation overwhelms memory savings; candidate remains separate and default shared-score kernel is unchanged. |
| RVV packed outer tails | 8b91ba3d | 21ad19b4 | CIX P1 | fast aligned baseline: I8 3.86s; F16 12.16s; W4 5.05s | aligned core unchanged | neutral | tail fallback only | scalar parity on M/N/K tails | retained | Adds safe arbitrary M/N tails around existing M4N32 kernels; no claim of aligned-shape speedup. |
| IME2 persistent pooled 4x4 | 21ad19b4 | no code change | CIX P1 | single 2048x4x2048: 20.8ms | pooled: 23.0ms | 0.90x | pool overhead 0.55-0.76ms | existing parity | rejected | Packing is only ~5us; pool dispatch and host tile stores do not amortise for one matmul. Keep single packed kernel and reserve pools for multi-GEMM batches. |
