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
| FP8 E4M3 x4 batch rows | fd7592a2 | working tree | i7-12700, 1 core | B4: 1.84ms; B8: 3.68ms | B4: 1.69ms; B8: 2.09ms | 1.09x; 1.76x | Buf API 0 B; convenience API 6KiB | exact batch/dynamic parity | retained | LUT gather remains 6-7x faster than arithmetic scalar decode for single-row decode; bounded output threading starts at 1024 rows. |
