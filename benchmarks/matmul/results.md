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
| Dense AVX2 NN 2x32 | 00b4b7f2 | working tree | i7-12700, 1 core | M=32: 2.230ms; M=227: 15.876ms; Whisper-like: 4.084ms | M=32: 1.162ms; M=227: 8.748ms; Whisper-like: 2.669ms | 1.92x; 1.81x; 1.53x | none | NN parity/tails pass | retained | M=1 keeps the original row kernel; expansion/contraction shapes improve by roughly 1.1-1.8x despite host variance. |
