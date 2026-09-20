## Needle cached decoding and native ARM64 validation

Needle's FP32/CQ reference models, gradient tests, decoded archives and incremental decoder now pass on the CIX P1 CD8160 at `orangepi6plus.local`. Cross-compilation alone had missed several existing NEON bugs. The first native run failed; the passing results below include the fixes, with no tolerance increases or scalar substitution.

## What failed and why

The ARM64 `SgemmNT` kernel reduced its four vector lanes into `F20`, then accumulated scalar tails and stored from `F0` instead. That discarded most of each dot product and affected Needle attention and tied-output projections. Both the tail and store now use `F20`.

BF16 vector addition, narrowing and BF16 RMSNorm encoded XTN for 64-to-32-bit lanes where the code required 32-to-16-bit lanes. The instruction words now match the intended element widths. Weighted RMSNorm and unweighted RMSNorm scalar tails also used `F4` after vector input/weight loads had overwritten `V4`; those tails now use the preserved inverse RMS in `F6`.

`neon_regression_test.go` covers all NT reduction widths 1..65 with padded strides and output canaries, weighted RMSNorm lengths 1..65, and BF16 conversion/addition lengths 1..33. Existing tests exposed the bugs; these additional cases keep the vector/tail transitions covered.

## Native runs

Cross-built Linux ARM64, CGo-disabled test executables and their tiny fixtures were copied to `/tmp/go-pherence-needle-ad7f4b14` on the board. Runs used `GOMAXPROCS=2`, `nice -n 10` and 120-second test timeouts. Existing host jobs were left running; no model downloads, GPU commands or service changes were made.

| Test executable | Final result | Passing tests/subtests counted in verbose log |
|---|---|---:|
| `model/needle` | PASS | 63 |
| `loader/needle` | PASS | 45 |
| `cmd/needle` | PASS | 5 |
| `backends/simd/runtime` | PASS | 523 |

The SIMD source-boundary tests initially failed because the test-only bundle did not contain repository source. The final run includes the actual source directories those tests inspect and `go.mod`; they were not skipped or replaced with empty fixture directories.

Needle checks include both generation-specific FP32 references, all-gradient parity, Needle 3 CQ and AB scale derivatives, auxiliary heads, nested depth slicing, training/checkpoint reload, archive/BPE parity, and cached-versus-full-prefix logits. Tests use the same reference values and tolerances as amd64. These binaries were not race-instrumented; the race suite runs on the Intel host.

The transferred test bundle SHA-256 is `c105deb8d6fb6af351a08a4a243fbde814b29caee06e508431dc3510936b7fd4`; the source-boundary bundle is `573ebaa307d22b1fcc53ee9d71f8d44a4a7e21abbf9443a9b671318736f0f547`. Local verbose logs are retained under `/workspace/tmp/needle-port/` as `model-needle-final.log`, `loader-needle-final.log`, `cmd-needle-final.log` and `backends-simd-runtime-final.log`.

## Cached decoding on Intel

The decoder keeps K/V, raw QKV convolution taps and engram value history. CQ/AB parameters are prepared once, local attention uses fixed rings, and state is committed only after a complete successful token step. Cancellation, invalid input and workspace failures leave the previous position and contents intact. `Reset` clears prompt and cache buffers.

On the Intel i7-12700, the tiny twelve-token Needle 3 fixture took 0.577--0.598 ms with a reused decoder versus 2.408--2.426 ms for repeated full-prefix forwards, about 4x faster. Sequence allocations fell from 4.39 MB / 72,221 allocations to 0.923 MB / 21,445 allocations. These are short microbenchmarks, not production-model throughput. The board was busy with other jobs, so there is no ARM speed claim from this run.

## Repository checks and remaining work

After the assembly fixes, the NVIDIA-disabled whole-tree race suite passed for 114 packages; 54 had no tests. Host vet/build and Linux ARM64/RISC-V cross-builds passed. Needle tests also passed on the Intel host with AVX2/FMA disabled.

Native ARM correctness is established for these bounded fixtures and kernel tests, not every backend or a full-size model. Direct packed-weight kernels, production-sized throughput/memory work, global KV eviction, schema-constrained tool calling and the remaining Needle 2/3 features are still open. This does not close the broader ARM issue or the repository safety audit. Frozen evaluation work and GPU-service state are unchanged.
