## Jevlike validation record

The measurements below are the historical 2026-09-18 port checks. Later Qwen3
GPU/direct/head/prefix results and failed quality gates are in the
[current experiment status](../../docs/experiments/jevlike-qwen3/status-report-20260919.md).
The final evaluation remains blocked at 676/1,440 records; GPU safety candidates
have host tests, not recovered-device validation. The later [CPU profile](../../docs/performance/allocation-simd-audit-20260919.md)
measures tiny-scorer allocation hotspots without reopening the held-out study.

Host: Intel Core i7-12700, linux/amd64. Tests and measurements collected on 2026-09-18.

Passed: package and CLI tests, package/CLI vet, race tests and cgo-disabled tests. ARM64 and RISC-V test binaries cross-compile; this is not runtime validation on either architecture.

## Whole-workload timings

The order was native, CPU-features-disabled, CPU-features-disabled, native. Each command used `taskset -c 0`, `GOMAXPROCS=1`, `-benchtime=1s` and `-count=1`. The disabled runs used `GODEBUG=cpu.all=off`; this affects more than one kernel and is not a single-kernel attribution experiment.

| Workload | Native samples | Features-disabled samples |
| --- | --- | --- |
| Eight-example scoring, including byte batching and probabilities | 0.938, 1.109 ms | 3.305, 3.656 ms |
| One training epoch, 32 examples, including initialisation and validation | 187.763, 238.408 ms | 218.802, 231.893 ms |

Scoring consistently improves with native CPU dispatch in this initial sample. The initial training ranges overlap; the later profile-driven backward change below supplies a separate controlled comparison. The high allocation counts (4,287 per scoring batch and 112,978 per training epoch) leave room for buffer reuse before adding more assembly.

Raw outputs are in `testdata/benchmarks/native.txt` and `testdata/benchmarks/portable.txt`.

```sh
GOMAXPROCS=1 taskset -c 0 go test ./model/jevlike \
  -run '^$' -bench BenchmarkTiny -benchtime=1s -count=1
GODEBUG=cpu.all=off GOMAXPROCS=1 taskset -c 0 go test ./model/jevlike \
  -run '^$' -bench BenchmarkTiny -benchtime=1s -count=1
```

## Training backward dispatch

With upstream-compatible unit-normal embedding initialisation, the baseline CPU profile attributed 86.35% of samples to the scalar linear-backward accumulator. Its two independent updates per weight row now use `simd.Saxpy`, which dispatches to Plan 9 assembly where available without changing row accumulation order. SIMD reduction/FMA rounding is tolerance-tested, not claimed bit-identical.

Two retained test binaries were run in before/after/after/before order, pinned to core 0 with `GOMAXPROCS=1`, `-test.benchtime=2s` and `-test.count=1`. The benchmark includes initialisation, one 32-example epoch and validation at width/rank 64.

| Implementation | Samples | Two-sample median |
| --- | --- | --- |
| Scalar backward | 189.170, 187.148 ms | 188.159 ms |
| Plan 9 SAXPY backward | 46.108, 44.386 ms | 45.247 ms |

The median ratio is 4.16 times faster. Both versions allocate 31,509,952 bytes and 112,978 objects per epoch. The post-change profile attributes 36.89% flat CPU to `saxpyAsm`; backward accumulation accounts for 55.11% cumulatively rather than 86.88% before. Raw timings and profile summaries are `testdata/benchmarks/training-paired.txt`, `training-profile-before.txt` and `training-profile-after.txt`.

Tests cover vector/tail widths 1 through 127, multiple accumulated updates, slice guards, unchanged inputs, finite-difference gradients, PyTorch head-gradient parity and synthetic training convergence. Native, CPU-features-disabled, cgo-disabled and race test suites pass. These timings are for a bounded synthetic workload, not arbitrary frozen-backbone training throughput.

## Visual encoder

The complete width-4 RGB/motion encoder now matches a deterministic upstream PyTorch fixture: maximum patch-feature difference 6.56e-7. This covers preprocessing, convolution, GroupNorm, SiLU and positions; it is not parity for a complete trained game policy.

The width-32 visual workload includes preprocessing and all convolutions. Pinned single-core runs measured 3.719, 3.734 and 3.770 ms with native SGEMM, versus 13.189, 13.120 and 13.110 ms with CPU features disabled. Median ratio is about 3.5 times faster. Raw outputs are `testdata/benchmarks/vision-native.txt` and `vision-disabled.txt`.

## Frozen backbone and visual action scoring

The upstream-default Qwen2.5-0.5B checkpoint now has an opt-in full-model fixture, using float32 Transformers execution and a deterministic scorer head. Both context token sequences match exactly. Final hidden-state differences are at most 0.00166; scored option logits differ by at most 1.06e-5 across two examples with variable option counts and Unicode input. The test exposed missing Q/K/V biases in the native single-token forward path; both singleton options and batched prefill now apply those biases.

```sh
JEVLIKE_FROZEN_MODEL_DIR=/path/to/Qwen2.5-0.5B GOMAXPROCS=6 \
  go test ./model/jevlike -run PublishedFrozenQwen -v -count=1
```

The reusable visual action scorer also has a deterministic upstream fixture for two reads and per-example option selection, covering logits, value estimates, averaged query/key/value matrices, attention, probabilities and entropy. No game environment is needed.

## Remaining scope

The frozen-backbone fixture covers Qwen2.5-0.5B, not arbitrary Hugging Face architectures. Complete trained visual-policy parity and visual checkpoint interchange remain unverified; the reusable adapters accept named Go parameter arrays. These limits must not be described as arbitrary-model or whole-upstream-demo compatibility.
