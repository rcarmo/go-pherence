# MoJev PTX shared ancestors

The NVIDIA scorer now computes each question's common state/question prefix
once and forks candidate histories inside a bounded GPU tree. It follows the
[CPU SIMD tree pass](mojev-simd-tree-20260925.md), published as `f8a8a81a` with
CI `36200758236` passing.

## Kernel semantics and storage

Four tree-aware kernels complement the retained separate-branch kernels:

- Convolution walks parent indices, so its four-tap history excludes siblings.
- Linear recurrence snapshots the post-question state in per-thread registers
  and restores it at each candidate boundary. No state survives the launch.
- Q/K normalisation uses branch-local RoPE positions from tree metadata.
- Full attention reduces over ancestors followed by the current candidate,
  skipping every sibling row and preserving separate-branch reduction order.

The other matrix, norm, activation and residual kernels are unchanged. One
question's rows are projected together. Trees use the existing scorer capacity,
3–512 tokens. If any tree in a request exceeds scratch capacity, the request
uses the retained separate-branch executor; every individual branch must fit.
Questions still execute independently.

Each token has four validated int32 metadata fields: parent, local position,
node start and visible end. The root parent is -1. The metadata builder validates
all boundaries before writing output or uploading. At capacity 256 the scorer
adds 4096 device bytes and 4096 host bytes; reported GPU residency is
2,024,996,096 bytes. No full attention mask or per-candidate device-state cache
is allocated. Inputs, scratch and borrowed hidden rows remain protected by the
scorer mutex; public results remain owned.

## Paired full-request benchmark

Host: i7-12700, Linux amd64, RTX 3060, driver 580.173.02, Go 1.26.3,
`GOMAXPROCS=6`. Checkpoint revision
`0c8695b6252f4205907433d4e196a94f032e60c3`; safetensors SHA-256
`eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`.
Only existing approved assets were used. Separate processes ran sequentially,
with one warmup and five samples each. Parsing, tokenisation, inference and
response marshalling are timed; preparation, checkpoint hashes and HTTP are
excluded. The saved `4b91f4a8` GPU executable is the separate-branch baseline;
`f8a8a81a` changed only CPU tree execution.

| Request | Separate ms | GPU tree ms | Allocations separate → tree | Bytes separate → tree |
|---|---:|---:|---:|---:|
| Two choices | 90.23 | 46.09 | 697 → 663 | 54,000 → 36,808 |
| Eight choices | 366.32 | 47.83 | 1,262 → 1,011 | 170,184 → 48,960 |
| Two questions | 178.07 | 89.36 | 1,100 → 1,032 | 97,528 → 63,384 |
| Longer context | 185.07 | 93.44 | 1,639 → 1,610 | 94,080 → 71,360 |

Values are medians; latency falls 49–87%. An earlier tree run measured
46.32/48.08/89.52/92.87 ms. All four saved public responses match the
separate-branch baseline exactly. No statistical significance or held-out
accuracy result is inferred from the benchmark.

Peak process RSS was 6,130,492 KiB before and 6,128,436 KiB after. That small
difference is not a memory-admission improvement. Weights and host loading still
dominate residency. Post-change allocation attribution is led by tokenisation;
no further allocation hotspot was hidden in a per-candidate state cache.

## Verification

- Eight independent F32 cases preserve exact isolation and original gates.
  Maximum logit error is `4.7087669e-5` against `3e-4`; hidden error is
  `7.8797340e-4` against `2e-3`.
- Full-model GPU tree-versus-separate logits match exactly for 2, 8 and 64
  unequal-length candidates, reverse permutations and over-capacity fallback.
- Direct convolution, delta recurrence, Q/K RoPE and attention tree kernels
  match physically compacted independent branches exactly, including unequal
  lengths and repeated execution.
- Metadata tests cover exact records, invalid boundaries without mutation,
  512 tokens/64 candidates, and zero warm allocations; `fillGPUTree` coverage
  is 100% in that scoped test.
- Released-model race and whole-tree CPU race pass (exit 0). Four concurrent
  callers, repeatability, ownership and close/use tests remain enabled.
- Compute-sanitizer memcheck reports zero errors; racecheck reports zero hazards
  for both tree and retained direct kernels.
- Generated PTX reproduces byte-for-byte with `scripts/generate-mojev-ptx.sh`
  (CUDA 12.8, compute_86, `--fmad=false`).
- Vet, build and ARM64/RISC-V cross-builds pass. GPU execution is qualified here
  only on the RTX 3060; cross-builds do not establish foreign hardware behaviour.
- A focused independent review of tree kernels, metadata and capacity found no
  issue in parent traversal, recurrence forks, local positions or visibility.

Raw executables, samples, allocation profile and logs are in
`/workspace/tmp/mojev-gpu-tree-20260925`. The test selectors are
`TestMoJevPTXTreeKernels`, `TestFillGPUTree*`, and the existing opt-in
`TestMoJevAcceleratedReleased` with `GO_PHERENCE_MOJEV_NVIDIA=1` and the
hash-checked checkpoint path.

CPU scheduling, maximum-context/concurrency admission, cancellation, broader
changed-code coverage and held-out task calibration/quality remain open.
Nsight Systems kernel attribution is still unavailable on this host.
`RuntimeReady` remains false.
