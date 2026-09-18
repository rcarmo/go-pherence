# First native Vulkan speech-operator qualification

The F32 GELU, linear projection, LayerNorm and fused attention operators passed native execution on Intel Iris Xe (RPL-P). A six-stage resident plan matched six separately fenced GPU dispatches bit-for-bit. This qualifies the tested synthetic fixtures on this device/driver; no trained model or complete encoder ran.

## Authorisation and machine state

Rui authorised compute runs in this chat after the earlier offline-only checkpoints, then authorised stopping the LLM server. Run timestamps are retained in the raw event logs.

The first small native run occurred while the idle CPU-only LLM remained resident. The later qualification and timing runs used the explicitly stopped `llama-gemma-local-provider.service`. `systemctl --user stop` succeeded, both supervisor/server processes exited, and port 18092 closed. Model files and saved runtime state were not changed. `whisper-stt.service` and `whisper-stt-diarizer.service` stayed inactive. None of these services was restarted.

Device selection forced `/usr/share/vulkan/icd.d/intel_icd.x86_64.json` through both `VK_DRIVER_FILES` and `VK_ICD_FILENAMES`. The test also required an integrated/discrete physical GPU and an `Iris` device-name match. Hardware properties: vendor `0x8086`, device `0xa7a0`, driver version `109056008`; device name `Intel(R) Iris(R) Xe Graphics (RPL-P)`. The driver library and ICD hashes are in the resource snapshots/evidence. No validation-layer package was present; this was not a validation-layer run.

All final snapshots show the LLM and speech services inactive/dead, MainPID zero. Available RAM was about 28 GB. Swap was already allocated before this work; `pswpin=36292` and `pswpout=635627` did not change across the final three repeats. Resource snapshots are before/after measurements, not continuous telemetry.

## Numerical results

Final native test command: three repetitions in one bounded test process. All 24 test/subtest pass events completed with zero failures/skips. The totals below count repeated fixtures separately; they are not unique shape counts.

| Operator | Numerical cases | Compared values | Maximum absolute error |
|---|---:|---:|---:|
| Erf-form GELU | 6 | 24,636 | 3.873717941971222e-7 |
| Tiled linear | 30 | 6,336 | 9.660682177781155e-6 |
| LayerNorm | 66 | 426,636 | 2.298833243763454e-7 |
| Attention, including stress | 45 | 40,878 | 1.69888056578138e-6 |
| Six-stage plans | 18 | 885,654 | 5.364418029785156e-7 |
| Total | 165 | 1,384,140 | — |

Budgets were fixed before execution: GELU `2e-6 + 2e-6*abs(reference)`; linear/LayerNorm/ordinary attention `2e-5 + 2e-5*abs(reference)`; large-logit attention stress `2e-4 + 2e-5*abs(reference)`; the six-stage plan `5e-5 + 5e-5*abs(reference)`. No budget was relaxed after a device result.

References use float64 erfc, dot products, centred population variance, and full-score attention. The composed CPU reference rounds each materialised intermediate to F32. Outputs must be finite. GELU and LayerNorm run in-place and out-of-place. Fixtures cover odd tails, single keys, head widths through 64, 32 heads, a single sequence dimension at 4096, LayerNorm width 16384, linear K=16384 and changing/tied large attention logits. The full 4096×4096 attention workload did not run.

The plan is linear → LayerNorm → GELU → attention → linear → residual add. It is a synthetic six-operator graph, not a Whisper transformer block. The residual uses the existing vector-add shader through a manually checked generic stage; a public checked residual wrapper and convolution stem have not been implemented here.

Guard tensors surround cases, including attention stress after review. Allocation count and live allocated bytes return to their starting values after every operator group. Repeated plan outputs match independently fenced GPU dispatches bit-for-bit, exercising the actual inter-stage barriers, shared descriptors, in-place operations and later residual store. These checks do not prove all possible races or lifetime/error paths.

## Warm synthetic latency

Each repeat measured six alternating ABBA/BAAB blocks: 12 samples per path, 24 per repeat, 72 overall. Both paths execute identical shaders and tensor shapes, and every timed output must be bit-exact against the separately fenced GPU reference. There are two explicit timing warm-ups per path after prior correctness calls.

Shape: 256 rows, 384 channels, six attention heads of width 64. Weights and inputs stay resident. Constructor, shader compilation, allocation, upload, download, CPU reference and guard checks are outside the timed region. Timings include Go dispatch setup, submission and fence waiting; they are host-wall durations, not GPU timestamps.

| Repeat | One plan median | Six dispatches median | Separate / plan |
|---|---:|---:|---:|
| 1 | 6.623 ms | 9.970 ms | 1.505× |
| 2 | 8.300 ms | 12.002 ms | 1.446× |
| 3 | 8.264 ms | 11.856 ms | 1.435× |

The repeat medians vary. Pooling all 36 samples per path gives diagnostic medians of 8.263 ms and 11.823 ms, but does not replace the paired repeat results. No CPU-versus-GPU, cold-start, whole-model, realtime-factor or energy claim is supported by this comparison.

## Harness and review

`TestVulkanNativeSpeech` is disabled unless `GO_PHERENCE_TEST_VULKAN_SPEECH=1`. Once enabled, missing Vulkan, a non-GPU physical device, a mismatched requested name, constructor/dispatch/cleanup errors and numerical failures are failures rather than skips. It requires a Go test deadline no more than three minutes away; every dispatch also has a five-second context. A native driver call that cannot honour context remains bounded by the test process deadline. The harness does not retry, reinitialise or continue to later operator groups after a failed subtest.

The read-only harness review checked cleanup order, comparators and equal-work timing. It identified incomplete deadline coverage, weak renderer-name-only gating and missing stress guards. The final harness requires a whole-test deadline, queries physical-device type/IDs and adds the guards. This review is not independent GPU qualification. The final three-repeat run and the make target passed after these changes.

The first-run and intermediate three-repeat logs are retained separately. The intermediate log emitted long timing JSON lines that Go split into output chunks; final sample/summary lines are bounded separately for reliable extraction. Final source/evidence hashes are in [evidence.json](evidence.json).

## Reproduction

Run only within a coordinated compute window, with no competing LLM/speech work. Select the installed GPU ICD explicitly; the path below is machine-specific.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-native-check

# Optional modest synthetic host-wall timing, with three repeats:
GO_PHERENCE_TEST_VULKAN_SPEECH=1 GO_PHERENCE_TEST_VULKAN_TIMING=1 \
GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 CGO_ENABLED=0 \
go test -p=1 -count=3 -timeout=120s -json ./backends/vulkan \
  -run '^TestVulkanNativeSpeech$'
```

The opt-in-disabled test skips before initialising Vulkan. Existing mock selection still passes 115 top-level tests/423 events; speech/affine/media regressions, affected vet/builds and arm64 test cross-build pass. Full-tree/backend compile-only failures match the prior checkpoint. The race build cannot run because `gcc` is absent. No shader or runtime implementation changed in this checkpoint.

Hardware qualification remains incomplete for other devices, invalid/nonfinite contents, full Whisper/Community-1, cancellation during native execution, device loss/recovery, quantised formats and production workloads. Strict SincNet still has four failures. Convolution/residual wrappers, encoder wiring, trained quality and whole-job performance are unfinished. Nothing was pushed, deployed or restarted.
