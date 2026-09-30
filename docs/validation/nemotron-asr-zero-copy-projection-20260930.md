# Nemotron ASR scoped shared-memory projection

The experimental Vulkan subsampling projection supports CPU producers and consumers using the same native allocation as the shader. Native analytic, trained JFK parity and an isolated ASR-plus-diarization smoke passed on Intel Iris Xe after the user's 21:56 execution authorisation. Application-level per-chunk projection copies are zero; latency did not improve consistently. Production remains on the deployed CPU build `c9569c86`.

## Implementation

- [`NewVkSharedTensorArena`](../../backends/vulkan/vulkan_arena.go) requires `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`. Allocation fails if no eligible memory type fits the existing budgets. Legacy arena selection is unchanged.
- [`WithCPURead` and `WithCPUWrite`](../../backends/vulkan/vulkan_arena.go) expose a bounded F32 slice only during a synchronous callback. The Vulkan lane stays held through the callback, excluding transfer, submission and destruction. Invalid owners, ranges, alignment, pending work, uncertain submission and device loss reject access.
- Existing host-to-compute and compute-to-host [barriers](../../backends/vulkan/vulkan_barrier.go) apply. CPU reads require confirmed fence completion. Coherent memory needs no flush or invalidate calls.
- [`ProjectScoped`](../../model/nemotronasr/subsampling_projector_device.go) runs producer, projection, then consumer. The CPU subsampler flattens its last convolution output directly into mapped input. The CPU encoder consumes mapped output before reuse. Immutable projection weights and bias upload once during preparation.
- [`SubsamplingStream`](../../model/nemotronasr/subsampling_stream.go) commits convolution caches only after successful consumption and cancellation checking. Existing owned-output methods copy the shared result before returning. The compatibility `Project` method intentionally copies caller input and returns owned output.
- [`PCMGenerationStream`](../../model/nemotronasr/pcm_generation_stream.go) uses the scoped consumer boundary for the tower. Failure closes the request as before. CPU, copied Vulkan and PTX remain supported; shared projection requires explicit selection.
- Failed device cleanup retains arena/operator references so an explicit drain followed by another `Close` can release them.

Callbacks must not retain the slice, mutate read-only input, use it asynchronously, or reenter any Vulkan API. Go cannot enforce those caller obligations. Callback errors, cancellation and panics release the lease but do not roll back writes. The projector clears and fully rewrites input before each dispatch. Request objects are single-stream and must not be used concurrently. Holding the global lane through CPU tower consumption can delay unrelated Vulkan work.

## Experimental selection

The CLI accepts the following option without changing any default or server profile:

```sh
nemotron -task asr -backend vulkan -vulkan-shared-projection \
  -input recording.wav -model /path/to/model.safetensors \
  -tokenizer /path/to/tokenizer.json
```

This command uses a GPU and requires device clearance. Shared mode rejects unsupported configurations or uncached memory; it has no silent copied/CPU fallback.

The candidate server also accepts `nemotron_asr.projection` with `backend: "vulkan-shared"`, `allow_experimental: true`, a nonempty `device_contains` and pinned `backend_sha256`. Omission preserves CPU stage/profile identities. Shared execution has a separate stage identity, per-request native resources, device matching before inference and cleanup before checkpoint publication. Diarization remains CPU-only.

## Verification

Base: `a8e9b48f11273b2d59e1e41422e5dab4fa89a9b8`. Go `1.26.2`, Linux amd64, `GOMAXPROCS=2`, NVIDIA disabled, `GO_PHERENCE_VULKAN_CPU=0`. Ordinary tests load no models and probe no devices.

| Check | Result |
|---|---|
| Full-tree CPU `go test`, `go vet`, `go build` | Pass |
| Affected packages, ten repetitions | Pass |
| Full-tree `go test -race -p=2 ./... -count=1 -timeout=180s` | Pass, explicit exit 0 |
| `make model-layout-check` | Pass in isolated toolchain container |
| Linux arm64/riscv64 Vulkan and ASR test-binary cross-builds | Pass; no foreign execution |
| `git diff --check` | Pass |
| Scoped CPU lease steady-state allocation assertion | 0 allocations/call |
| Native Vulkan analytic gate | Pass, three runs on Iris Xe |
| Trained CPU/copied/shared JFK parity and cancellation/fresh retry | Pass |
| Isolated shared-ASR + CPU-diarization JFK job | Pass, four verified artifacts |
| New server opt-in validation and legacy CPU checkpoint identity | Pass |
| Full-tree CPU tests/vet/build, full-tree race and layout after server wiring | Pass |

The host lacks `make` and a C compiler and defaults to `CGO_ENABLED=0`. Layout and race checks therefore ran in an existing Go 1.26.2 Bookworm image, ID `962e4695d843cf55fac32114d1e26e465eb9ef1efb9713f4f29dd72e57809e1d`, with no network, no passed GPU devices, a read-only source mount, two CPU quota, 3 GiB memory and no swap. Earlier shell waits expired while their containers continued; the final run completed within its wait budget and wrote `layout_exit=0` and `race_exit=0` before exiting. All validation containers exited.

Model-free [backend tests](../../backends/vulkan/vulkan_cpu_access_test.go) verify pointer identity, nonzero offsets, exact slice bounds, prefix views, surrounding canaries, cached-memory selection/rejection, callback error/panic/cancellation, concurrent close exclusion, invalid owners, quarantine and barrier/fence gating. The shared mock uses Go-backed storage and mocked Vulkan calls; it establishes API/lifetime behaviour, not driver behaviour.

Model-free [ASR tests](../../model/nemotronasr/subsampling_shared_test.go) use a synthetic three-convolution subsampler and an independent single-nonzero-column projection oracle. They verify CPU/shared parity across three chunks, direct consumer storage identity, owned-output stability, input preservation, malformed/nonfinite output rejection, failure rollback and preservation of an acknowledged prefix.

Changed public CPU access and stream wrappers have 100% statement coverage; `withCPU` has 95%. `ProjectScoped` has 38.7% coverage because its real Vulkan setup/dispatch/consume branches require device execution. Whole affected-package coverage is Vulkan 87.8%, ASR 34.9%, CLI 25.9%; these figures include existing model/driver-gated code. Coverage does not measure shader execution.

The opt-in [native analytic gate](../../backends/vulkan/vulkan_cpu_access_native_test.go) requires `GO_PHERENCE_TEST_VULKAN_CPU_ACCESS=1` and a nonempty `GO_PHERENCE_VULKAN_DEVICE`. It checks shared input → identity-matrix projection plus bias → shared output across reuse, copied/mapped read agreement and guards. It passed three native runs. The trained gate in `model/nemotronasr/shared_projection_native_test.go` hashes the model before loading and records application transfer counters, repeated timings, stage parity, cancellation and fresh-request reuse. Independent upstream trained parity and listening-based task quality remain unverified. Two earlier read-only review delegates timed out without producing findings.

## Transfer scope and performance

At the existing four-row seam, each projection uses 69,632 input bytes and 16,384 output bytes. Scoped generation removes the per-chunk upload/download copies and the owned flatten/projection buffers at this boundary. Weight/bias preparation still copies. Convolution, encoder intermediates and returned transcript/token outputs still allocate.

For 35 JFK projections, native application counters report copied input 2,437,120 bytes and copied output 573,440 bytes, totalling 3,010,560 bytes (2.87 MiB). Shared mode reports zero input/output copy bytes, 35 CPU-write borrows and 35 CPU-read borrows. Both modes upload 17,829,888 weight/bias bytes once per request. Counters do not describe driver-internal movement.

Five alternating-order JFK whole-request samples, excluding model/WAV load and including projection setup/teardown:

| Backend | Median | Mean | Range |
|---|---:|---:|---:|
| CPU | 3.6117 s | 3.7845 s | 3.5293–4.5602 s |
| Copied Vulkan | 3.7725 s | 3.8241 s | 3.7407–4.0757 s |
| Shared Vulkan | 3.8215 s | 3.9109 s | 3.7669–4.1589 s |

Shared versus copied shader/subsampling and tower values matched exactly. Against CPU, subsampling max/mean absolute error was 0.001708984375 / 0.0000375458; tower error was 0.000000834465 / 0.0000000107770. Existing 3e-3+4e-5-relative subsampling and 3e-4+2e-5-relative tower gates passed without changes. All token decisions and frame positions matched. Cancellation after a completed chunk and a fresh shared request passed; native memory counters returned to baseline after close.

The isolated candidate on loopback port 18094 completed an English JFK ASR-plus-CPU-diarization job in 7.097 seconds for 11 seconds of audio. All four exports passed size/hash verification and plain ASR cue/text comparison. The full Decoder run is separate and is not covered by this smoke. The projection covers one matrix multiplication; a resident FFN block or encoder tower needs separate implementation and measurement.

Native environment: Intel Iris Xe RPL-P (`8086:a7a0`), i915, Mesa 26.1.5, Vulkan device API 1.4.354; container image `73955bdf70a7b14e89100610502f01d2e4dc231697fd17e785275f538cad6204`, Intel ICD only, only `/dev/dri/renderD128` passed, CPU4/8GiB/swap0, NVIDIA disabled, live-Qwen-idle and host-available-memory guards. Analytic tests used CPU2/1GiB. Evidence is under `tmp/nemotron-zc-native-20260930/`, including analytic/trained logs, `trained-results.json`, smoke metadata/exports and explicit validation exit records.

The live transcription PID `2755577` and Qwen PID `2756230` remain unchanged with zero restarts. Transcription still has `PrivateDevices=yes`. No service, deployed binary, model, profile, GPU access setting or job was changed.
