# Owned resident F32 Whisper encoder

`NewVulkanEncoder(ctx, source, frames)` now builds the complete Whisper encoder graph from an existing F32 `Encoder`. Reduced complete graphs passed native Iris Xe boundary/final comparisons and cancellation/reuse checks. This is an explicit API; existing model defaults, decoder calls and transcription paths are unchanged. No trained checkpoint ran.

## Ownership and execution

The caller explicitly initialises Vulkan before constructing the encoder. Construction validates configuration, all required tensor lengths and finite weight values before allocating native resources. A missing K-bias is represented by an uploaded zero vector. Source weights must not change during construction; after return the Vulkan encoder owns independent device copies, and the source may be released or changed.

The API fixes the input frame count at construction. Input is channel-first mel `[melBins,frames]`; output is time-major `[ceil(frames/2),width]`. The full positional table is checked first, then only the required prefix is uploaded. This includes odd input lengths. Configuration limits are even maximum length 2–4096, mel bins/model width 1–2048, FFN width 1–16384, 1–32 layers/heads, head width 1–64 and matching total head width. These are admission bounds, not whole-envelope performance or accuracy qualification.

Weights occupy a stem/final arena and one arena per layer, avoiding a single oversized descriptor backing. Scratch is shared across layers: mel, stem, hidden state, normalised values, Q/K/V, projection output and FFN hidden output. Arena sizes account for the queried device offset alignment and checked host integer/range bounds.

Plans are private and created once:

- Stem: CF convolution stride1 → GELU → TM convolution stride2 → GELU → position addition (5 stages).
- Each transformer layer: pre-attention norm → Q/K/V projections → attention → output projection → residual → pre-MLP norm → FC1 → GELU → FC2 → residual (12 stages).
- Final normalisation (1 stage).

Each plan has its own submission/fence. `Forward` uploads mel once, keeps intermediate activations resident, checks cancellation between plans, and downloads the final hidden state once. It scans input/output for nonfinite values. Finite inputs/weights can still overflow intermediate arithmetic; final nonfinite output is rejected. No hidden scalar fallback occurs.

Copies of the public handle share one serial admission gate. `Close` blocks future `Forward` calls, releases plans/arenas/operators in reverse order and retains failed resources for retry. A pending GPU submission may require explicit `VulkanDrain`. Constructor failure normally returns nil after rollback; if rollback itself fails, it returns a nonnil stopping handle with the joined error so the caller can retry `Close`. No restart or automatic drain is performed by the API.

## Tests and native results

Offline layout/preflight/ownership checks pass three top-level tests and 19 test/subtest events. Thirty shuffled repeats pass 570 events. They cover tensor topology/read-before-write, odd-frame shapes, zero K-bias, geometry/length/nonfinite rejection before device access, 19 metadata cancellation checkpoints, arena overflow, shared-copy close, partial close/retry and admission cancellation. Existing Vulkan mock selection remains 125 top-level tests/434 events.

Native tests use three reduced configurations and both odd/even frame counts:

| Mel bins | Frames | Width | Layers | Heads × head width | FFN width | Plans / stages |
|---:|---|---:|---:|---|---:|---|
| 3 | 17, 18 | 8 | 2 | 2 × 4 | 13 | 4 / 30 |
| 80 | 33, 34 | 32 | 2 | 2 × 16 | 63 | 4 / 30 |
| 128 | 65, 66 | 64 | 4 | 1 × 64 | 128 | 6 / 54 |

Three final native repeats pass 24 test/subtest events, with no failures/skips: 138 boundary/final comparisons, 139,920 values, maximum absolute error `1.1920928955078125e-6`. The budget was fixed before execution at `1e-4 + 1e-4*abs(reference)`; it was not changed after results.

The reference uses direct scalar convolution/projection and the existing scalar LayerNorm, with an independent full-score float64 attention routine and erf-form GELU. Boundary checks download after each plan for diagnosis, separately from normal `Forward`. Every graph boundary is compared, including both residual paths in each complete layer through their end-of-layer result. Each scalar intermediate is materialised as F32. No accelerated CPU/model dispatch is used by this reference.

Native ownership/lifecycle checks also pass:

- Three repeated forwards have bit-exact output, as do later forwards after all source weight slices/configuration are overwritten.
- Invalid mel lengths, NaNs and pre-cancelled contexts reject without corrupting later inference.
- Nine deterministic cancellation cases across three repeats occur at Err-checkpoints 18, 36 and 69 of 73. All cancel and then reproduce the original output bit-for-bit. At checkpoint 18, all three runs retained a native submission; explicit bounded drain succeeded before reuse. These are fault checkpoints, not measured cancellation latency or device-loss recovery.
- A forced second arena-allocation failure rolls back all earlier allocations. Live allocation count and bytes return to baseline after every shape and rollback test.
- Closing a copied handle invalidates the original; repeated close succeeds.

The first native run and one standalone make-target run passed separately and are not included in the three-repeat totals. The native test requires explicit opt-in, an expected device-name match and a Go test deadline of at most three minutes. Runs used the Intel ICD and Iris Xe RPL-P from the prior operator qualification; there were no competing LLM/speech services.

## Review and remaining limits

Read-only review found no scoped scratch reuse, weight ownership, constructor rollback or serial close/forward issue under the checked Vulkan primitive contracts. Private plans prevent callers from mutating generic stages after operator admission. The review did not execute GPU work.

Implementation checks initially caught a uint64/int constructor argument and a nil test context; the native test also initially named a nonexistent assertion helper. These were corrected before final runs. The full Whisper arm64 test cross-build fails on pre-existing missing `simdfft.PrecomputeHannWindow` and `PrecomputeMelFilters`; hiding all four new encoder files through a Go overlay reproduces identical errors. The Vulkan package alone cross-builds for arm64. Full-tree/backend compile errors match the earlier baseline; race compilation still lacks `gcc`.

Focused speech/affine/media/Vulkan regressions, affected vet and amd64 builds pass. The LLM and both speech services remain inactive with MainPID zero. Before/after swap counters are unchanged (`pswpin=36292`, `pswpout=635627`). Resource records are snapshots, not continuous telemetry.

No trained model, full-size encoder, quantised weights, decoder/cross-KV integration, ASR quality, device-loss recovery or whole-job speed claim is established. Strict SincNet retains four failures. This checkpoint adds no shader/runtime default change, media backend change or deployment.

## Reproduction and evidence

```sh
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-encoder-check

# Only in an authorised compute window, selecting the installed GPU driver:
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-encoder-native-check
```

[Evidence](evidence.json), numerical metrics, native/offline logs, resource snapshots and baseline errors are included. Nothing was pushed, deployed or restarted.
