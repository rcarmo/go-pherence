## Repository audit: package review complete, remediation still open

All **164 host-listed Go packages** now have a recorded source-boundary review.
There are **zero inventory-only packages** in the [coverage matrix][coverage].
That closes the package-review gap from the first pass; it does not certify every
function, assembly instruction, device backend or deployment. The [open-findings
ledger][findings] and issue [#16][issue] retain the work that needs further fixes,
native dependencies or authorised hardware tests.

The final pass covered the remaining 64 packages in groups: command entry points,
speaker/audio loading and service ownership, PTX signatures and index arithmetic,
SpacemiT dispatch/fallbacks, and test/schema-only packages. Completed read-only
delegated reviews supplied selected sections for the image/small commands, model
diagnostics, SpacemiT commands, PTX and RVV/AICPU groups. Two larger command reviews
timed out; they add no credit. Their selected entry/setup/owner boundaries were
read directly. None of those reviews executed a model or hardware probe.

## Errors must not turn into successful validation

The Gemma MTP parity CLI previously returned `matched: true` for a consistent
trimmed fixture when either model asset was missing. It now requires both models
for normal execution. Explicit `-fixture-only` checks report
`fixture_validated: true`, `executed: false`, `matched: false`; the portable Make
recipe selects that mode. Expected logit probes cannot disappear because a row
or token ID is missing, and nonfinite values produce a JSON-safe failure.
Historical real-model results were not rerun or reinterpreted.

Qwen's MTP metadata inspector now uses the shared fail-closed resolver. Absent
default weights permit metadata-only inspection; a present corrupt index, dangling
index or missing shard does not fall back to a stale monolithic file. Layer
metadata enumeration is bounded to4,096 and output labels the legacy loadability
fields as metadata/name checks, not proof of loading or execution.

GGUF expectation-only flags now request their static/runtime plans, and runtime
plan errors no longer silently bypass validation. Speculative golden files are
limited to16MiB and require exactly one JSON document. The LLM MTP diagnostic checks
cache element and byte products before allocating. Empty modelcoverage family
checklists are rejected rather than becoming100% completion. Filtered category
semantics and some CLI expectation combinations remain in the findings ledger.

## The other DiffusionGemma server needed admission too

The earlier audit fixed `diffusiongemmaserver`; the separate older
`diffusiongemmaserve` entry point still decoded unbounded request bodies and queued
callers behind a blocking model mutex. It now requires one1MiB JSON document,
limits message count and requested output, checks prepared token count, and returns
429 rather than accumulating a model queue. HTTP header/read/write/idle deadlines
are configured. The request owns the model lock through response handling.

Malformed/oversized/busy handler tests use nil model state and therefore prove
rejection before model access. They do not exercise native denoising. Execution is
still monolithic and not interrupted by a disconnected client; authentication,
aggregate process admission and image-placeholder expansion remain deployment
and follow-up work. An output cap is not a preemptible kernel.

## Diagnostic ownership and code generation

VAE smoke loading now runs inside an error-returning scope that closes its
safetensors file on decoder failure. It rejects latent grids outside1..64 before
opening weights, checks empty output and propagates output errors instead of
panicking. Tests use missing/incomplete synthetic checkpoints, not VAE inference.

The SpacemiT ORT probe generator passes its output directory as a Python argument
instead of interpolating a path into executable source. Nonpositive iteration and
thread values are rejected. The test checks argument construction without running
Python, ONNX or ORT. Fixed-file overwrite behaviour and other native diagnostic
limits remain explicitly listed rather than folded into a broad safety claim.

## PTX review found host/kernel contract mismatches

Whisper's full attention kernel stops without writing for head dimensions above128
or KV sequences above2,048; the online variant also requires head dimension at
most128. Its host wrappers did not enforce those limits. Host preflight now checks
those shapes, grid limits, finite scale, packed float extents and u32 linearised
products. Convolution output geometry/buffers and the mel kernel's fixed512-point
DFT, thread and input extents are checked too. Tests use fake pointers and pure
validators, never initialise CUDA, and establish no device numerical parity.

The wider PTX review found unresolved u32-product risks in other batched kernels
and non-atomic duplicate-position Q8 scatter updates. Those are source findings,
not observed device faults. SpacemiT review likewise found global/pool TCM lifetime
mismatches, success after skipped work, incomplete extent/tail checks and runtime
flags selecting unsupported C-shim stubs. Native execution is unavailable here;
none is hidden behind a green stub test.

## What the audio and speaker review established

The speech-job client has explicit endpoint/token policy, no redirects or proxies,
bounded responses and authenticated artifact size/hash checks before private,
no-clobber publication. The service bounds config/assets, owns startup resources,
and drains or quarantines uncertain native work rather than freeing it early.
Those are selected source contracts plus existing host tests, not a new network
penetration test or device qualification.

Legacy audio CLIs and the simpler speaker package do not share all those bounds.
Whole-file audio/ffmpeg work, nonfinite timestamp conversion, predictable temporary
paths, common-prefix scoring and quadratic clustering remain follow-ups. The
Community-1 loaders copy checked weights; PCM geometry is bounded and the Vulkan
composition serialises ownership. Its experimental numerical/physical-device gates
remain unresolved. The ledger separates these paths so the stronger service
contract is not attributed to every older command.

## Final checks and preservation

The final committed-code NVIDIA-disabled race sweep exits zero: **109 packages
pass and55 have no tests**. Earlier pass logs are retained separately rather than
being substituted for the stable sweep after the last source fix. Whole-tree
vet/build, Linux ARM64/RISC-V builds,22 Bun tests and the338-document link/layout
check pass; foreign builds are compilation only. The coverage table matches `go list ./...` exactly, with164
unique package rows and no inventory-only remainder.
Local evidence remains under `/workspace/tmp/go-pherence-repo-audit/`.

Both evaluation executables, all16 freeze-manifest entries and all676 result hashes
are verified without inspecting predictions. No completed row was replayed, no
calibration or frozen tokenizer changed, and no GPU recovery or service restart
was performed. The approved UI architecture is Bun packaging with Go-embedded
assets, but UI implementation is separate from this audit. The held-out evaluation
remains blocked at676/1,440.

Package review is complete. Remediation and exhaustive semantic review remain
open in the ledger and issue#16. Unavailable native validation is now filed once
per platform/backend: [NVIDIA/CUDA #17](https://github.com/rcarmo/go-pherence/issues/17),
[K3/RISC-V #18](https://github.com/rcarmo/go-pherence/issues/18),
[Vulkan #19](https://github.com/rcarmo/go-pherence/issues/19),
[ARM64/NEON #20](https://github.com/rcarmo/go-pherence/issues/20) and
[GGML/CGo #21](https://github.com/rcarmo/go-pherence/issues/21). None of those gates
is marked passed merely because a tracking issue exists.

[coverage]: repository-safety-audit-coverage-20260919.md
[findings]: repository-safety-audit-open-findings-20260920.md
[issue]: https://github.com/rcarmo/go-pherence/issues/16
