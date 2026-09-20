## Repository-wide safety audit: coverage and findings

Latest: the [sixth-pass findings](repository-safety-sixth-pass-20260920.md) fix Qwen3-TTS planning/config checks, packed-Q4 native bounds, BF16 NaNs and compressed-KV storage checks. The [fifth pass](repository-safety-fifth-pass-20260920.md) covers failed GLiNER loads, helper/stream limits, ioctl descriptors and RoPE/audio bounds. Native ownership and source-coverage gaps remain open. Earlier [follow-on findings](repository-safety-followon-20260919.md) address HTTP admission, worker drain/join, copied mmap lifetime, NaN comparisons and graph/CPU-fallback bugs. The findings below describe the first pass; use the linked reports and coverage matrix for current dispositions.

The Qwen GPU failure exposed a context-boundary bug, but the owner requested an
audit of **the entire repository**, not just Qwen or CUDA. This report records
the first repository-wide pass and the fixes it produced. It does not claim that
all 2,313 Go source files or every assembly kernel have been exhaustively reviewed.
The inventory contains 161 Go packages and 240 tracked assembly/C/C++/header/
TypeScript/Python files. Generated assets and other repositories are not audit
inputs.

The final held-out evaluation remains stopped at 676 complete records. Its
original executable, frozen model/prompt/calibration files and results have not
been changed or rescored. GPU memory-safety tests are blocked by the lost RTX 3060;
the source fixes below are not proof of the Xid79 root cause.

## Coverage of this pass

| Area | Work performed | What it does not establish |
|---|---|---|
| All 161 Go packages | Repository-wide vet, host build and race-test sweep; ARM64/RISC-V compilation | Hardware execution, skipped-fixture coverage or line-by-line correctness |
| NVIDIA shared runtime, all model callers | Inventory of every raw CUDA API call; focused review of context, module/stream/event/graph lifetimes, JIT and global reset | Same-model concurrent use or capture transactions are not generally isolated |
| Shared model GPU loader | Thread pin/unpin and cleanup ownership inspection | Numerical correctness of every model's GPU path |
| Safetensors/GGUF/NPZ and audio loaders | mmap/borrowed-view, size/offset and parsing review; regression tests for concrete findings | Adversarial filesystem mutation or a complete fuzz audit of every format |
| Tensor/internal arithmetic and CPU/SIMD backends | Shape/overflow guard inspection, unsafe-boundary inventory and package/race checks | Assembly ABI and tail handling on every ISA |
| SpacemiT AICPU/IME pools | Dispatch/close/thread-affinity inspection; cross-build | K3 execution or safe concurrent shutdown, which remains a finding |
| Vulkan | Driver lane/init/rollback inspection; package tests | Hardware execution; synthetic bounds tests do not prove device correctness |
| Speech-job/runtime queues and schedulers | Shutdown/drain/cancellation ownership inspection plus race tests | Every external integration and failure mode |
| Other model families (BERT, speaker, Whisper, OmniVoice, GLiNER, diffusion/image, Qwen variants, etc.) | Global package tests and lifecycle/unsafe/allocation call-site inventory; selected ownership files read | A full semantic/numerical audit of every architecture |
| Commands/services | Process/HTTP/input/cancellation and serialisation boundary inventory, focused server review | Production security testing or exhaustive endpoint coverage |
| Scripts/build/docs | All Bun script tests, both Python script test suites, layout/docs checks and source inventory | Every operational script or external download/service tested end to end |

The [package coverage table][coverage] lists every host package, selected source
files and explicit inventory-only gaps. Independent reviews returned CUDA/loader
findings, including checkpoint fallback and capture-copy defects. Other delegated
slices timed out and add no independent coverage. One review's PCM32 scaling
warning was rejected: conversion of MaxInt32 to float32 already rounds to 2^31.
The native BF16 sm_86 target excludes sm_80 despite its broader comment; that
pre-existing compatibility gap remains open, not an untested target change.

## Reproduced defects and candidate fixes

### CUDA driver scopes beyond the original copy helpers

Stream/event creation and destruction, prefetch events, explicit-stream kernel
launches, PTX module loading/unloading and argmax-index download contained raw
context-dependent calls outside an OS-thread pin and driver lock. `DevCopyN`
also called asynchronous DtoD directly with no context scope. Module
initialisation via `sync.Once` prevents duplicate work but does not pin an OS
thread.

The patch routes these operations through a scoped thread/context lock, retains
host argument owners through synchronous foreign calls, serialises shared module
registration, and puts asynchronous copies on the active capture stream when
capture is in progress, including the CopyDtoD path used for KV appends. A missing
capture-copy API now fails rather than silently issuing an uncaptured copy. The
zero-buffer helper also rejects a count that would overflow its byte-size check.
It prevents duplicate native BF16 module creation and
serialises graph object launch/destroy and compiled-kernel launch/destroy, draining
submitted work before module destruction. Fake-driver tests check lock ownership,
thread identity across yields, concurrent module registration and stream choice.
An AST inventory test rejects newly introduced raw CUDA calls outside reviewed
wrapper functions; that is a review guard, not a formal lock proof.

Lazy argmax and diffusion-sampling function handles were retained after their
modules/context were destroyed. Shutdown now clears them before unload. This
prevents those helpers from returning a stale `CUfunction` after reinitialisation.
The general GPU model loader also had an unbalanced `LockOSThread`; it now releases
its pin on both success and error return. This is separate from long-lived
K3-affinity workers, whose exit may intentionally discard a modified OS thread.

### JIT identity, arguments and ownership

The compiled-kernel cache key encoded op kinds and buffer indices but omitted
constants and graph edges. Distinct expressions could therefore reuse the wrong
kernel. The key now includes the name, ABI, constants and input topology. Codegen
uses a private node copy instead of modifying caller-owned `RegName` fields.
Concurrent misses are serialised so a replaced cache entry cannot leak a module.
Destroyed entries are rebuilt rather than returned from cache.

`CompiledKernel.Launch` accepted extra buffers, which shifted the trailing `N`
argument in the foreign-call ABI. It now requires exactly the declared buffer
count. Validation rejects unsupported reductions/ops, wrong input arity,
non-topological references, duplicate nodes and graphs exceeding the emitted
register budget. These checks use synthetic graphs and do not change a trained
model or held-out prompt.

### Cross-model loaders

`loader/audio.ReadWAV` divided by channel/sample widths before validating them.
The new malformed-input tests reproduced division-by-zero panics. They also
showed truncated data being accepted and odd-length ancillary chunks breaking
alignment. The parser now checks read errors, RIFF/chunk bounds, supported format,
frame/byte rates, padding, data alignment and finite decoded samples. Buffer
allocation grows from bytes actually read rather than a forged declared length.
It keeps buffered little-endian decoding; it does not reintroduce the old
reflection-per-sample path.

Safetensors eager prefetch wrote an unsynchronised global sink, so independent
files loaded concurrently could race. The sink is now atomic. Sharded index names
previously allowed parent paths and symlink escapes. They are now restricted to
files whose resolved parent is the model directory, under the existing immutable
filesystem assumption. The first containment patch incorrectly canonicalised
relative paths through the symlinked workspace; repository integration tests
caught it. Absolute-then-symlink canonicalisation and a dedicated test fixed that
regression; the local DiffusionGemma integration suite subsequently passed.
Metadata resolution now accepts an explicit sharded index and only falls back to
the single file when the index itself is absent. A broken index or missing shard
must not silently select a different checkpoint.

GGUF semantic header errors leaked their open file until the Go finalizer ran:
the deferred cleanup checked a local error variable that stayed nil when a
validation branch constructed its return error. Cleanup now checks the returned
error. A Linux descriptor regression fails on the baseline for bad magic, version
and tensor count and passes with the fix; GC is disabled during that test so the
finalizer cannot hide the leak.

### Vulkan validation before dispatch

The failing wrapper test exposed inconsistent validation order. Wrappers now
validate shapes and buffer extents before optional pipeline initialisation, so
malformed requests get a geometry error even without Vulkan. Shader indices and
products must fit uint32, and ceil-dividing dispatch counts no longer wraps near
its upper bound. The offline valid-buffer RoPE fixture now uses distinct buffers,
as required by its existing no-alias contract. No failed test was simply skipped.

## Findings not silently treated as fixed

These are tracked in [issue #16][followup]:

* CUDA graph capture is still a process-global multi-call transaction. Per-call
  locking does not keep an unrelated caller's launch out of another caller's
  capture. Shared scratch lifetime and same-model Generate/Close ownership also
  need explicit owner contracts or stronger synchronisation. Global Shutdown
  requires all inference joined and encoders released; it is not safe to race
  against live inference merely because each driver call is locked.
* AICPU's shared fn/gen/done dispatch has no caller-serialisation protocol, and
  Close can unmap TCM without joining in-flight workers. IME channel/condition
  pools also lack a complete idempotent drain/join protocol; concurrent Run/Close
  or multiple submitters can race or panic. Changes need K3 tests, not fabricated
  scalar hardware substitutes.
* The legacy WAV API still materialises full decoded audio in RAM. Untrusted
  streaming callers need an explicit limit or the bounded media decoder.
  Safetensors raw slices remain borrowed from an mmap and must not survive Close;
  a path check cannot solve concurrent file replacement or borrowed-view lifetime.
* The DiffusionGemma HTTP generator decodes an unbounded request body and queues
  behind a model mutex; cancellation/aggregate work admission is not established
  by that mutex. The LLM server has a body read limit but still needs complete
  request/trailing-data and cancellation/queue review.
* Legacy model entry points still rely on caller-valid input and exclusive
  mutable model ownership. BERT attention-mask lengths, tensor lazy realisation,
  inspector error suppression and scheduler waiting-queue admission need deeper
  review. The coverage table names the inspected boundaries and the untouched
  packages instead of presenting all model tests as source-review coverage.

## Validation and limits

The baseline repository race sweep completed 89 passing packages, 71 packages
without tests and one failed Vulkan package. A later sweep caught the temporary
safetensors containment regression in DiffusionGemma. Those failed logs are
retained. After the loader and Vulkan corrections, the full NVIDIA-disabled host
race sweep passed: **90 packages passed, 71 had no test files**, with no reported
race. This includes locally available DiffusionGemma integration fixtures, not GPU
execution. The final publish sweep includes the GGUF follow-on regression and
also exited zero with the same package counts.

Reproduction (create `.gotmp` first):

```sh
GOTMPDIR=$PWD/.gotmp GO_PHERENCE_DISABLE_NVIDIA=1 \
  go test -race -p=2 ./... -count=1 -timeout=180s
```

The release and publish logs are retained with earlier failing runs; no held-out
scorer was invoked.

Focused NVIDIA/loader/model/CLI race tests pass, including 20 repetitions of the
fake-driver context/module checks. Repository vet and host builds pass. Linux
ARM64 and RISC-V source builds pass as compilation only. Bun script tests pass
20 tests/2,565 assertions across eight files; Python tests pass six label-mass and
two parquet-adapter tests. GPU opt-in tests were not forced to run on the missing
device; skip is not a pass of GPU behaviour.

Raw logs and package inventory are under `/workspace/tmp/go-pherence-repo-audit/`.
The fixes and open findings apply across model families. No weights were fetched,
no final-test accuracy was inspected, no GPU recovery attempted, and no other
repository was edited. The runtime candidate must still pass approved GPU
sanitizer/parity/lifetime testing before replacing the pinned evaluation binary.

[followup]: https://github.com/rcarmo/go-pherence/issues/16
[coverage]: repository-safety-audit-coverage-20260919.md
