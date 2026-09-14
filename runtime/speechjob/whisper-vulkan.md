# Experimental resident Vulkan Whisper job stage

`NewVulkanWhisperWindowStage` binds the existing resident F32 `whisper.VulkanEncoder` to durable `asr-windows` journals and a host-only Go decoder. It is Linux/amd64-only, explicit and experimental. The normal `NewWhisperWindowStage` remains CPU-only and unchanged. There is no hidden CPU fallback, Vulkan initialisation, model load, listener or worker startup.

## Ownership

The caller supplies one immutable `Whisper` host decoder, tokenizer and already constructed resident encoder. `AllowExperimental:true` is mandatory. `BackendSHA256` attests the exact device/driver/shader/runtime/precision contract; this package cannot derive it. The stage version binds that identity, resident geometry/statistics and the complete host model/tokenizer/generation configuration. Host and device journals cannot resume each other. Changing backend identity rejects old checkpoints.

After successful construction the `VulkanWhisperStage` owns exclusive encoder use and teardown. Other copies of the owner share state. The caller must not use/close another reference, mutate model/tokenizer/backend state, run another global Vulkan client or close the global Vulkan context while the owner is live. The host decoder remains caller-owned and immutable. `Stage()` returns a copy whose closure keeps the owner reachable; keep the owner handle for `Status` and `Close`.

`Status` returns scalar running/draining/quarantined/stopping/closed state and a stable code. It is operational state, not progress, device utilisation or model-quality evidence.

## Cancellation and drain

Vulkan has one global submission lane and retains at most one pending submission. A timeout/cancellation can return `ErrVulkanInFlight`; dropping the stage then could release queue/resource/store ownership while native buffers remain in use. The owner therefore holds the stage callback until `vulkan.VulkanDrain` returns nil. It polls with a configured 1 ms–30 s per-call budget and a fresh background context. The cancelled request never cancels native ownership retention. Each poll is bounded, but total drain time is deliberately unbounded because safe teardown requires fence proof.

Only explicit context timeout/cancellation is retryable. A fatal helper/driver error is quarantined even if joined with `ErrVulkanInFlight`. Device loss or uncertain submit/wait state is quarantined immediately. Inference, drain or encoder-close panic is also quarantined. Quarantine keeps the job, HTTP/queue admission and model/native ownership until **process teardown**. It never marks native resources safe, falls back to CPU, resets the device, returns a job failure or automatically retries. This can intentionally leave shutdown blocked; the operator must terminate the isolated process after preserving durable media/checkpoints.

A normal drain returns the original inference/cancellation error. Durable complete-window acknowledgements remain, the failed window is absent and explicit retry starts at the verified prefix. Queue claims still require explicit retry after restart. Device process death does not itself certify output correctness; the next process loads a fresh checked backend.

## Close

`Close(ctx)` first stops new jobs, waits for active inference/drain and closes the resident encoder. A timeout leaves ownership intact. Ordinary encoder close errors are retryable because the encoder's close contract retains failed resources. A close panic is quarantined. Once closed, repeated close is a no-op even with an expired context. `Close` does not close the host decoder/global device or initialise/drain unrelated Vulkan resources.

## Limits

The existing stage window/file/4-hour limits and resourcebudget reservation apply. Resident and transient Vulkan bytes must be included in the operator's declared estimates. `VulkanEncoderStats` describe requested owned arenas, not actual device RSS/heap high-water. There is no current available-memory admission, device-memory budget, multi-device scheduler, general device-loss recovery or external LLM coordination.

The resident encoder currently covers the Whisper encoder only. The decoder remains Go/CPU. Quantised/F16 encoder paths, Vulkan decoder, Community-1 Vulkan graph, general device recovery and whole-job placement are unfinished.

## Verification

Model-free tests use the private orchestration seam with synthetic window inference and mocked drain/close callbacks; public construction only accepts the concrete `*whisper.VulkanEncoder`. Tests verify:

- queue cancellation retains resource admission and store/media ownership until drain;
- per-window journal resume after pending native work drains;
- host/device/backend checkpoint identity separation;
- close serialisation, retry and idempotence;
- context-independent drain and bounded polling;
- fatal/panicked drain and close quarantine;
- real child SIGKILL for pending, device-loss, uncertain, inference-panic and drain-failure quarantine, followed by explicit journal recovery.

Existing backend offline tests provide mocked Vulkan pending-buffer/fence/quarantine mechanics. Existing native/trained tests are not rerun without explicit GPU/model authorisation. These lifecycle tests do not establish device execution, numerical parity, trained quality, performance, power-loss durability or production safety.

## Refinement notes

1. Problem: job cancellation could release resources while a Vulkan submission remains in flight.
2. User: the explicit single-tenant speech worker using a resident encoder.
3. Success: store/admission/model ownership survives until fence proof or process teardown.
4. Minimum: explicit owner, device-keyed stage, drain/quarantine and close lifecycle.
5. Exclusions: GPU initialisation, CPU fallback, decoder/Community Vulkan, device recreation and deployment.
6. Scope: additive Whisper validation and Linux/amd64 speech-job stage/tests/docs.
7. Platform: existing Linux/amd64 resident encoder; other builds keep core queue/store support.
8. Failure: timeout polls; fatal/device/uncertain/panic quarantines; ordinary inference errors drain then fail.
9. Persistence: existing immutable window journals; no native lease persistence.
10. Pattern: Vulkan global pending-submission ownership plus Store.Run callback lifetime.
11. First change: mocked owner invariants before any native run.
12. Avoid: forced idle, inferred safe teardown, auto-retry/fallback and fabricated progress.
13. Naming: distinct backend-bound `asr-windows` version; stable status codes.
14. IO: checked resident encoder/host decoder/media checkpoint; same validated raw windows.
15. Compatibility: CPU stage/default/server unchanged; opt-in consent required.
16. Limits: existing file/windows/duration; per-poll 1 ms–30 s; total safe drain unbounded.
17. Trust: immutable caller assets/backend identity; process isolation for quarantine.
18. Export: same raw/final transcript allowlist after downstream stages; no device buffers.
19. Proof: mock/offline/repeated/process-kill tests; native/trained gates separately authorised.
20. Closure: scoped lifecycle commit/evidence; full Vulkan/Community/quality/performance remains open.
