# Explicit Go Whisper job server

`speechjobserve` is an opt-in Linux/amd64 command that loads one local checked Whisper model and serves the existing authenticated job API. It does not download assets or install/restart services. It executes recovered pending tickets at startup only when `queue.enable` and `queue.start_worker` are both explicitly true. The default serving profile is **CPU Whisper ASR**, with FFmpeg decoding and conservative exact-overlap transcript/VTT output. An explicitly consented experimental resident F32 Vulkan encoder can replace only the encoder; decoding and the Whisper token decoder remain Go/CPU. Community-1 serving and trained/performance qualification are separate work. An opt-in [browser interface](../../../runtime/speechjob/httpapi/ui/README.md) is available with `http.enable_ui:true`, an exact matching `http.origin`, and at least two ordinary request slots; it is disabled by default.

## Launch requirements

Use an absolute configuration path and set the process environment **before launching**:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 ./speechjobserve --config /absolute/server.json --check
# Only after reviewing the limits and authorising model loading:
GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 ./speechjobserve --config /absolute/server.json
```

Supply `SPEECHJOB_TOKEN` through a secret manager/environment, never a command argument or config file. It must contain 32–256 printable non-space ASCII bytes with operator-generated cryptographic entropy. Serving also requires `allow_execution:true`; the shipped example is false and contains nonworking paths/hash placeholders. `--help` is available on other platforms, but model loading/checking is explicitly unsupported outside Linux/amd64.

Whisper has legacy package-init backend probes and environment-derived modes. `GO_PHERENCE_DISABLE_NVIDIA=1` must already be set at process start, even for `--check` or help on Linux/amd64; setting it later cannot undo driver initialisation. `GOMAXPROCS` must equal configured `threads` (1–16). Nonzero `WHISPER_*`, `GO_PHERENCE_WHISPER_*` and `GO_PHERENCE_A100_NOINIT` switches are rejected by this initial profile. This prevents accidental experimental/quantised/GPU path selection; it does not establish host-wide CPU exclusivity.

All asset paths must be absolute, regular, non-symlink files under administrator-controlled immutable directories. Each model/config/tokenizer/generation/FFmpeg/ffprobe asset has an explicit SHA256. Paths, binaries, shared libraries and loaded-model files must remain immutable throughout startup/jobs. Hash-then-reopen is not protection against malicious same-UID path replacement or concurrent mutation. The runtime SHA256 is an operator attestation to the executable/build/ISA/runtime environment, not an automatically derived reproducible-build proof.

## Configuration

See [example.json](example.json). The server configuration uses exact lowercase JSON keys, rejects unknown/duplicate/case-aliased keys, null values and excessive nesting, and is capped at 64 KiB. Missing fields are accepted only where their zero value passes explicit validation (for example, zero overlap, false exact-silence skip and zero generation-owned timestamp override). Assets are operator-provided; no model/path or generation options come from HTTP clients.

The single profile fixes language, media extension, duration, input/output caps, window overlap and generation settings. The tokenizer must include the complete multilingual vocabulary and match the pinned checked generation document. Generation owns the initial timestamp policy, so `max_initial_timestamp_index` must be zero in the profile. `max_new_tokens` can only shorten its limit; zero uses the checked generation default. Exact-digital-silence skip is optional and false in the example. It is not VAD/no-speech detection.

`--check` validates config, TLS files if configured, asset hashes, tokenizer/generation/model metadata, supported dtypes/shapes/extents and configured byte-admission estimates. It maps the safetensors file for metadata inspection and hashes its complete bytes, but does **not** call the tensor payload loader, create/open/recover a store, bind a listener or infer. It reports `metadata_checked:true, model_loaded:false, listening:false`. This is not full model-schema/numerical qualification: required tensor-name/shape binding and finite-value checks happen in the checked loader during execution startup.

Execution validates token and TLS first, then opens/recover-locks the private store before loading model weights. A second process cannot load a duplicate model against that same store. The store parent must already exist; store recovery may mark interrupted jobs failed. A separately enabled queue worker executes pending tickets only after loading succeeds and the listener binds; interrupted claims require explicit retry. Invalid model startup after acquiring the store may therefore have performed normal recovery before failing. Each load failure closes the store, and the listener is bound only after checked loading and all four stages are constructed:

`decode → asr-windows → transcript → vtt`

The checked loader owns widened finite weights and closes the safetensors source after binding. The profile version incorporates pinned assets, complete generation/settings and runtime identity. HTTP clients select the configured profile ID; it remains the sole source of model options. Conflicting ASR overlaps fail conservatively with raw windows retained; this server does not add general alignment or promote neural quality.

## Durable queue configuration

Omitting `queue` retains synchronous request-owned execution. To accept durable intents without executing them, add:

```json
"queue": {
  "enable": true,
  "start_worker": false,
  "directory": "/absolute/private-queue",
  "max_entries": 32,
  "max_bytes": 1048576,
  "job_seconds": 3600
}
```

Review pending tickets, profiles and host admission before explicitly setting `start_worker:true` and starting the process. There is no HTTP worker-start endpoint. `allow_execution:true` is still required to serve/load the model, even with the worker off. `--check` never opens queue/store metadata or starts a worker.

Queue storage must be private, outside the media store, under an existing administrator-owned immutable parent. Entries are capped at 1–128, bytes at 256 KiB–4 MiB and job timeout at 1–28,800 seconds. Disabled queue options must be omitted or zero. Queue mode rejects `http.enable_ui:true` and synchronous `/run`; use the [CLI queue commands](../speechjob/README.md). Interrupted running claims require explicit retry, while recovered pending intents can run after worker startup. Upload never implies enqueue. See [queue recovery](../../../runtime/speechjob/queue.md).

Without `resources`, the command uses one process-local `SerialAdmission` callback. The optional declared budget below provides weighted accounting within this process. Neither mode shares resource budgets with other processes, LLMs or stores. Queue job timeout includes admission wait and is independent of the HTTP request deadline. SIGINT/SIGTERM drains admission, callbacks and release before closing queue/store resources.

## Resource limits

- Store: up to 1,000 retained jobs, 512 MiB upload, 1 GiB artifact and 64 GiB total file caps (actual configured values can be smaller). Store size includes orphaned scratch/journals/manifests.
- Model input: at most 8 GiB pinned safetensors file, 4 MiB header, 2,048 metadata entries and rank 1–4 F32/F16/BF16 tensors. The preflight reserves conservatively twice the sum of widened F32 tensor bytes against `owned_weight_bytes` (at most 16 GiB) before materialisation. Tokenizer is capped at 32 MiB, model JSON at 1 MiB and generation JSON at 16 KiB.
- Network: explicit literal-IP listen address and port, 2–128 accepted connections, 1–32 ordinary handler requests plus its reserved cancellation slot. Connections must be at least ordinary requests plus one. An OS backlog is still kernel-owned; a connected slow client can consume a slot until its deadline. This is not transport-level cancellation priority or per-client rate limiting.
- Deadlines: 1–30 second header timeout no greater than request seconds; read/write timeouts and handler context 1 second–8 hours; idle timeout 1–300 seconds. Header/socket phases and handler computation have distinct timer origins. Context checks are cooperative: a synchronous kernel/syscall may finish after its deadline.

The widened-weight estimate is a loading-allocation admission check, **not RSS enforcement**. Source mapping, JSON/tokenizer maps, Go GC behaviour, model activations, HTTP buffers and filesystem caches add memory. Use OS resource limits and external shared-compute admission for real models; one store lock cannot coordinate unrelated stores, LLMs or GPUs. New byte buffers use bounded sizes, but this slice does not claim measured allocation or speed improvements. Allocation/escape profiling remains separate coordinated work; no blanket pool or unbounded cache was added.

## Optional declared CPU/memory budget

Omit `resources` to preserve existing behaviour. An explicit configuration can add:

```json
"resources": {
  "cpu_slots": 2,
  "memory_bytes": 6442450944,
  "max_waiting": 32,
  "load_bytes": 5368709120,
  "resident_bytes": 4294967296,
  "work_bytes": 2147483648
}
```

These numbers illustrate the field relationship; they are **not qualified model peak measurements or RSS limits**. The operator must supply estimates for the actual model/runtime/worker. Capacity CPU slots must cover configured `threads`. Loading bytes must cover resident bytes and fit capacity; resident bytes must cover the existing `limits.owned_weight_bytes` cap. Work bytes must be positive and fit alongside resident bytes. `max_waiting` is 0–128. CPU slots are declared capacity, not affinity or CPU throttling.

After the store lock and before `buildProfile`, the server acquires `threads` CPU slots and `load_bytes`. Once loading succeeds it shrinks that lease to zero CPU and `resident_bytes`, retaining it through server drain. Each queue or synchronous run acquires `threads` plus `work_bytes` from the same private budget, holding the lease until callbacks finish. Startup failures release their accounting; RSS need not fall immediately after Go objects become unreachable. `--check` validates relationships but does not create this budget or start execution.

The private server budget has two active leases (resident and one job). Embedders can share a [Budget instance](../../../runtime/resourcebudget/README.md) across multiple handler/store owners, but the standalone command cannot control unrelated processes. There is no current available-memory probe, external LLM resource protocol, cgroup/RSS enforcement or device-memory accounting. Keep external host admission and OS limits when running real weights. Disabled/default execution has not been promoted to these illustrative values.

## Experimental resident Vulkan encoder

The CPU profile remains the default. Vulkan requires `resources` plus these profile fields:

```json
"vulkan": {
  "enable": true,
  "allow_experimental": true,
  "device_contains": "Intel(R) Iris(R) Xe",
  "backend_sha256": "REPLACE_WITH_DEVICE_DRIVER_SHADER_RUNTIME_SHA256",
  "encoder_weights": "q8-kv-mlp",
  "drain_milliseconds": 10
}
```

The original five values remain mandatory. `encoder_weights` is optional: absent or `"f32"` preserves the existing F32 encoder and checkpoint identity; only `"q8-kv-mlp"` selects the qualified K/V/FC1/FC2 Q8 placement. Full Q8 is deliberately unsupported because a retained 63-second gate shifted one timestamp. Device substring is checked against the selected Vulkan device after explicit `VulkanInit`. Backend SHA256 is an operator attestation and part of the job/profile checkpoint identity. The selected Q8 precision mode is additionally bound into that identity, so F32 and Q8 work cannot resume one another. Check mode validates config/assets/metadata but never initialises Vulkan. Execution takes the store and loading-resource lease before initialising the device or constructing the resident encoder.

Resident construction copies encoder weights into owned Vulkan arenas. The server then clears the redundant host encoder tensor references; this makes Go storage eligible for GC but does not force RSS reduction. The host decoder remains resident. Resolved decode/ASR/transcript/VTT stage versions are embedded in the HTTP profile identity, so CPU/device, F32/Q8, or changed backend versions cannot share a configuration.

Handler/queue callbacks drain before resident encoder close, store close and resource release. A pending, uncertain, device-lost or panicked Vulkan owner can intentionally prevent graceful process exit until the operator terminates the isolated process; no in-process device reset exists. Startup failures after construction retry owner teardown with bounded sleeps rather than returning while native ownership is unresolved. No CPU fallback or automatic retry is provided. Read the [Vulkan job-owner contract](../../../runtime/speechjob/whisper-vulkan.md).

This configuration has only model-free/mock server-lifetime coverage. It has not been run with a native GPU through this server, trained checkpoints or private recordings. It does not add Vulkan decoder/Community-1 execution or quantify resident/device memory.

## Experimental Community-1 profile

Community-1 is omitted by default. [`example-community.json`](example-community.json) is a nonworking placeholder with `allow_execution:false`. An explicit `profile.community` object can add the pure-Go CPU diarization graph and separate speaker-labelled outputs. It requires:

- `enable:true` and `allow_experimental:true`;
- pinned model revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`;
- absolute SHA256-pinned segmentation, lowered-filter, embedding, xvector-transform and PLDA assets;
- complete segmentation LSTM/head, embedding and PLDA geometry;
- explicit PCM window/step/speaker/clustering policy;
- explicit `scalar` or `simd` segmentation modes, `scalar`/`simd`/`gemm` embedding mode and `reject` or diagnostic `lowest-index` tie policy;
- a configured process-local resource budget and result-byte cap.

Execution verifies all asset bytes under the loading lease, uses the checked safetensors and numeric-only NPZ loaders, closes sources before serving and transfers the graph to the resident Community owner. Configuration identity includes model revision, xvector hash, geometry, prefix, asset/runtime hashes and execution policy. No model defaults are inferred from tensor sizes.

The combined stage order is:

`decode → asr-windows → transcript → vtt → diarization → speaker-transcript → speaker-vtt`

Plain transcript/VTT commits before experimental diarization. A diarization failure therefore leaves plain text downloadable. Speaker outputs remain separately marked experimental and require conservative complete-cue coverage. Shutdown drains handlers/queue and closes owners in reverse construction order before store/resource release.

`--check` verifies complete hashes, safetensors inventories and NPZ signatures, but does not construct Community models or execute PCM. This profile has model-free reduced-container server tests only. Strict trained SincNet/embedding and ambiguous-tie failures remain unchanged; no trained quality, long-file scaling, Community Vulkan graph or production deployment is claimed.

The server JSON uses lowercase snake-case fields at every level. Test/benchmark manifests that use Go field names are not accepted as server configuration. See [Community owner](../../../runtime/speechjob/community1-owner.md).

## TLS and shutdown

TLS certificate/key files are capped at 1 MiB each. Startup verifies the pair, current validity period and hostname coverage for every Host allowlist entry. It serves TLS 1.2+ and HTTP/1.1; HTTP/2 multiplexing is not enabled. Client trust/chain verification still depends on configured client/system roots. Plain HTTP is permitted only with `allow_loopback_http:true` and a literal loopback listener. Host/Origin policies, bearer auth and transcript-only downloads use the existing [HTTP contract](../../../runtime/speechjob/httpapi/README.md). The process does not trust forwarding headers or install network ACLs.

SIGINT/SIGTERM closes the listener and HTTP connections, cancels request contexts, and waits for handler-owned work. If `shutdown_seconds` expires, it reports that resources remain retained, then waits for cooperative drain before closing the store/models. The grace timeout is **not a hard process-exit deadline**. Arbitrary stuck callbacks cannot be safely abandoned; an operator can forcibly terminate the process, but that is process death, not graceful shutdown. Only after drain can the store lock and model closure ownership be released. Startup/status errors avoid logging tokens, model/store paths or callback bodies.

## Verification and current status

Run `make speech-job-serve-check` for synthetic tests. The explicit media target `make speech-job-serve-integration` generates a tiny two-channel zero-layer Whisper model plus WAV fixture and runs real FFmpeg through the four-stage profile. No trained weights or private recordings are used. Listener tests use temporary local ports, TLS test certificates and fixture callbacks; a real child process receives SIGTERM, exits cleanly and releases listener/store ownership. Checks cover config/hash/budget rejection, tokenizer case/null behaviour, suffix profile composition, TLS connection state, idle handshake deadlines, connection caps and grace-timeout retention.

The initial synthetic generation fixture omitted most language IDs and was rejected; it was corrected to the complete vocabulary without loosening parser gates. Existing production services stay stopped. The command is implemented and locally tested; it has not been deployed, exposed to the LAN or qualified with trained checkpoints. Core cross-builds are compilation only; model execution remains Linux/amd64-only. General numerical/tie/WER/DER, performance, long-recording diarization and host-wide resource enforcement/external LLM coordination remain open. Process-local weighted estimated admission and the durable queue are implemented and synthetically tested; its current browser integration is unsupported. Browser checks use separate model-free fixtures and do not qualify trained inference or production deployment.
