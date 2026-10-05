## Pocket TTS flow-hybrid runner -- 1 October 2026

`ptx-flow-hybrid` and `vulkan-flow-hybrid` now have explicit Linux CLI factories and replay identities. Flow-head forward, time-JVP and reverse-over-forward derivatives use bounded device tapes; transformer/FlowLM, time embeddings and their scalar adjoints, weighting MLP, loss reductions, clipping, AdamW/EMA and checkpoints use CPU code. Native execution is held and has not run.

The parent is `a90aec30567d485784f2e04d1803c543c9f2817f`. All training changes are uncommitted; the held inference index is unchanged at eight paths / 469 added lines. No model, recording, GPU runtime, recovery or listening work was performed.

## Tape and objective

`model/pockettts/training_flow_tape.go` builds an owned graph of dual linear, SiLU, normalisation, modulation and vector operations. The pinned mixed-flow fixture uses 512 tensors, 423 commands and 34 parameter gradients for each selected time. The host interpreter matches output, JVP, condition/input/time adjoints and every parameter gradient. Tolerances remain 3e-6 for output, 5e-6 for JVP, 8e-6 for input adjoints and 1e-5 for parameter adjoints.

`training_lsd_backend.go` composes four tapes for minimal-LSD: primary forward/JVP, endpoint forward, endpoint input VJP and primary mixed backward. Endpoint direct parameter, condition and time gradients are discarded. The diagonal objective uses two tapes. `training_step.go` returns flow conditioning derivatives to CPU FlowLM backward and combines target derivatives with the same objective weights as the CPU path.

Pinned independent fixtures are `flow_head_backward_pytorch.json` and `training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`. Minimal-LSD tolerances remain 2e-6 for loss/raw square, 5e-6 for flow values, 2e-5 for scalar/input derivatives and 3e-5 for parameter gradients. The full-step test retains 3e-5 loss and 5e-5 gradient tolerances, including derivatives returned to FlowLM. These tests execute a host tape interpreter. Device output error distributions are unmeasured.

PTX allocates tensor buffers, uploads parameters/seeds and synchronises each operation. Vulkan allocates tensor arenas and executes ordered plan chunks of at most 64 stages. Reshape commands share storage. Intermediate tensors stay on device within a tape; parameters are uploaded again for each invocation. Logical tensor ceilings include graph views conservatively and exclude native metadata, driver overhead, host graph copies and process RSS. This is an allocation/transfer-heavy baseline with no performance qualification or persistent model residency.

## Admission, replay and failure

Both CLI scopes require `-allow-experimental-device-training`, a valid caller-declared `-device-runtime-sha256`, a positive byte ceiling no greater than 2 GiB, and sampler opt-in. Native factories follow local input admission. Full `ptx`/`vulkan` and combined attention-plus-flow requests are rejected. SIMD remains the default.

`ptx-flow-hybrid-v1` and `vulkan-flow-hybrid-v1` are separate replay identities. Flow options bind scope and runtime digest; SIMD and attention-hybrid v1 digest construction is unchanged. Runner construction, update, checkpoint and restore check the live scope. Cross-scope or changed options/runtime bundles cannot mutate a fresh runner.

A hybrid backward error poisons the runner before the host optimiser runs; retry and new checkpoint publication are rejected. Tests inject failures at the primary diagonal, primary LSD and final mixed calls, verify unchanged parameters/buffers/moments/EMA/raw replay state, and verify no retry. Executors retain uncertain tape storage and operators. A failed close returns a retained-resource error whose owner stays reachable. PTX and Vulkan executors check context cancellation before constructing each tape; no mid-kernel interruption or automatic recovery is implemented.

## Offline verification

Controls: Go 1.26.3 linux/amd64, local toolchain with `GOPROXY=off`, `GOSUMDB=off`, `GOENV=off`, empty `GOFLAGS`, `GOMAXPROCS=2`, nice 10, both `GO_PHERENCE_DISABLE_NVIDIA=1` and `GO_PHERENCE_DISABLE_VULKAN=1`, serial packages/tests, 45-second inner and 120-second outer deadlines. Test selectors exclude all native and released-asset tests.

The final gate ran **21:40:03.986411094--21:40:36.070517444 UTC** on 1 October 2026, exit 0. Eight new/affected tests repeated ten times gave 80 top-level and 80 subtest passes; race gave eight top-level and eight subtest passes. No failures or skips. Minimal-LSD and full-step host PyTorch parity each passed after the final topology-admission change. An earlier affected runner/run-state/bundle/attention/flow regression gate gave 29 top-level and 48 subtest passes, with no failures/skips. Focused vet, whole-tree host build and Linux ARM64/RISC-V CLI cross-builds passed; foreign binaries were not executed. The source-layout gate also passed (Go docs tests plus five Bun tests / 3,008 assertions), and whitespace checks passed.

Final scoped coverage: backend scope/options/notice/retained-error helpers, flow identity methods, close methods and tape-building adapter each reach 100%. Combined execution identity reaches 87.5%; native PTX/Vulkan tape execution wrappers reach 18.2%; CLI factory reaches 37.9%, attention factory 25.0%. Native successful construction, launch/transfer/close and corresponding failure branches are not exercised by this host-only gate. Selected package total is 33.2%; none of these percentages establishes native correctness. Missing coverage and native ownership validation remain open.

The first runner test build failed because the test used a nonexistent `runDigest` helper; it was replaced with an explicit fixture digest. Earlier flow-fixture key-prefix and graph-sort compilation failures are documented in prior archives. The independent read-only review timed out after 90 seconds without findings or approval.

Sources, commands/logs, coverage and cross-builds: `/workspace/exports/pocket-gpu-training-flow-hybrid-20261001/`. Earlier graph/minimal-LSD/full-step and native-mock ownership tests are in `/workspace/exports/pocket-gpu-training-flow-tape-20261001/`. Native device numerics, transfer-inclusive performance, released-size resource use, persistent residency and trained-voice listening require separate authorised work.
