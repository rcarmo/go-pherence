## Pocket TTS resident weighting MLP -- 1 October 2026

`ptx-resident-weight-hybrid` and `vulkan-resident-weight-hybrid` now execute the learned log-variance weighting MLP on resident tapes alongside flow, transformer and AdamW/EMA. Sequence assembly, conditioner/EOS, scalar losses, time embeddings and final global norm-summary reduction stay on CPU. Full host state synchronises after each completed update. Native execution and complete device training are unqualified.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. Work is uncommitted, with the held inference index unchanged at eight paths / 469 insertions. No native GPU runtime/probe/launch/recovery, real model, recording or listening work occurred.

## Graph and scope

`model/pockettts/training_weight_tape.go` builds affine/ReLU hidden layers and one affine output, with input, bias and weight VJPs. Batched linears reuse the existing dual kernels with zero tangent inputs/seeds. Hidden ReLU derivatives are zero at nonpositive preactivations. New ReLU primitives overwrite disjoint outputs, require explicit forward/VJP mode and perform no runtime initialisation/fallback.

Graph geometry and all logical tensor storage are admitted before input/parameter copies or zero-tensor allocation. Rows are 1..1024, layer count 1..16, linear dimensions use the existing 16384 bound, and graph tensor/command limits stay 4096. The output and named parameter gradients use the checkpoint names `flow.w_s_t.{0,2,4,...}.{weight,bias}`. Decoded values are owned; resident boundary values are finite, shaped, mutually disjoint and disjoint from inputs.

The weighting backend is captured for the step's forward/backward pair. Forward returns log-variance values with accumulation disabled; one loss-seeded backward accumulates all weighting parameters on device and returns boundary values/time-input VJP only. Host weighting gradient arrays remain zero. The full optimiser upload excludes those device-owned names, preventing double accounting. Scalar loss reductions still form the log-variance seed on CPU.

Separate `*-resident-weight-hybrid-v1` scopes bind the runtime digest and all four graph/optimiser owners, including the weighting backend. Older `*-resident-hybrid-v1` keeps CPU weighting; neither scope silently accepts the other's checkpoint or later backend injection. Failed weighting forward/backward leaves host parameters/moments/EMA and replay at the prior completed boundary, blocks retries/publication and retains active owners.

## Offline numerics and failure tests

An independent F64 affine/ReLU implementation checks output and scalar-objective finite differences for every input/bias/weight derivative. Fixed bounds are 2e-7 forward and 2e-6 VJP. Exact integer cases cover negative, zero and positive preactivations; derivative at zero is zero. Signed-zero input follows the existing finite CPU convention. Host tape interpretation checks graph wiring; no native shader arithmetic was executed.

The combined weighting scope passes the pinned independent full-step fixture `training_step_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`. It checks all accumulated gradients, including six weighting tensors, at unchanged 5e-5 bounds, loss at 3e-5 and parameter/EMA updates at 4e-6. Tests verify exactly one weighting backward accumulation, no weighting parameter-gradient downloads, host zero gradients, unchanged conditioner branches and the minimal-LSD endpoint input-only boundary. The tiny complete step downloads 104 boundary tensors (100 prior resident boundaries plus four weighting value/input outputs).

Both named scopes pass tiny runner checkpoint/resume and retain resident allocation counts across steps. Cross-scope restore under resident-v1 rejects before mutation. Failure tests inject the first weighting forward and final weighting backward tape admissions, verify unchanged host/replay and no retry, and reject removed/injected weighting owners. Topology, finite-input/output, byte budget, decoded-output ownership and borrowed/overlapping boundary rejection are covered. The read-only review delegate timed out after 60 seconds without findings or approval.

## Gates and coverage

GLSL ReLU compiles with `glslangValidator -V -S comp`, validates with `spirv-val`, and stays within the existing closed SPIR-V envelope. Readable CUDA compiles offline with CUDA12.8 to PTX8.7/sm80 using `--fmad=false -ftz=false -prec-div=true -prec-sqrt=true --std=c++17`; `ptxas -arch=sm_80` assembles the whole module. Offline compilation does not execute kernels.

Final main window: **23:21:24.816230430--23:22:30.339593817 UTC**, 1 October 2026, exit 0. Twelve tests repeated ten times gave **120 top-level and 20 subtest passes**; race gave twelve top-level and two subtest passes. Affected regressions gave **61 top-level and 79 subtest passes**. Two ownership/topology follow-ups passed ten times (20 top-level passes) plus race in **23:23:28.044134661--23:23:48.227661096 UTC**. No final failures or skips.

Controls: local Go1.26.3, `GOENV=off`, empty `GOFLAGS`, `GOPROXY=off`, `GOSUMDB=off`, `GOMAXPROCS=2`, both GPU disable flags, nice10, serial packages/tests, 30-second inner / 120-second outer deadlines. Focused vet, whole-tree host build, Linux ARM64/RISC-V CLI cross-builds, layout (Go docs tests plus five Bun tests / 3,008 assertions) and whitespace checks pass. Foreign builds were not executed. Follow-up vet/build also pass.

Scoped coverage: shared ReLU admission and PTX/Vulkan wrappers 100%; weighting graph builder 92.3%, decode and boundary validation 100%; resident weighting evaluation and scope/owner checks 100%. Selected main package total is 13.9%; follow-up profile covers model weighting paths separately. Native transfers/dispatch/cleanup, released-size memory and performance and signal/voice quality are not established.

Archive: `/workspace/exports/pocket-gpu-training-weighting-20261001/`. Next source work is device conditioner/EOS and scalar loss/time-embedding orchestration, then all-device global norm and reducing graph/activation churn. Native numerical/resource/performance, statistical sampler and new-voice/listening validation require authorised work.
