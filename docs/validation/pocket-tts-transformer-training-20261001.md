## Pocket TTS transformer training tape -- 1 October 2026

`ptx-transformer-hybrid` and `vulkan-transformer-hybrid` now offload stateless transformer forward and all input/parameter derivatives through explicit tapes. Sequence assembly, conditioner/EOS/flow, weighting/loss reductions, clipping, AdamW/EMA and checkpointing use CPU code. Complete resident GPU training is unfinished. No native execution or trained-voice test ran.

Parent: `a90aec30567d485784f2e04d1803c543c9f2817f`. New work is uncommitted; the held inference index is unchanged at eight paths / 469 added lines. Native GPU execution/recovery and inference listening/publication holds are unchanged.

## Graph and backend integration

`model/pockettts/training_transformer_tape.go` composes bias-free projections, affine population LayerNorm/cache, adjacent-pair RoPE, exact-key causal attention, GELU-tanh, residual sums and optional layer scales. Backward includes all linear weights, norm affine values, layer-scale reductions and input adjoints. Q/K/V gather/scatter uses explicit row/block geometry; scatter zeroes other blocks before disjoint sums. No float atomics or in-place output aliases occur.

New shared row kernels gather/scatter `[rows,parts,width]`, multiply shared `[width]` scales and reduce row products into layer-scale gradients. Shapes are preflighted with rows<=1024, width<=16384 and parts<=4. GELU kernels consume flat F32 vectors and overwrite forward or seeded input derivatives. As with other raw stages, callers must keep cubic/derivative intermediates representable; device buffers are not host-scanned. `|x|<=1e10` suffices for those intermediates, but released-model ranges and device Exp errors are unmeasured.

The first GELU derivative schedule used `1-tanh²`, losing precision in saturated tails. At x=-5.1 it exceeded the fixed 6e-7 host bound (8.04e-7 error). The stable exponential form computes half-one-plus-tanh and sech-squared without subtracting nearly equal values; the same bound then passes. Across 642 forward/derivative values from -16 to 16, independent F64 tanh/central-difference comparison has maximum error **3.6949909976158324e-7**, RMS **4.0408695368122656e-8**. No threshold was widened.

Graph construction charges all logical tensor storage before copying parameters/inputs or materialising zero tensors. Metadata and a bounded shared RoPE table precede that gate. The graph admits at most 16 layers and the existing 4,096 tensor/command limits. Views are charged conservatively; native metadata/alignment and host/native peak RSS are separate. Parameters, inputs and decoded outputs are owned. Host topology tests cover two layers, final norm on/off, layer scales on/off and contexts 0/1/2/128; oversized context windows are restricted to the available keys, preserving CPU semantics.

`training_transformer_backend.go` exposes an explicit model backend. It shares bounded native tape owners with flow execution, retaining terminal resources/cleanup errors. `flowlm_training.go` routes transformer forward and backward to the captured backend with no CPU fallback, while preserving default nil CPU execution. It computes conditioning using a zero-seed transformer tape, then re-evaluates with the loss seed. Parameters remain immutable between those calls. This baseline transfers parameters twice per update and has no persistent forward tape, resident optimiser or measured speedup.

## Replay, failure and CLI

Separate `ptx-transformer-hybrid-v1` / `vulkan-transformer-hybrid-v1` identities bind backend scope and caller-declared runtime digest. The CLI requires native and sampler opt-in, a valid runtime pin and a positive per-pass tensor ceiling no greater than 2 GiB. Native setup follows option/local input admission. SIMD, attention-hybrid and flow-hybrid v1 digest construction is unchanged. Combined device scopes and full `ptx`/`vulkan` remain rejected.

Runner construction, update, checkpoint and restore check live scope. Synthetic transformer-hybrid steps checkpoint/resume correctly; cross-scope and changed-backend restores reject before mutation. Forward/backward failure leaves host parameters, moments, EMA and replay state unchanged, poisons the runner and blocks retry/checkpoint. A backend inserted or removed between CPU/device forward and backward is rejected. Returned outputs must be finite, correctly shaped, disjoint and not borrowed input storage; named gradients must cover the exact live parameter inventory.

## Offline verification

The pinned transformer fixture (`transformer_backward_pytorch.json`, upstream `0acce6b2f390150267557770d2098c5caa9a18ac`) passes host tape interpretation for output, input derivatives and all 12 parameter gradients. The tiny graph uses 175 tensors and 127 commands. Existing tolerances are unchanged: output 4e-6, input 8e-6, parameters 1e-5.

The backend passes the pinned FlowLM fixture with unchanged Z/EOS 5e-6, audio/voice 1.5e-5 and parameter 2e-5 bounds. Combined flow+transformer host tapes pass the pinned complete training-step fixture with 3e-5 loss and 5e-5 derivative bounds. These are independent fixture checks of host graph wiring. Device output distributions remain unmeasured.

GLSL compiles and validates using `glslangValidator -V -S comp` / `spirv-val`; reflection stays inside the existing closed SPIR-V envelope. Readable CUDA compiles offline to PTX8.7/sm80 under CUDA12.8 with `--fmad=false -ftz=false -prec-div=true -prec-sqrt=true --std=c++17`; the complete module assembles with `ptxas -arch=sm_80`. Compilers did not initialise CUDA/Vulkan.

Final gate: **22:20:30.506984227--22:21:24.981046336 UTC**, 1 October 2026, exit 0. Eighteen selected tests repeated ten times gave **180 top-level and 20 subtest passes**; race gave **18 top-level and two subtest passes**. Earlier affected regression gave 40 top-level and 56 subtest passes. No final failures or skips. Focused vet, whole-tree host build, Linux ARM64/RISC-V CLI cross-builds and whitespace checks pass. Layout passed in the prior implementation gate (Go docs tests, five Bun tests / 3,008 assertions). Cross-built binaries were not executed.

Controls: local Go 1.26.3, `GOENV=off`, empty `GOFLAGS`, `GOPROXY=off`, `GOSUMDB=off`, `GOMAXPROCS=2`, both NVIDIA/Vulkan disable flags, nice10, serial packages/tests, 30-second inner / 120-second outer deadlines. Test selectors exclude all native and real-asset cases.

Shared shape helpers, PTX/Vulkan GELU/row wrappers, model backend identity/evaluation/close, gradient conversion and result validation have 100% scoped statement coverage. Transformer graph builder is 94.4%, decode 93.3%, parameter helper 80%; most graph helpers are 100%. Direct forward/backward API is 76.9%; native executor entrypoints are 75% from host rejection paths, with successful device execution untested. Selected package total is 18.7%. Missing branches, native allocation/launch/transfer/cleanup and released-size resource use remain open.

Initial logs retain the test's nonexistent gradient-helper build failure, the flow-executor refactor return-type build failure and the GELU numerical failure. All were corrected before the final gates. A read-only independent delegate timed out after 60 seconds without findings or approval.

Archive: `/workspace/exports/pocket-gpu-training-transformer-20261001/`. Next source work is persistent parameter/gradient state, model-wide accumulation/global norm, device AdamW/EMA and coherent checkpoint transfer, followed by full-device orchestration. Native numerics/performance, released-size resource use, sampler distribution and real voice/listening qualification require separate authorised work.
