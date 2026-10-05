## Pocket TTS RoPE and causal attention tapes -- 1 October 2026

Vulkan/PTX training tapes now support adjacent-pair RoPE, causal attention forward, the existing five-stage attention backward and inverse RoPE. The composed `BuildAttentionTrainingTape` accepts unrotated Q/K/V and dOutput, and returns output, probabilities and Q/K/V gradients. Projection, normalisation/MLP, parameter gradients and optimiser work remain outside this graph. No new complete-transformer or full-device CLI scope is exposed.

The source parent is `a90aec30567d485784f2e04d1803c543c9f2817f`; these changes are uncommitted. The inference index remains eight paths / 469 added lines. No native runtime initialisation, probe, launch, reset/recovery, released-model, recording or listening run occurred.

## Rotation and exact keys

`backends/training/rope.go` prepares an immutable `[rows,dim/2,2]` cosine/sine table with host F64 trigonometry rounded to F32. Both backends consume the same table, shared across heads, and own disjoint output storage. Forward applies R; input adjoints apply R transpose. The table takes an explicit absolute origin; it does not infer positions from padding. Each shader invocation owns both adjacent output elements, avoiding cross-writer races.

RoPE admits rows 1..1024, heads 1..32 and even dimensions 2..256. Origin is nonnegative and the final exclusive position is bounded by 2^24; invalid periods and overflowing angles are rejected before the table is written. Table generation uses zero heap allocations on the admitted path. Kernels perform separate F32 products and add/subtract, with no device trigonometry or implicit CPU fallback.

Forward attention has three overwrite stages: scaled dense scores `[heads,rows,rows]`, stable row softmax of the exact causal/context keys, and weighted values `[rows,heads,dim]`. Context zero uses the full causal prefix; otherwise `key <= query && query-key < context`. Masked cells are zero, and masked probability cells never enter the value reduction. Softmax uses a row maximum and serial exp/sum reductions. Large finite inputs can still overflow F32 scores; finite decoded results are required and released-size numerical bounds have not been qualified.

`model/pockettts/training_attention_tape.go` composes two forward rotations, three forward-attention stages, five attention-gradient stages and two transpose rotations. It copies graph inputs and decoded outputs. Before input copies it admits the entire logical graph: 12 tensor-sized tensors, five dense tensors and one shared RoPE table. The byte estimate covers logical tensor storage; driver metadata, plan resources, allocator alignment and host/native peak RSS are separate.

Both native tape constructors now inspect every referenced kernel owner before allocation. PTX mixed tapes require the dual forward/RoPE module and the attention-backward module. Vulkan verifies each required live host kernel. Readiness is a host metadata check; it does not probe device health. New PTX rotation/forward-attention wrappers poison their operator after an attempted launch/synchronisation failure and reject retry. Vulkan plans retain their existing terminal submission/resource contract.

## Offline numerical checks

RoPE host schedules were compared with independent F64 complex multiplication and central differences of a scalar objective. Shapes cover one row, multiple heads, awkward even dimensions, 256-wide heads, nonzero/high origins and a period below one. Across 2,216 forward/adjoint values, maximum absolute error was **3.5826442612574283e-7**, RMS **6.209976275916252e-8**. Fixed test bounds are 1e-6 forward, 2e-7 adjoint and 1.5e-6 supplementary round-trip. The model-level tape also matches the existing transformer scalar convention within 1e-6.

Forward-attention host schedules were compared with an independent F64 dot/softmax/value reduction. Cases include rows 1/3/13, dimensions 2/5/14, multiple heads, context 0/1/2/full and amplitude 12 inputs. Across 4,605 probability/output values, maximum error was **2.9527005085405733e-7**, RMS **4.29484896583284e-8**, within the fixed 2e-6 bound. Each probability row sums to one within 2e-7 and all masked cells are zero.

The composed RoPE/attention tape's forward values and all Q/K/V gradients match a separate F64 complex-rotation/attention scalar objective and central differences within 2e-6, for context 0/1/2/full. A separate model integration test matches the existing CPU transformer attention convention at rows 1/3/13. These checks execute a host schedule interpreter, not native kernels. The existing pinned PyTorch flow full-step and mixed-derivative gates also pass without tolerance changes.

## Compilers, tests and coverage

The new GLSL shaders compile with `glslangValidator -V -S comp` and validate with `spirv-val`. Shader reflection passes inside the existing closed SPIR-V opcode/extended-instruction envelope; no parser expansion was needed. Readable CUDA source compiles with CUDA 12.8 to embedded PTX 8.7/sm80 using `--fmad=false -ftz=false -prec-div=true -prec-sqrt=true --std=c++17`; `ptxas -arch=sm_80` assembles the complete module offline. Device Exp approximation error is unmeasured.

Test controls: local Go toolchain, `GOENV=off`, empty `GOFLAGS`, `GOPROXY=off`, `GOSUMDB=off`, `GOMAXPROCS=2`, nice 10, both NVIDIA/Vulkan disable flags, serial packages/tests, 30-second inner and 120-second outer deadlines. Exact synthetic selectors exclude native tests and real assets.

RoPE gate: **21:50:08.567307724--21:51:02.682474098 UTC**. Seven tests repeated ten times gave 70 top-level passes; race gave seven passes. Seven affected owner/tape/flow regressions passed. Admission, table preparation and both wrappers each have 100% scoped statement coverage. Vet, host tree build, Linux ARM64/RISC-V CLI cross-builds and the layout gate passed. Foreign binaries were not executed.

Attention/tape gate: **21:58:14.319482126--21:59:20.797947291 UTC**. Nine tests repeated ten times gave 90 top-level passes; race gave nine passes. Seventeen affected regressions and eight subtests passed. Additional backward-tape rejection cases passed ten times plus race at **22:00:06.312749662--22:00:09.447478243 UTC**. There were no failures/skips in the final gates. Shared forward admission, both forward wrappers and both whole-tape kernel preflight helpers have 100% scoped statement coverage; composed model builder is 95.7% and decode 100%. Selected package total is 4.5%, since unrelated paths were excluded. Native execution/creation/transfer coverage and numerical qualification are open.

Focused vet, whole-tree host build, Linux ARM64/RISC-V CLI cross-builds and source layout passed after the final implementation changes. The layout check includes Go docs tests and five Bun tests / 3,008 assertions. Whitespace checks passed. No independent review was obtained for this source slice.

Two initial test defects are preserved in the logs: a RoPE rejection test assigned the already-valid output shape, and the PTX forward-attention test allocated dense-sized mock buffers smaller than the Q/K/V extent. Correcting these test fixtures produced the passing gates; no numerical tolerance changed.

Sources, compilers, test logs and coverage are in `/workspace/exports/pocket-gpu-training-rope-20261001/` and `/workspace/exports/pocket-gpu-training-attention-forward-20261001/`. Full-device transformer norm/MLP/projections/residual gradients, persistent parameter/gradient residency, global norm, device AdamW/EMA and coherent checkpoint transfer still need implementation. Native tests require Rui/@default coordination; real corpus, listening and publication holds are unchanged.
