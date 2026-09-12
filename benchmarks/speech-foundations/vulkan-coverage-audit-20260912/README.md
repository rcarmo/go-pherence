# Vulkan speech coverage audit — 12 September 2026

This model-free checkpoint audited the actual resident F32 Whisper and Community-1 graphs, legacy Vulkan primitive wiring and current backend documentation.

## Findings and changes

- The resident Whisper encoder remains explicit and fallback-free: Conv1D, add, LayerNorm, linear, erf-GELU and non-causal attention execute through owned Vulkan arenas/plans.
- Community-1 remains an explicit hybrid: recurrent LSTM and ResNet CNN work is resident Vulkan; SincNet/head/pooling/projection/postprocessing placement remains stated as CPU. There is no automatic recurrent/CNN fallback.
- The recurrent owner uploads either supplied or zero hidden/cell state before every call. Mutable state is not inherited intentionally between requests.
- Removed dead hand-built SPIR-V placeholder builders. The legacy assembler now exposes only vector-add and returns an owned copy. Production operations consume generated embedded SPIR-V.
- Added a regression check that legacy operation names such as GEMV/BF16/attention fail closed and that the generated GEMV contract is distinct from vector-add.
- Fixed the legacy GQA attention-score wrapper: its shader owns one workgroup per `(head,time)` pair, but the host previously dispatched only the head dimension. It now dispatches `nHeads × seqLen`, rejects non-integral/invalid GQA head ratios, and rejects non-finite scale.
- Updated stale backend/inventory documentation. Generic `VulkanBF16Ready` is device admission only; it is not a BF16 speech-graph claim.

The legacy GQA score wrapper is separate from the fused resident Whisper attention operator, but is used by the SpacemiT Vulkan bridge and must remain numerically correct.

## Verification

- The previously failing `TestVulkanAttentionScoresF32Parity` now passes (`4 heads × 3 time positions`).
- Complete model-free `backends/vulkan`, `models/whisper` and `models/speaker/community1` suites pass once.
- Ten shuffled repetitions of all three packages pass.
- Affected `go vet` passes.
- Linux/ARM64 test-binary cross-builds pass for Vulkan and Community-1.
- `git diff --check` passes.

The Linux/ARM64 Whisper helper gap observed during this audit was subsequently fixed by `f1aca19`. Windows is owner-declared out of scope and is not a build or acceptance gate.

No native GPU, trained checkpoint, model download, corpus, service, deployment, default, dependency pin or quality threshold changed. Quantised/F16 speech graphs, trained/native placement and broader quality/performance gates remain open.
