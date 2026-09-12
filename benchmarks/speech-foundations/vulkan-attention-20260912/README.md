# Vulkan fused F32 attention

`VkAttentionF32` adds arena/plan non-causal attention over time-major F32 tensors. A fused 16-query/16-key tile computes online softmax without a quadratic score allocation. No Vulkan device or trained model ran; CPU/model defaults are unchanged.

## Contract

- Q and output: `[seqQ, heads*headDim]`; K and V: `[seqKV, heads*headDim]`.
- Sequence lengths 1–4096, heads 1–32, head dimensions 1–64. Q/K/V share the head count and width. Empty keys are rejected before recording.
- Fixed scale `float32(1/sqrt(headDim))`, matching the current Whisper encoder convention. No mask, causal mode, GQA, F16 or quantised input.
- All tensors have rank 2 and exact byte extents. Shape and device-grid/range checks precede native recording. Every output/input overlap is rejected, including exact alias; read-only inputs may alias.
- Input contents must have finite representable products, dot sums and weighted-value sums. There is no content scan or fallback. Finite individual elements alone cannot prevent arithmetic overflow.
- Four descriptors and 20 push bytes: seqQ, seqKV, heads, headDim, scale. Dispatch grid is `[ceil(seqQ/16), heads, 1]` with local size 16×16×1.

The operator API accepts the same flattened time-major layout produced by the F32 projection operator. No host packing is required between these components. The current Whisper encoder has not been switched to them.

## Tile execution

Each group owns 16 query rows in one head. It loads the query tile once, then iterates over 16-key tiles. Shared storage is 9408 bytes: query and K/V arrays of 1024 floats each, 256 probability floats, and three 16-float row arrays. Each lane holds four output channels in a private array.

1. Cooperatively load Q and K, zero-padding missing rows/channels.
2. Each lane computes one score for its query/key pair.
3. The x=0 lane of each row computes its tile maximum, exponential weights and running sum. Missing keys get zero weight.
4. Rescale the prior output accumulator by `exp(oldMax-newMax)`, then add the current weighted V tile.
5. After all key tiles, divide each owned output channel by the row sum.

K and V reuse the same shared allocation. Barriers separate K loads, score consumption, probability updates, V loads and output accumulation. Tail queries remain in every barrier; only the final store is query-guarded. No output element is shared between workgroups.

For tile maximum `m` and previous maximum `oldMax`, the updated maximum is `max(m,oldMax)` after the first tile. Both running sum and output accumulator receive the same exponential rescaling. The first tile starts with alpha zero. Positive key length guarantees at least one valid key and a positive denominator for finite representable scores.

`precise` constrains score/private accumulator arithmetic. This does not prescribe every driver reduction/contraction/transcendental result; the Go model uses explicit F32 rounding and Go's float64 Exp rounded to F32. Device rounding, subnormals, occupancy, register spills, serial row-softmax cost and performance have not been measured.

## Verification

| Check | Result |
|---|---|
| Full offline suite | 115 top-level tests, 423 passing test/subtest events, zero failures/skips |
| Shuffled repeats | 48 Attention/GELU/Linear/LayerNorm/plan/shader tests ×30; 5190 passing events, separately counted |
| Ordinary numerical fixtures | 12 shapes ×4 lane/group orders; 53,280 compared output values |
| Ordinary float64 oracle error | Maximum absolute error `7.073017253000913e-7`; acceptance `2e-5 + 2e-5*abs(reference)` |
| Softmax stress | 306 values with rising/falling/tied maxima and logits separated by more than 1000 |
| Stress error | Maximum absolute error `1.8180898553321612e-6`; separate stress budget `2e-4 + 2e-5*abs(reference)` |
| Static validation | 15 embedded +15 rebuilt modules pass `spirv-val --target-env vulkan1.3` |
| Rebuild comparison | 15 narrowly normalised matches; attention, GELU, linear, LayerNorm and RoPE byte-identical |
| Checker tests | Six Bun tests, 53 assertions |

The independent oracle stores a full score row and computes float64 softmax and weighted sums; it does not reuse the online recurrence. The Go tile model follows barrier-separated phases, uses shuffled group/lane order, checks shared-tile and output ownership, and preserves output canaries. It does not execute SPIR-V or prove device race-freedom.

Shapes include odd sequence/head/channel counts, dimensions around 16/32/64 boundaries, sequence 4096 with the other sequence set to 1, and 32 heads. No full 4096×4096 workload ran. Metadata-only maximum-shape checks require no large allocation. A single-key case returns the exact corresponding value vector. Stress fixtures have a larger explicit absolute budget to account for F32 logit subtraction at large magnitudes.

Seven new top-level tests cover numerical models, actual wrapper descriptor/push/grid capture, independent stage slices, rank/shape/range/alias admission, constructor/close, and attention→linear plan retention. The plan safely reuses Q storage as the later projection output after attention has consumed Q. Cancelling after submission retains both operators and both arenas until confirmed drain.

The initial wrapper/plan tests failed because their inherited memory mock only allowed 64-byte allocations. A bounded 2048-byte Go-backed mock fixed the test setup; the final runs pass. No production memory code changed. The shader uses the existing admitted SPIR-V instruction envelope; no parser broadening was needed.

Focused make targets, speech/affine/media regressions, affected vet/builds and arm64 test cross-build pass. Full-tree/backend compile-only errors match the GELU checkpoint. The race build cannot run because `gcc` is absent; the arm64 executable was not run.

## Review and evidence

The first source-review delegate timed out without findings. A second shader-only review found no scoped barrier, index or online-softmax issue for positive key length. It flagged the zero-key division case; the wrapper rejects `seqKV=0`, and invalid-K shape admission is tested. That review did not inspect the wrapper, generated binary or external barriers. Parent review checked wrapper indexing bounds, shape/alias admission and mock lifetime evidence. No GPU accuracy/performance approval was obtained.

- [evidence.json](evidence.json): checks, scope and source/evidence hashes.
- [Static verification](static/verification.json) and [shader disassembly](attention.spvasm).
- [Test events](tests.jsonl), compressed shuffled events and focused command logs in this directory.

```sh
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-attention-static
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-offline-check speech-affine-check speech-foundations-check speech-media-integration
```

Set `SPIRV_VAL` and `GLSLANG_VALIDATOR` if needed. Use a new static report directory.

No trained-model quality, speedup, quantised execution, complete resident encoder or device recovery is qualified. The strict SincNet four-failure hold is unchanged. No private media, model weights, service changes, deployment, restart or push were used.
