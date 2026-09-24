# simple-jev upstream review -- 2026-09-20

The 2026-09-20 review of `featherless-ai/simple-jev@7cba7d121980e6a230477afa808ea0da0a7719e8` found no project-wide licence. On 2026-09-23, upstream `b02aa81c915a8193759b3cd33fef74721d6e005b` includes a root Apache-2.0 `LICENSE` (SHA-256 `91ce9931497b6e930c3cbe70a48fd85ddce3e249448c76aa4a8320ef7ca47a19`, copyright Featherless AI / Recursal AI, 2026); upstream [#3](https://github.com/featherless-ai/simple-jev/issues/3) is closed with an Apache-2.0 confirmation. The licensing uncertainty that blocked this review has cleared for that revision.

No simple-jev source or prompt text has been copied, translated or executed inside go-pherence. `model/simplejev` now has separately authored, model-free ordered-label scoring: finite F32 logits, distinct token/public IDs, bounded softmax, exact-logit tie handling and explicit ordinal values. A separate strict, bounded JSON request envelope now admits one state and ordered choice/ordinal questions, but does not reproduce the upstream request schema. An injected model-free label-logit provider can now assemble owned choice/ordinal answers transactionally. An injected model-free adapter requires explicit public-ID/model-text mappings, verifies that each label adds one distinct token at the actual tokenizer boundary and reads only selected logits. It ships no upstream prompt or concrete tokenizer/model backend. A separately generated model-free scoring fixture now records upstream Choice/Score distributions and the nine-bin Noul numerical mapping, without asserting request/prompt or released-model parity. This record describes externally observable contracts to size the remaining work. `go-pherence` uses MIT; copying or deriving upstream Apache-2.0 source requires preserving its licence and applicable notices. A separately authored MIT implementation must avoid transplanting Apache-licensed code or prompt text. The older Jev bake-off excluded Simple-JEV under its frozen policy and results; a new evaluation needs a new untouched cohort.

## Model-free scoring fixture (2026-09-24)

A separate read-only checkout of `featherless-ai/simple-jev@b02aa81c915a8193759b3cd33fef74721d6e005b` was run with Python 3, Pydantic 2, NumPy 2 and an injected map of finite logits. The pinned upstream scorer `common/response_scoring.py` has SHA-256 `b1b1303ace5218fb40904e7aaf5a1ad01aa3d072c30678d45e921bb0e8613dfa`. `scripts/simplejev_oracle_scoring.py` verifies the upstream revision and source hash before generating `model/simplejev/testdata/upstream_scoring_v1.json`. No checkpoint, HF adapter or service was run. Upstream's model-free pipeline suite passed 20 tests with four skipped when FastAPI was present; a first diagnostic run with FastAPI absent produced three import failures and is not counted as a pass.

The fixture pins three branches and synthetic unequal, tie and extreme finite-logit rows: Choice takes the first tied input, Score returns the expected zero-based level, and Noul uses nine rating bins mapped to public `[0.01,0.99]`. Three malformed upstream rows (missing label, extra branch, non-finite logit) are recorded as rejected. A Go offline test hashes the fixture and generator, compares Choice/Score probabilities and Noul's *numerical* nine-bin expectation within `1e-6`, and retains separate native error tests. Go does not expose upstream Noul request or response envelopes. The test does not render upstream prompt strings, tokenize labels, run a model, or qualify HTTP/wire compatibility. No Apache source or prompt string was copied into the MIT package. Ten repeated native focused tests and ten race runs passed with 94.3% package statement coverage. Vet, build, docs/layout, and Linux ARM64/RISC-V cross-builds passed. The first whole-tree CPU race pass encountered a `backends/vulkan` offline device-loss deadline failure; its focused test passed ten repeats and the subsequent whole-tree CPU race rerun exited successfully. That first failure is not counted as a passing gate.

## Text-state request subset (2026-09-24)

A second model-free oracle calls only upstream `ClassifierRequest.model_validate` at the same revision. `scripts/simplejev_oracle_text_state.py` checks `common/request_schema.py` SHA-256 `6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac` and writes 12 accepted/rejected observations to `model/simplejev/testdata/upstream_text_state_v1.json`. The separately named Go `DecodeTextStateRequest` preserves order and null/text distinctions for one string state and Choice, Score or Noul questions. It rejects duplicate keys, nested unknowns, trailing data, malformed UTF-8, oversized requests and more than 256 questions; choice/score counts cover 2 and 50. It accepts empty text state/instructions, unlike the older internal `DecodeRequest` envelope, which is unchanged. Unsupported messages, structured JSON state/instructions, media, tools and other adapter-specific fields fail explicitly. Top-level unrelated fields are ignored, matching the observed subset. This is deliberately narrower than upstream Pydantic (which also admits structured state and messages); no prompt template, tokenizer, model or HTTP response was qualified. No Apache source or prompt strings entered the Go package.

## Pinned contract

simple-jev performs structured classification from selected next-token logits rather than a separate classifier head. The shared `common/` layer is backend-independent; `hf-server/` renders model-native chat prompts and runs Transformers, while a newer Laya backend uses Typed Decisions. The public request accepts exactly one context source (`state` or nonempty `messages`) and one to 256 ordered questions.

Three question kinds exist:

* `choice`: 2--50 insertion-ordered candidate IDs/descriptions. Model labels are case-sensitive `A`--`Z`, then `a`--`x`; exact ties choose the first input candidate.
* `score`: 2--50 ordered rubric levels. Up to ten levels use numeric labels; larger rubrics use letters. Public score is the expected zero-based level, which may be fractional.
* `noul`: optional true/false descriptions but fixed model labels `1`--`9`. The expected rating maps to public `[0.01,0.99]`; it is not a calibrated probability or binary-token softmax.

Question/options objects reject unknown fields. Top-level unknown fields are ignored for client compatibility; message extras are retained for backend validation. Instructions and descriptions may be text, structured JSON or null. Structured content uses deterministic sorted-key, compact UTF-8 JSON. Non-finite values fail.

Prompt template version `v1` is explicit and immutable. It contains a common system prefix, a briefing containing all question instructions, caller context, a shared suffix, then one selected-question branch with options/rubric and an incomplete JSON answer prefix. The adapter applies its native chat template and must verify that every output label is a single distinct token at that exact boundary. One model forward branch is scored per question; sampled text and JSON parsing are not part of the scoring contract.

Responses use finite raw logits for exactly the planned branches and labels. Stable float32 softmax normalizes only allowed labels. Choice returns the winning public ID, distribution and uncalibrated confidence; score returns expected rubric index and legend; Noul returns the mapped expected rating without a separate confidence. Raw logits require both request opt-in and advanced output. Usage counts are supplied by the backend and output tokens default to zero.

## Relationship to go-pherence

go-pherence's existing `model/jevlike` is a learned context/option scorer with its own byte or frozen-decoder encoders and trainable scoring head. It does not implement simple-jev's next-token label-logit prompt contract. Reusing Jevlike as if it were compatible would change both model and scoring semantics.

A future implementation should therefore be a distinct package or adapter layer, sharing only generic components whose contracts genuinely match: bounded HTTP decoding, tokenizer/chat rendering, model next-token logits, stable softmax and response envelopes. It should not reuse Jevlike calibration or head training by name.

## Proposed implementation gates

1. Pin `b02aa81c915a8193759b3cd33fef74721d6e005b` or another licensed upstream revision. For derived work, preserve Apache-2.0 licence/attribution. For a distinct MIT implementation, use independent source and fixtures without copying upstream code or prompt strings.
2. Implement strict request types, deterministic structured JSON and all v1 plan strings/mappings. Generate Go fixtures from upstream rather than transcribing expected output manually.
3. Match upstream validation for XOR context, question limits, key ordering, 2/10/11/50-level score branches, Noul mapping, top-level/message extra handling and unsupported template versions.
4. Match float32 softmax/scoring for maps and vocabulary-indexed logits, including exact ties, NaN/Inf rejection, branch/label completeness, distinct token IDs and diagnostics filtering.
5. Add a model adapter that renders native chat templates, verifies labels as single distinct tokens, batches branches within explicit memory/token limits and returns zero generated tokens.
6. Validate at least one supported local model against upstream HF output logits and complete JSON responses. Separate prompt/scoring parity from model numerical parity and quality.
7. Add bounded OpenAI-adjacent HTTP endpoints only after the library contract passes; preserve existing APIs rather than overloading Jevlike routes.
8. Profile allocations and SIMD/scalar model paths under repository `AGENTS.md`; run whole-tree, docs, cross-build and native gates.

The original review touched no GPU, service or frozen evaluation artifact. The licence now permits a scoped implementation decision; model/fixture approval, an independent MIT design if required, and evaluation remain separate gates.
