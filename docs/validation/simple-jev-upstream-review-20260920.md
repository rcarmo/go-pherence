# simple-jev upstream review -- 2026-09-20

Implementation is blocked on licensing. The reviewed repository is `featherless-ai/simple-jev` at commit `7cba7d121980e6a230477afa808ea0da0a7719e8` (2026-09-20). It has no root `LICENSE`, `COPYING` or `NOTICE`; package metadata does not grant a project-wide software licence. Upstream issue [#3](https://github.com/featherless-ai/simple-jev/issues/3) already asks which licence applies. Public source visibility and README “open source” wording are not sufficient permission to copy or derive its prompt strings and implementation into go-pherence.

No simple-jev source was copied, translated or executed inside go-pherence. This record describes externally observable contracts solely to size future work. Implementation must remain queued until upstream adds a compatible licence or the owner supplies explicit permission.

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

## Proposed gates after licensing clears

1. Pin the licensed upstream revision and preserve its licence/attribution.
2. Implement strict request types, deterministic structured JSON and all v1 plan strings/mappings. Generate Go fixtures from upstream rather than transcribing expected output manually.
3. Match upstream validation for XOR context, question limits, key ordering, 2/10/11/50-level score branches, Noul mapping, top-level/message extra handling and unsupported template versions.
4. Match float32 softmax/scoring for maps and vocabulary-indexed logits, including exact ties, NaN/Inf rejection, branch/label completeness, distinct token IDs and diagnostics filtering.
5. Add a model adapter that renders native chat templates, verifies labels as single distinct tokens, batches branches within explicit memory/token limits and returns zero generated tokens.
6. Validate at least one supported local model against upstream HF output logits and complete JSON responses. Separate prompt/scoring parity from model numerical parity and quality.
7. Add bounded OpenAI-adjacent HTTP endpoints only after the library contract passes; preserve existing APIs rather than overloading Jevlike routes.
8. Profile allocations and SIMD/scalar model paths under repository `AGENTS.md`; run whole-tree, docs, cross-build and native gates.

No GPU, external API, service or frozen evaluation artifact was touched during this review. The queue item is **blocked by upstream licensing**, not completed or abandoned.
