# MoJev native text isolation fix — 2026-09-25

The Go text scorer now executes the released MoJev weights with separate candidate histories. Its real-weight tests keep unrelated scores exactly unchanged when a candidate or question changes, including length changes and permutations. The upstream release still contains the packed-history leak; this implementation does not modify upstream.

## Execution

`model/mojev.LoadTextScorer` loads the text encoder and four head tensors from a caller-owned `weights.Source`. It checks the pinned topology, arithmetic settings, tensor dimensions, types and finite values. Text tensors are decoded to owned F32 arrays; the source can close after loading. No vision tensors are loaded. The same released checkpoint used in the earlier probes was reused; no new model was downloaded.

`TextScorer.ScoreEncoded` accepts pre-tokenised state/question/candidate segments. For each candidate, it runs all 24 Qwen3.5 layers on only its state, question and candidate. Full attention sees all tokens in its node and ancestors. Linear attention remains causal along the path with fresh convolution/recurrent state. `Qwen35BaseModel.ForwardTextBranch` is a separate entry point; existing causal generation and MTP APIs are unchanged.

Positions start at zero within each branch. Keeping the original packed positions, as the first repaired probe did, would leave a dependency on sibling lengths and order. The native implementation deliberately drops that dependency. It uses F32 arithmetic over the unchanged released BF16 weight values, not the original BF16 execution trajectory.

`TextScorer.ScoreText` accepts a decoded `TextRequest` and loaded tokenizer, rejects reserved tokens, validates field order, tokenises the request, runs the actual Go encoder/head and builds public answers. Its logical input-token usage follows the existing packed accounting, excluding repeated ancestor computation. For small menus the upstream prompt includes option text: changing a menu changes that question's input. Candidate-only independence is tested at `ScoreEncoded` with question tokens held fixed; cross-question independence is also tested through `ScoreText`.

## Independent reference and tests

`scripts/mojev_oracle_native_text.py` loads the unchanged approved checkpoint with pinned upstream `a74d58cd19ec573e83e8e27f9fecd837b8d830fb`, Transformers 5.17.0 and the pinned Qwen3.5 implementation hash. It converts the model to F32 and calls upstream `PackedScorer` separately on each candidate path using local positions. Its helper script and model/config/source hashes are checked. Two runs produced byte-identical `model/mojev/testdata/native_text.json` (SHA-256 `80b2d914a1dd1ac0a678f491b0b9f869d3fc27499dfe8d477926f9cb44178e1a`). The generator SHA-256 is `8b7c0f07b24253921af3a9ba4f84013f152401303c05c91b1d9dd3ca894dcb27`.

The eight cases are baseline, candidate substitution, question substitution, longer sibling candidate, longer sibling question, candidate order reversal, question order reversal and multi-token nodes. The largest path has 52 tokens. Go executes weights, not injected hidden rows or logits:

- Maximum absolute logit error: `4.73559e-5`, below the fixed `3e-4` gate.
- Maximum absolute final encoder-row error across all four baseline paths: `7.54952e-4`, below the fixed `2e-3` gate.
- Unrelated native logits remain bit-identical under substitutions and length changes; reordered logits map back exactly.
- Repeated calls, a failed invalid-token call followed by recovery, and two concurrent requests produce identical scores.
- A real-tokenizer two-question `ScoreText` test changes the first question's length/content and keeps the second public answer identical. Reserved-token input is rejected.
- The weight-backed test passes with the race detector. Default offline tests cover fixture pins, synthetic full-attention bidirectionality/ancestor visibility, fresh linear state, malformed input, loader failures, source ownership and concurrent calls.

Run the released test with:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 \
GO_PHERENCE_MOJEV_CHECKPOINT_DIR=/path/to/verified/mojev \
go test -race ./model/mojev -run '^TestReleasedNativeTextScorer$' -count=1 -v -timeout=400s
```

Whole-tree CPU race, vet, build and Linux ARM64/RISC-V cross-build checks passed. Native execution was on Linux amd64 with Go 1.26.3. Cross-builds do not qualify other hardware.

## Limits

This fixes cross-question/candidate-history contamination for the Go text-only path. It costs one encoder pass per candidate, repeating ancestor work. No speedup or maximum-context memory claim is made. The input guard admits at most 4,096 total encoded tokens; released numerical evidence covers the bounded cases above, not every admitted length.

Images, held-out classification quality/calibration, cancellation, broad memory/concurrency admission, HTTP service wiring and native ARM64/RVV execution still need qualification. No training weights were changed, and quality under the repaired policy can differ from the leaky release. The broad multimodal `ReadConfig.RuntimeReady` flag remains false; callers explicitly select `TextScorer` for this text-only implementation.
