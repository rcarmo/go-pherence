# MoJev hybrid encoder contract

The pinned released MoJev checkpoint cannot yet serve independent questions from one packed request. A candidate change reaches another question in the first Qwen3.5 linear-attention block. The Go scorer must keep `RuntimeReady=false` until the intended isolation rule and checkpoint behaviour agree.

## Evidence and scope

The [released isolation probe](../validation/mojev-released-isolation-20260924.md) uses the approved `MoLeMo-Lab/mojev@0c8695b6252f4205907433d4e196a94f032e60c3` checkpoint (1,710,234,304 bytes; SHA-256 `eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`), MIT source `a74d58cd19ec573e83e8e27f9fecd837b8d830fb`, and Transformers 5.17.0 CPU code. A same-position candidate change leaves earlier branches unchanged but moves another question's logits by +0.214305326 and +0.175506070. Forward hooks first observe cross-question differences after layer 0 (`linear_attention`), before the first full-attention layer at index 3. These values describe one synthetic twelve-token probe, not a held-out quality result.

`PackedScorer.build_mask` builds a 4D additive visibility matrix. In the pinned Transformers implementation, Qwen3.5 passes that mask to full-attention layers. `create_recurrent_attention_mask` returns `None` for a non-2D mask, so linear-attention layers receive no tree restriction. Their recurrent state processes the packed sequence in token order. The layer-zero observation establishes that this path leaks for the probe; it does not prove the extent of leakage on other inputs or that a proposed repair preserves the trained checkpoint's predictions.

## Required visibility

For a text-only packed row `[state | q0 | a0 | a1 | q1 | b0 | b1 ...]`, each output token may depend only on its ancestor branch and its own node:

| Output node | Permitted inputs |
| --- | --- |
| State | State tokens |
| Question `qᵢ` | State tokens and the same question's tokens |
| Candidate `aᵢⱼ` | State tokens, `qᵢ` tokens and the same candidate's tokens |
| Padding | Its own diagonal; no pooled score may read padding |

Sibling questions and candidates have no visibility edge. A fixed-length same-position substitution inside one candidate must leave state, other questions and all their candidate hidden rows and logits unchanged. Candidate order is canonicalised by text at the serving boundary, then mapped back to caller order. The ordering rule does not replace attention isolation. A padding token may attend to itself for finite softmax, but valid tokens must not read it.

The current Go `TreeMask` and `AdditiveTreeMask` check the reference matrix for at most 4,096 positions. They allocate an L×L result and do not feed `model/qwen`. `PackEncodedRows` and `PackTextRows` provide bounded text spans; `LoadHead` and `ScoreHidden` compute a separate released-head readout from supplied upstream hidden rows. These pieces establish neither an encoder nor the checkpoint's advertised 16,384-token context.

## State lifetime and implementation order

1. Freeze an independent oracle for the intended repaired semantics. Choose explicitly between **released-checkpoint parity** (which includes the measured leak) and **repaired isolation** (which changes the checkpoint's predictions). A Go implementation cannot claim both for the current release. Keep the released-leak fixture as a regression detector, and pin a separate repaired fixture and implementation hash if the model owner adopts a fix.
2. Define segment evaluation for both Qwen3.5 attention types. A state branch may create a shared ancestor snapshot. Each question starts from a copy of that snapshot; each candidate starts from a copy of its own completed question snapshot. Never append a sibling's K/V or recurrent convolution/delta state to another branch. Process only branch-local tokens after a fork, with absolute position IDs preserved from the packed row unless an independent oracle validates a different policy. A copied causal state does **not** provide bidirectional within-node visibility; specify and test that rule separately before using this scheme as a replacement for the 4D mask.
3. Implement a bounded, request-owned, text-only masked encoder path outside the existing causal `Qwen35BaseModel.ForwardSequence` API. Keep the causal Qwen3.5 and MTP paths unchanged. Separate full-attention K/V, linear-attention convolution/recurrent state and position IDs per branch. Validate every tensor name/shape and return no partial outputs on failure. Do not allocate a 16,384² float32 mask by default (over 1 GiB for one matrix); profile memory against the admitted sequence length.
4. Compare per-layer hidden rows and final logits with the selected independent oracle for state, two questions, sibling candidates, changed question text, padding and candidate permutations. Include long multi-token nodes and mixed requests. Run repeated requests and race tests to detect state reuse. If using a repaired encoder with the released weights, measure calibration and held-out Choice/Noul/Score quality anew; the earlier head-only 3e-4 tolerance does not cover encoder drift.
5. Add tokenizer/processor and image-path parity, peak/retained memory, cancellation, concurrency, ARM64/RISC-V execution and service tests before declaring a native scorer ready. Leave `ReadConfig.RuntimeReady=false` until all required gates have evidence.

The existing `qwen.CloneQwen35BaseForwardState` deep-copies full-attention K/V (including spare capacity) and linear convolution/delta state. `model/mojev/branch_state_test.go` verifies state→question→candidate forks and concurrent candidate mutation against synthetic state without changing the shared Qwen3.5 runtime. It establishes ownership of the copies. It does not run attention, establish bidirectional within-node visibility, or repair the released checkpoint. If no independent repaired oracle is approved, stop before implementing a purported isolation fix.
