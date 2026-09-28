# MoJev released-model isolation probe — 24 September 2026

The pinned MoJev checkpoint leaks one candidate's change into another question's logits in a CPU text-only probe. The released-model output does not satisfy cross-question isolation for this input, although the separately tested 4D additive tree mask disallows the edge.

## Inputs and provenance

- Checkpoint: `MoLeMo-Lab/mojev@0c8695b6252f4205907433d4e196a94f032e60c3`, `model.safetensors` 1,710,234,304 bytes, SHA-256 `eae27bf03e0e44501316cafab2406b1505732ddf4b836d19fbb8deb642f55f50`. Config SHA-256 `1b3fb0dd8ae5a1e334b31bae1cfb2e8212231bd549884819304f29334313f4c9`. Explicit weight-use approval preceded the probe.
- MIT source: `MoLeMo-Lab/mojev@a74d58cd19ec573e83e8e27f9fecd837b8d830fb`, `mojev/modeling.py` SHA-256 `a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458`.
- CPU oracle: Transformers 5.17.0, `transformers/models/qwen3_5/modeling_qwen3_5.py` SHA-256 `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`; Torch 2.14.0+cu130 on CPU with two threads. The optional causal-conv and flash-linear-attention kernels were absent; Transformers used reference fallbacks.
- Script: `scripts/mojev_oracle_released_isolation.py` SHA-256 `0ce3b458f4799d7ef5350e162955a8770e7d192abc310569b07708b5c978ed6f`. Reproduced fixture: `model/mojev/testdata/released_isolation.json` SHA-256 `7084e08ddf96a7324fb6798354a1a57f97edd9b416a7b5b0efc8990dd78ec35d`. The script verifies the source and artifact hashes before loading; two runs produced byte-identical JSON.

The 12 integer token IDs form a fixed `[state 0:4 | q0 4:6 | a0 6 | a1 7 | q1 8:10 | b0 10 | b1 11]` sequence. A variant changes **only** a1 at position 7 from ID 42 to ID 123. Another changes only q1 at position 8 from ID 51 to ID 124. These are synthetic IDs, not a natural-language or quality probe. Spans, packed length, option masks and positions are unchanged.

| Input | q0 option 0 | q0 option 1 | q1 option 0 | q1 option 1 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.387547523 | 0.319101036 | -0.060931236 | -0.356784850 |
| Change a1 | 0.387547523 | 0.377660632 | 0.153374091 | -0.181278780 |
| Change q1 | 0.387547523 | 0.319101036 | -0.113518409 | -0.306189746 |

Changing a1 moved q1 option 0 by +0.214305326 and q1 option 1 by +0.175506070. The 4D tree mask rejects access from q1's candidates to a1 (`mask[10,7]` is masked). Changing q1 did not move q0 logits in this probe. The Go test pins these observations as a **regression detector**, not an accepted independence result.

## Mask boundary

In the pinned Transformers Qwen3.5 implementation, full-attention layers receive a 4D additive mask, while `create_recurrent_attention_mask` returns `None` for a non-2D mask. Linear-attention layers then run the recurrent path without a tree mask. The 24-layer encoder has 18 linear-attention layers and six full-attention layers. This code path explains a plausible route for the measured cross-question influence; the experiment does not isolate its exact layer or prove that no other mechanism contributes.

A second hash-pinned probe captures all 24 decoder-layer outputs using forward hooks. `scripts/mojev_oracle_isolation_layers.py` (SHA-256 `e820c3a915c9db7eddd34c0003c2a93fd835a8e5a3b35809b038ff0d996752c1`) produced `model/mojev/testdata/isolation_layers.json` (SHA-256 `844379006834b09a1db7606bc0b94f3cd5613ee169192cd66731ddcf45af1f5b`); two runs matched byte-for-byte. After **layer 0**, a linear-attention block, the maximum absolute differences from changing a1 are already 0.078125 in q1, 0.01953125 in b0 and 0.00927734375 in b1. State, q0 and a0 rows are unchanged. The first full-attention block is layer 3, so the initial cross-question influence occurs before it. By layer 23, q1/b0/b1 maxima are 1.375/1.9375/1.3125. This localises the onset to the first linear-attention block for this probe; it does not establish a general fix or qualify a replacement encoder.

A native Go scorer must not silently reuse causal Qwen3.5 KV/linear state as a substitute for the bidirectional tree contract. Before enabling it, specify and test tree-aware full and linear attention against an independent fixed checkpoint oracle, including same-position substitutions, permutations and padding. `ReadConfig.RuntimeReady` stays false. The model-free Go masks, text packer, tokenizer fixture and supplied-hidden-row head parity do not establish end-to-end isolation or quality.
