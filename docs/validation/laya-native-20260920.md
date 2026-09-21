# Native Laya validation — 2026-09-20

This milestone implements the released Laya typed decision model in native Go. It pins upstream source `42626c348753fbb17572a813127df2278a1ec527`, model revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, and the previously validated native ModernBERT encoder. Source and model declare Apache-2.0.

## Numerical and API evidence

`scripts/laya-reference.py` builds a deterministic width-16 model and records option/action logits. Go matches this fixture through both the allocating convenience API and reusable session.

`scripts/laya-released-reference.py` runs the pinned 842,609,210-byte checkpoint on CPU and records exact sequences, padded batch inputs/masks, marker positions/masks, raw option logits, raw action logits, and the public response for choice, score and Noul questions. Go matches all raw outputs at `3e-5 + 3e-4*abs(reference)`. The public typed API independently reproduces the selected `red` choice, support score `1.5291`, Noul value `0.7859`, option probabilities, confidence values and action probability. Choice criteria are represented as an ordered slice; map iteration cannot alter option semantics.

The checkpoint is 842,609,210 bytes with SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`. Exact token sequences and markers match upstream for all three question forms. A ModernBERT tokenizer compatibility fold reproduces the default GPT-2 ByteLevel handling of an optional leading space in numeric runs (`" 0"`) without changing the shared tokenizer (which is part of the frozen evaluation manifest) or Qwen behavior.

## Ownership, admission and performance

Public `New` copies caller tensors. The private released loader owns freshly decoded tensors, rejects malformed configuration, shape/rank/length mismatches, missing and unexpected head tensors, head/config disagreement, invalid calibration arrays/buckets and invalid action count. Sessions enforce bounded sequence and option capacities and reject malformed masks, markers and output buffers.

On an Intel i7-12700 with `GOMAXPROCS=2`, tiny warm inference measured about 34 µs and released 27-token choice inference about 0.44–0.58 s. Both report **0 B/op / 0 allocations/op** through `Session.ForwardInto`. CPU profiling attributes about 48% flat time to the FMA SGEMM tile. Heap profiling is dominated by one-time safetensor FP16-to-FP32 loading; it is not warm inference allocation. Package statement coverage with the opt-in released model is **94.0%**.

Native ARM64 validation uses `GOMAXPROCS=2 nice -n 10` and the deterministic all-layer fixture. The 842,609,210-byte released checkpoint is not copied to that host, so released ARM numerical validation is not claimed. Linux/ARM64 and Linux/RISC-V builds are compile gates only. SIMD-disabled tests exercise scalar dispatch on the local host.

## Limits

This is an inference implementation, not the upstream reinforcement-learning/training pipeline. It supports the released FP32/BF16/F16 safetensor conversion path but does not provide quantized heads. `SystemOne` evaluates ordered questions sequentially rather than as one padded batch; output semantics and token accounting match, but throughput claims must use the measured implementation. The released reference covers one representative question of each public type, not calibration quality on an external corpus. No GPU was queried or used, and frozen evaluation artifacts/services were unchanged.
