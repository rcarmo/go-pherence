# ModernBERT native encoder validation -- 2026-09-20

This milestone implements a reusable native FP32 ModernBERT encoder as the prerequisite for Laya. It follows Transformers commit `c587bc884db2c2e31fc2b8102314656b17aa07b1` and pins `answerdotai/ModernBERT-large` revision `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`, whose model and code are Apache-2.0 licensed.

Implemented model operations are token embedding and embedding LayerNorm; bias-free fused QKV/output projections; bidirectional full/sliding attention; full/local RoPE; first-layer identity attention norm and subsequent pre-norm; exact-GELU GeGLU MLP; residuals and final LayerNorm. Unsupported biases, activations, layer types and malformed geometry fail explicitly.

## Numerical fixtures

`scripts/modernbert-reference.py` creates a CPU-only, width-16, three-layer fixture with full/sliding/full attention, padding and deterministic nonzero weights. Go matches the embedding output, every encoder layer and final norm using `3e-5` absolute / `2e-4` relative tolerances. Padding masks keys but does not erase padded query rows, matching the pinned eager attention implementation.

`scripts/modernbert-released-reference.py` loads the released 28-layer checkpoint on CPU and records layers 0, 13, 27 and final output for `[CLS] Hello [SEP]`. Go matches those values using `3e-4` / `2e-3`; no tolerance was widened during implementation. The checkpoint contains 173 FP32 tensors and has SHA-256 `44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f` (1,583,544,840 bytes).

The strict model-directory loader validates released config defaults, derives layer types from `global_attn_every_n_layers`, supplies the documented 160,000/10,000 full/local RoPE defaults when config fields are null, requires every encoder tensor and rejects unexpected tensors. Public constructors copy caller-owned tensors; the private file loader transfers ownership of freshly decoded data, reducing released-test peak RSS from roughly 4.4 GiB to 2.95 GiB. The tokenizer wrapper checks released `[CLS]`, `[SEP]`, `[PAD]` and `[MASK]` IDs.

## Allocation and timing

`Session` owns workspace for a bounded maximum sequence length and is not concurrent-safe. `ForwardInto` writes a caller-owned `[sequence,hidden]` buffer. After warm-up both tiny and released inference measure **0 B/op / 0 allocations/op**. `Forward` remains an allocating convenience API because it creates a session and owned output.

On the Intel i7-12700 with `GOMAXPROCS=2`, five tiny samples were about 19--24 µs. Ten released three-token samples were 74.6--95.7 ms, with a median around 79 ms. These are encoder-only microbenchmarks, not Laya latency or throughput claims.

Native ARM64 CIX P1 runs used `GOMAXPROCS=2 nice -n 10`. The tiny all-layer fixture, admission/ownership/loader tests and warm-allocation assertion passed; `BenchmarkTinyForward` measured about 35.3 µs and 0 B/op / 0 allocations/op. The released 1.58 GB checkpoint was not present on that host, so released ARM numerical/runtime validation remains open. Linux/ARM64 and Linux/RISC-V builds are separate compile gates.

## Coverage and limits

With the opt-in released checkpoint/tokenizer tests enabled, package statement coverage is **90.7%**, above the repository target. Without local released assets it is 83.8%; CI therefore verifies the committed tiny fixture and admissions but cannot claim released-model coverage.

The package does not yet include masked-LM/classification heads, training/backward, quantization or a generic pooling API. It does not implement Laya's type embeddings, decision transformer, option scorer or action head; that remains the next milestone. No GPU was queried or used, and frozen evaluation artifacts/services were unchanged.
