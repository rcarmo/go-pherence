# Gemma 4 Plan 9 post-change profiling decision

Date: 2026-09-18

Code checkpoint: `2e76bec2fad81f6069d54229f79460a558017972`

Final review tree: `ee9889f83a618debc54d2100d5747c11a213e7f0`

## Profile availability

A fresh frozen-request CPU profile could not be collected on this host because the required model file, `gemma-4-E4B_q4_0-it.gguf`, is not present under `/workspace` or `/srv`. No post-`2e76bec2` profile is retained in the repository: the newest retained whole-request profile is `fused-8x16-prefill-20260809.pprof`, collected before the Plan 9 Q4, GELU, and RoPE promotion.

This is an environment/evidence limitation, not a runtime failure. The reproducible command remains:

```sh
GOMAXPROCS=6 taskset -c 0-5 env \
  GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  GO_PHERENCE_GEMMA4_PREFILL_ONLY=1 \
  GO_PHERENCE_GEMMA4_MAIN=/path/to/gemma-4-E4B_q4_0-it.gguf \
  GO_PHERENCE_GEMMA4_PREFILL_CPU_PROFILE=benchmarks/gemma4-gap/audit/gemma4-plan9-post-change.pprof \
  go test ./model -run '^TestGemma4RealCPUGap124x48$' -count=1 -v

go tool pprof -top -nodecount=40 \
  benchmarks/gemma4-gap/audit/gemma4-plan9-post-change.pprof
```

Do not label cross-compilation or an older profile as post-change runtime evidence.

## Softmax and argmax decision

No additional assembly is justified.

The last applicable dynamic inventory attributed only 0.53% inclusive CPU time to attention softmax. Argmax was not sampled materially. Even eliminating both completely would therefore have a sub-one-percent whole-request ceiling in the available evidence, while adding new numerical and ABI contracts for transcendental handling, NaNs, infinities, stable reduction, and first-maximum ties.

The promoted mechanisms address the measured costs instead:

- paired Plan 9 Q4 projection beat the retained cgo path by 4.2% at the median;
- exact Plan 9 FP16-table GELU was approximately 9.9 times faster;
- partial RoPE was approximately twice as fast;
- the frozen three-run prompt median reached 89.993 tok/s and cleared the 89.405 tok/s target.

Until a future whole-request profile shows either operation as material, softmax remains stable Go orchestration around the existing Plan 9 vector arithmetic, and argmax remains scalar Go with first-maximum semantics.

## Final coverage disposition

| Operation | Final disposition |
| --- | --- |
| Q4_0 x Q8_0 projection, full and tail tiles | Plan 9 assembly in production; portable and unsupported-CPU fallbacks retained |
| Exact Q8 activation preparation | Plan 9 AVX2 block quantiser; scalar and non-amd64 fallbacks retained |
| Exact FP16-table GELU x up | Plan 9 AVX2/F16C in production; exhaustive scalar oracle retained |
| Partial RoPE | Plan 9 AVX2 in production; malformed-input, scalar-tail, and non-amd64 fallbacks retained |
| Q6_K x Q8_K, F32/BF16 GEMV/GEMM, vector and RMSNorm arithmetic | Existing Plan 9 coverage retained |
| Attention softmax | Deliberately retained in Go; measured 0.53% inclusive before promotion and no contrary profile evidence |
| Argmax | Deliberately retained in Go; not sampled materially and first-maximum contract preserved |
| Logit soft-cap, token suppression, embedding gather, checkpoint/KV handling | Retained Go post-processing or memory orchestration; no measured arithmetic gap |
| SiLU/tanh GELU variants | Retained scalar/transcendental fallback; not the frozen E4B exact-GELU path |

## Validation at final review

The focused kernel packages pass on the final review tree:

```text
ok github.com/rcarmo/go-pherence/backends/simd/runtime
ok github.com/rcarmo/go-pherence/internal/ggmlfp16
ok github.com/rcarmo/go-pherence/loader/gguf/llamaq4plan9
ok github.com/rcarmo/go-pherence/loader/gguf
```

The wider `model` package currently reports three failures outside this audit's changed paths: a truncated GGUF test fixture (`data offset 32 exceeds file size 24`), a visible CUDA device with an unavailable NVIDIA runtime, and an MTP verifier batch/single precision delta. These arrived with the concurrently merged `origin/main` work and do not invalidate the focused Plan 9 kernel gates.

## Conclusion

The Plan 9 coverage audit is closed. All measured material Gemma CPU arithmetic gaps have production assembly paths and fallbacks. Softmax and argmax are consciously excluded on the available performance budget rather than added for nominal coverage. A future profile may reopen that decision only if it supplies material whole-request samples from the frozen model and request.
