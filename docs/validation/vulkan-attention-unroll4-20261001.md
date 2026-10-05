# Exact-order F32 attention unroll4 — 1 October 2026

The explicit headDim64 attention unroll improves the full-shape kernel median by3.2% and fresh trained request medians by0.9–2.0%. Retained hidden values and expanded recording outputs remain exact. This is a small native Go improvement, not matched original-engine speed acceptance. Defaults and services remain unchanged; unqualified activation quantisation is not extended.

## Candidate and numerical contract

`NewVkAttentionKey32ScoreILPUnroll4F32` replaces the variable-bound Q·K channel loop with groups of four statically indexed channel iterations. It keeps the same two independent score accumulators and increasing-channel FMA order for each key. The online softmax,32-key reduction, output FMA recurrence, shared layout and all barriers remain unchanged.

The shader requires headDim64 exactly. Host operator stage admission rejects every other dimension; there is no alternate-kernel fallback. `NewVulkanEncoderOriginalQ5AttentionUnroll4` and benchmark arm `vulkan-original-q5-attention-unroll4` explicitly combine this attention kernel with retained original packed Q5 FFN/F32 activation arithmetic. They do not select the FC1 quantised mode. Source/file value admission, packed-only benchmark loading, decoder and word alignment are unchanged.

Local size stays16×16, shared storage14,528 bytes, storage bindings4 and push range20 bytes. Native encoder weight/scratch bytes, plan count and stage count match the baseline. Model-free tests cover contract reflection, head-dimension refusal, cancelled construction, shared-memory preallocation bounds, invalid encoder admission and unchanged defaults.

The generated shader is included in the closed baseline-compatible regeneration inventory, now29 shaders. No opcode/feature admission was broadened. The optional integer-dot shaders remain separate and unchanged.

## Structural trial results

Physical Intel Iris Xe, same process, retained score-ILP baseline, five alternating-order timed executions per arm on1500 queries/1500 keys/20 heads/headDim64. All full-shape outputs match baseline bits and guards pass;17-query/37-key tails also pass. No timed sample was discarded.

| Trial | Baseline median (ms) | Candidate median (ms) | Change | Decision |
|---|---:|---:|---:|---|
| Four-channel score unroll | 67.154 | 65.001 | −3.21% | Retain explicit candidate |
| Eight-channel score unroll | 67.050 | 65.131 | −2.86% | No added gain |
| Full64-channel score unroll | 67.015 | 64.927 | −3.12% | No useful gain over smaller unroll |
| Remove one potentially redundant barrier | 66.975 | 66.514 | −0.69% | Too small; preserve conservative barrier |
| Four-channel unroll + barrier removal | 66.920 | 64.758 | −3.23% | No useful added gain; preserve barrier |
| Transposed shared K | 66.994 | 67.471 | +0.71% | Reject |
| Shared K row stride65 | 67.011 | 244.838 | +265.37% | Reject |
| Four-channel unroll + transposed K | 67.181 | 128.386 | +91.10% | Reject |

The padding/transpose regressions are native measurements, not an assumed bank-conflict improvement. None changed arithmetic. Temporary diagnostic shaders/tests are archived only; runtime retains just unroll4. An initial hypothesis about serial exponentials was corrected after reading the actual retained shader: exponentials were already parallel; no softmax-prefix candidate was implemented or qualified.

## Five-repeat trained requests

Fresh baseline `vulkan-original-q5-source`, candidate unroll4, same pinned original-Q5 stored values, physical device, four CPU threads and greedy language settings. Each arm is a separate guarded process with five retained requests. First request is included. Baseline precedes candidate for PT/FR/VAD/groups; JFK candidate precedes fresh baseline. These orders are not randomised and the modest gain should not be generalised to other devices/workloads.

| Fixture | Baseline median (s) | Candidate median (s) | Change | Every returned output |
|---|---:|---:|---:|---|
| JFK English | 6.918 | 6.825 | −1.35% | Exact |
| MINDS Portuguese row0 | 6.871 | 6.810 | −0.89% | Exact |
| MINDS French row0 | 6.372 | 6.247 | −1.97% | Exact |
| JFK native Silero VAD + words | 8.142 | 8.036 | −1.30% | Exact speech/audio windows, segments and word times |
| Two speech groups, VAD + words | 16.355 | 16.211 | −0.88% | Exact speech/audio windows, segments and word times |

Loading, preparation and five-request full-arm totals remain separate in raw JSON. No original-engine process/request timings were collected in this window. Whole-Go improvements do not establish original Q5 precision/boundary equivalence.

## Expanded output and hidden gates

Five candidate requests per additional fixture match every returned baseline output from the expanded-quality evidence: MINDS PT rows1/2 (including43.76s two-window recording), podcast20s without VAD, and podcast20s with native Silero VAD/words. The podcast regression previously seen in FC1 quantisation does not occur here; all segment text/content tokens/times and word boundaries are baseline-exact. These additional baselines come from the earlier expanded-quality window and are **output gates, not fresh speed pairs**. Their raw timing differences are retained but not claimed as matched gains.

Five silence/VAD/word requests return no windows. No broader independent acoustic accuracy is inferred. Existing Go-vs-original timing gaps from [expanded quality](whisper-fc1-expanded-quality-20261001.md) remain; this exact-Go candidate does not fix them.

A pinned complete encoder on synthetic mel produces1,920,000 bit-identical hidden values. Three cancellation checkpoints, native drain if needed, fresh reuse and complete native cleanup pass; latest final run records1295 checkpoints. Native stats remain unchanged. The independent full-matrix float64 attention oracle passes six headDim64 shapes ×three input signs, including4096 sequence bounds,32 heads and tails, using the existing2e-5 absolute/relative diagnostic budget. This budget was not widened. Exact baseline/native hidden comparisons remain separate from that analytic reference budget.

## Verification and isolation

`make model-layout-check host-build host-vet host-test docs-check`, whole-tree race, independent marked ARM64/RISC-V builds, latest affected race and affected vet/tests pass. All29 stored/embedded shaders validate and regenerate with normalised identity; checker tests plus optional integer-dot checker tests pass (7 tests/86 assertions). Final native independent and complete hidden/cancel tests pass on the retained tree.

Runner retained CPU4/8GiB/no-swap, heap target4GiB, physical Intel render node, no network/read-only root, Qwen idle and host available memory≥6GiB guards. Qwen LAN/Gemma/services/resources/defaults are unchanged. Native invocations had≤120s test deadlines. Outer tool timeouts were followed by continued guards and actual container exit0/noOOM inspection/removal; no release relied on the outer timeout.

A first benchmark invocation omitted the new mode from a diagnostic legacy allowlist and failed before inference. A wrongly named native reference helper caused compile failure; one old test binary then reported no tests. That zero-test invocation is explicitly not a passed native gate. After correction, the named independent reference test executed18 cases and passed. These failures are retained as evidence. No independent review approval is claimed.

[Hashed evidence](../../benchmarks/speech-foundations/vulkan-attention-unroll4-20261001/) retains rejected source/SPIR-V, exact diagnostic harness, raw timings, all trained/expanded JSON and environments, container states, full gates, shader report, comparison script, runner and provenance. No rejected layout/barrier runtime prototype remains.

## Remaining goal

Attention remains around2s per encoder, and FC2 plus Q/K/V/O are still substantial F32 costs. Continue structural/exact-order work and qualify on the expanded output set rather than broadening failed quantisation. Matched original Vulkan+flash+VAD speed, independent acoustic/word timing, long-form/resume and fault coverage are still open. The overall goal is unmet.
