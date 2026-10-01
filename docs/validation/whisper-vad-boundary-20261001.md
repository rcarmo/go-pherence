# VAD crop and generation-boundary trace — 1 October 2026

The extra `Thank you.` appears after Go and the original reach the same28 generated tokens on identical cropped JFK PCM. Original scores favour EOT; Go scores favour a repeated timestamp, then `Thank`. The responsible mel/encoder/decoder operation is not isolated. No crop margin, stopping rule, text filter or default is changed.

## Inputs and unchanged inference path

Source is mono16kHz JFK,176,000 samples. The retained Go native-Silero gap-preserving output supplies the speech/audio sample spans; diagnostic files copy those exact PCM samples. Canonical PCM16-to-F32 conversion uses `int16 / 32768`, with raw-PCM hashes recorded. No re-encoding, interpolation or noise suppression occurs.

Go uses the accepted original-Q5 outputILP/decode4 Vulkan encoder and explicit four-worker CPU decoder, pinned source `d93515783780e68d0b1a725f9ca5b0910c1d1f75`. Original uses pinned whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, original turbo Q5 file, greedy English, Vulkan/flash and the reusable-context diagnostic harness. The original comparison receives the same raw diagnostic PCM with **VAD disabled**: it isolates the retained-audio/generation boundary, not a new VAD timing baseline.

The real gap-preserved audio is samples `[5152,169952)`,164,800 samples (10.3s). Its source offset is0.322s. The four compact speech spans are `[5152,36320)`, `[52256,70624)`, `[86048,122848)`, `[130592,169952)`, totalling125,696 samples (7.856s). The same retained Go PCM source supplies both layouts.

Each Go input is encoded once and reused by two fresh CPU decoder states: baseline checked timestamp decode and a test-only close-timestamp stop rule. Timings exclude shared encoder setup and are not request comparisons. Gap and compact cases receive final five-repeat decoder tests per rule; other edge controls have one execution per rule. Every native invocation has a≤120s deadline.

## Real-audio end guard

Original `whisper.cpp` completes a decoder when `has_ts && seek + seek_delta + delta_min >= seek_end`, with `delta_min=10` centisecond frames. The Go diagnostic adds the corresponding test-only condition after a complete closed text segment: `2*timestampIndex + 10 >= validSamples/160`. It does not change logits, timestamp masks or audio.

On the gap crop, the main speech closes at10.06s,240ms before the10.3s end. The100ms guard cannot stop at that point. Go then emits `Thank you.` through10.3s. The guard saves the final EOT decode call (36→35 calls) but leaves both segments unchanged. Five final repetitions preserve this output with and without the guard.

On compacted speech, both rules produce two identical segments and33 calls. The diagnostic guard preserves segment outputs on every tested edge control; one100ms-prefix control saves a final call. This guard is not a fix for the observed disagreement and is removed with the diagnostic source.

## Leading-edge sensitivity

Times below are relative to each diagnostic input, before any original-timeline mapping. Each control has fixed trailing bound169952 unless noted.

| Control | Original sample range or preparation | Go result without end guard |
|---|---|---|
| Gap-preserved crop | `[5152,169952)` | Main0–10.06s; `Thank you.`10.06–10.30s |
| Compact four spans | Copy only listed speech spans | Two speech segments0–5.18s and5.44–7.60s; no extra text |
| Keep original leading audio | `[0,169952)` | Main0–10.38s; no extra text |
| Keep only original trailing audio | `[5152,176000)` | Main0–10.06s; `Thank you.`10.06–10.68s |
| Full JFK | `[0,176000)` | Main0–10.40s; no extra text |
| Add30ms real leading context | `[4672,169952)` | Main0–10.08s; no extra text |
| Add100ms real leading context | `[3552,169952)` | Main0–10.40s; no extra text |
| Snap start to5120 (2ms earlier) | `[5120,169952)` | Two speech segments0–7.18s and7.88–10.06s; no extra text |
| Synthetic322ms zero prefix |5152 zeros + exact gap crop | Main0–10.38s; no extra text |

The synthetic prefix is a diagnostic only; it is not installed as padding or given a source-audio timestamp. Small shifts change the generated segmentation as well as the extra text. These single-fixture controls cannot establish a safe context margin, sample-grid snapping policy or independent word-timing accuracy. They reproduce sensitivity to the leading edge without selecting a crop that merely hides the failure.

## Original on identical PCM

Five original requests each on the exact gap crop return only the main0–10.06s speech segment. Five compact-input requests return the same two speech segments/text/content-token IDs as Go:0–5.18s and5.44–7.60s. This verifies that the inputs can produce matching output while preserving the gap-crop disagreement.

Original gap runs with integer-dot disabled, F16 disabled, and both disabled still return the main text without `Thank you.`. These arithmetic switches do not make every original/Go operation equivalent and do not identify a root cause. No speed advantage or accepted precision change is inferred.

## Token prefix and scores

A read-only original `logits_filter_callback` records generated prefixes and four logits without modifying them. The callback runs before later timestamp masks; it is not a post-mask token-selection observer. Original's last observed prefix contains28 tokens, ending with timestamp503 (10.06s). Its returned segment contains that closing timestamp and no extra text. The prefix matches the first28 Go-generated tokens exactly.

At the first token following that prefix:

| Raw score | Original normal | Original integer-dot+F16 disabled | Go |
|---|---:|---:|---:|
| EOT |10.9662 |10.9777 |10.8156681 |
| Repeated10.06s timestamp |10.6770 |10.7465 |10.9925690 |
|10.30s timestamp |4.2050 |4.26435 |4.46491003 |
| `Thank` token1044 |−3.03665 |−2.99791 |−2.82966566 |

Go selects the repeated10.06s timestamp. After consuming it, Go's `Thank` score is8.3207016 and EOT0.839388251; the consumed tail is `[50868,1044,291,13,50880]`: repeated timestamp, `Thank`, `you`, punctuation and10.30s timestamp. Original callback/returned outputs are consistent with earlier EOT termination; it has no later recorded prefix and never returns that tail in any of the five runs. EOT is not directly recorded as a sampled token by this callback.

The normal EOT-minus-repeat margin is+0.2892 original versus−0.1769009 Go. The both-disabled original margin is+0.2312. Original callback and Go raw-forward score locations are not identical complete masking stages, but neither raw candidate is suppressed at the inspected timestamp state by the source pairing rules. Other heuristics and model-operation details still need investigation; full logit/operator numerical parity is not established.

This locates the visible generation decision, not a specific mel, encoder, cross-attention, projection or decoder defect. The close-timestamp guard cannot change the preceding margin, and text filtering would conceal the unexplained divergence. No independent acoustic reference is used here; original output is an engine comparison, not human ground truth.

## Verification, corrections and retained source

The evidence parser verifies pinned inputs, equal28-token prefixes, stable original output across five repeats, stable Go gap/compact output across five decoder repetitions per rule, unchanged guard outputs and actual stopped container states. Native encoder cleanup restores allocations and leaves no in-flight work.

Diagnostic setup corrections are archived: a missing private packed-only constructor argument, followed by supplying the wrong packed-only boolean, failed before inference; the corrected constructor uses the verified metadata-only FFN source. Invoking an archived `.ts.txt` wrapper produced no report; execution then used its copied `.ts` entrypoint. A build attempted `gofmt` inside the read-only mounted source; formatting moved to the host before the successful score-test build. Failed or reportless invocations do not count as native result gates.

The source-reading review delegate timed out after60s. A later read-only judge completed on supplied attribution facts, without auditing raw sources, and supports the narrow decision-boundary finding. Its statement that VAD is ruled out is not adopted broadly: identical-PCM runs bypass VAD detection during inference, while the crop originates from the VAD pipeline. Crop selection and the integrated pipeline remain part of the unresolved quality issue.

Both diagnostic Go files and the test-only end guard are removed from test discovery/runtime. Original source/library and all production Go code remain unchanged. [Hashed evidence](../../benchmarks/speech-foundations/whisper-vad-boundary-20261001/) preserves generators, sample-range/hash pins, raw reports, prefix/score callback harness, copied guard/native tests, parser, settings, failed/successful logs, tool/binary provenance and states. No source recording, raw PCM, model weights or binaries are committed.

The retained tree passes `make model-layout-check host-build host-vet host-test docs-check`, affected Vulkan/Whisper race with accepted CPU row scheduling enabled and marked ARM64/RISC-V builds. Final documentation checks run after this report. Defaults, VAD segmentation/gap policy, generation masks/stop rules, resource allocations and services remain unchanged.

Fresh @llama coordination holds competing experiments/builds/restarts. CPU4/8GiB/no-swap, heap4GiB for Go, physical Intel GPU, Qwen-idle and host available memory≥6GiB guards continue throughout execution. Successful inference/gate containers exit0/noOOM and actually drain before explicit release; failed setup containers also drain. Qwen LAN and Gemma state are preserved.

## Next boundary

Capture or independently compare the model-operation states producing the same-prefix EOT/timestamp logits, beginning with exact PCM→mel and encoder outputs and then decoder states under a common input. The source switches alone cannot supply a numerically matched baseline. A verified cause is needed before choosing original-compatible arithmetic or changing VAD context. Original-speed parity, independent acoustic/word timing, long-form/resume and fault qualification remain open.
