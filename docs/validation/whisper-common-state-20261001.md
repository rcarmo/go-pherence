# Common encoder-state attribution — 1 October 2026

Original flash attention includes36 unmasked zero K/V positions beyond the1500 encoder rows. Matching that1536-key extent in Go removes the extra `Thank you.` on the tested JFK crop without changing text masks, generation stopping or audio context. The finding is limited to this fixture; no runtime option, quality acceptance or speed gain is published in this checkpoint.

## Diagnostic boundary

The input is the exact164,800-sample F32 gap-preserved crop from [the prior trace](whisper-vad-boundary-20261001.md), SHA256 `0f4302922d4f470606faf7bca41df64ade994a3fc6ff9774c485bb06155b4c4f`. Source weights are the pinned original turbo Q5 model. Both engines receive the same31 forced decoder inputs: the checked English transcription prompt plus the28 generated tokens ending at10.06s. Token selection cannot alter the replay prefix.

Original exports use a test-only copy of whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, compiled against the existing GGML Vulkan libraries. The production library and vendored source are untouched. Export/import occurs after the encoder and before cross-KV construction. Mel exports are channel-major128×3000; hidden values are contiguous1500×1280 time rows in both engines. Full logits are exported after each forced token.

Go uses the retained outputILP/decode4 encoder and accepted four-worker CPU decoder at `a50704b7`. Test-only plan factories can export layer states, substitute a GELU expression or append zero K/V rows. The final hidden result of fenced Go execution is checked against its complete encoder execution.

Original layer/node callbacks request downloads without modifying tensors. Original instrumentation is validated against the uninstrumented final hidden hash, rather than assuming downloads leave execution intact. Corrected layer and operator export runs match the plain both-disabled original hidden output exactly.

## Mel and hidden substitution

Original computes mel on the short PCM with its own right padding; Go explicitly pads PCM to480,000 samples before its checked frontend. The first128×3000 mel values differ by at most0.000020862, mean absolute0.00000003953 and RMS0.0000002689. Feeding original mel to Go barely changes the final EOT/timestamp scores.

Margins below are EOT minus repeated10.06s timestamp, after the identical31-token input prefix. Positive values favour EOT. These are raw logits, not independently selected free-decode outcomes unless specified.

| Encoder hidden source | Decoder | EOT margin |
|---|---|---:|
| Go retained encoder | Go |−0.176901 |
| Go encoder fed original mel | Go |−0.176902 |
| Original normal encoder | Go |+0.157191 |
| Original both-disabled encoder | Go |+0.137123 |
| Original normal encoder | Original |+0.290278 |
| Go retained encoder | Original |−0.060962 |

Swapping hidden values reverses the decision in both directions. Encoder-output differences materially affect this crop's decision. Decoder numerics still differ under common hidden inputs; no complete cross-engine decoder parity follows.

The original both-disabled configuration sets `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` and `GGML_VK_DISABLE_F16=1`. These flags disable particular arithmetic implementations, not all F16 tensor storage. They provide an attribution arm, not a fully numerically identical F32 reference.

## Layer and operator localisation

Final hidden arrays contain1,920,000 values. Baseline Go versus both-disabled original has RMS drift0.068497 and maximum4.468954. Layer snapshots show drift already in the stem, then a substantial increase after layer0. Introducing original-style GELU reduces stem error but does not restore the EOT decision alone.

Layer0 operator boundaries with the test-only original GELU expression:

| Boundary | Maximum absolute drift | RMS drift |
|---|---:|---:|
| Normalised stem |0.000590 |0.000044 |
| Q projection |0.000193 |0.000012 |
| K projection |0.000196 |0.000014 |
| V projection |0.000240 |0.000014 |
| Attention output |0.149104 |0.006403 |

The first large jump occurs at attention. Operator values before this boundary are not perfectly identical, so a common-Q/K/V test follows rather than attributing the jump to a kernel from these numbers alone. Complete per-layer/operator metrics are retained in the comparison JSON.

## Padded key extent

Original encoder source defines `n_ctx_pad = GGML_PAD(n_ctx, 256)`. With1500 rows, K/V caches expose1536 positions to `ggml_flash_attn_ext` with a null mask. Original intermediate type is F16. The source copies1500 positions and leaves the36 padded positions in zero-cleared storage. A diagnostic reads the actual padded tails after the last encoder layer: all92,160 F16 values across K and V are zero. The check leaves final hidden output bit-identical to the uninstrumented original.

Using identical original layer0 Q/K/V values in Go isolates storage/extent effects:

| Go attention input treatment | Maximum drift from original | RMS drift |
|---|---:|---:|
|1500 F32 keys |0.149111 |0.006402609 |
|1500 keys, K/V rounded through F16 |0.149123 |0.006402419 |
|1500 keys, Q/K/V rounded through F16 |0.149117 |0.006402420 |
|1536 keys,36 zero rows, F32 K/V |0.000598431 |0.000019503 |
|1536 keys,36 zero rows, F16-rounded K/V |0.000011444 |0.000000304 |

The extra zero keys contribute `exp(0)` terms to the softmax denominator even though their values contribute zero to the numerator. That differs from masking them out or attending to exactly1500 keys. The common-input experiment attributes almost all of this layer0 discrepancy to key extent; F16 storage accounts for much of the smaller residual. No tolerance is widened or hidden equality declared.

## Complete encoder compatibility controls

A test-only plan factory copies each layer's K/V to1536-row F32 scratch, appends36 zeros, and runs the existing attention shader with that extent. Q, actual K/V values and F32 arithmetic remain unchanged. The additional scratch payload is15,728,640 bytes shared across layers; two copy stages are inserted per layer. No production constructor or default is altered.

Padding alone with retained erf GELU changes the final Go margin to+0.084269. Free decoding returns only the main0–10.06s speech segment, without `Thank you.`. Repeating in a fresh process produces exactly the same1,920,000 hidden-value bits and the same free output. Feeding original mel gives the same segment and essentially unchanged scores.

Original uses `ggml_gelu`, the tanh-style approximation; Go uses erf-form GELU. A separate test-only shader copies the original Vulkan GELU expression. Combining it with padded attention moves the Go margin to+0.135036 and reduces hidden drift further:

| Go encoder control vs both-disabled original | Maximum hidden drift | RMS hidden drift |
|---|---:|---:|
| Retained1500-key/erf path |4.468954 |0.068497 |
|1536-key/erf diagnostic |4.372859 |0.034211 |
|1536-key/original-GELU diagnostic |0.026460 |0.000345 |

Both padded controls free-decode the same main speech segment as the original on this crop. Remaining decoder, storage, projection/reduction and activation errors are unresolved; no bit parity or acoustic ground truth is established. These controls are single-fixture experiments, not multilingual/word-timing qualification. No request timing, load-memory benefit or original-speed comparison is measured.

## Invalid diagnostics and checks

Three executed diagnostics are excluded from valid numerical conclusions:

- Initial forced replay used task token50359. Turbo transcription requires50360, verified against the previous consumed-token trace. Corrected own-Go replay reproduces the prior EOT/timestamp logits. Both wrong-prompt exports are labelled invalid.
- Initial GELU copy used `1 - 2/(...)` instead of original's `2 - 2/(...)`. Its large drift is invalid. Corrected shader compiles and validates before use.
- Initial operator callback returned false for an unsupported tensor view, cancelling graph computation and exporting a partial encoder result. Exit0 alone did not reveal this. Corrected callback skips unsupported/noncontiguous/inherited-name views; final hidden hashes match the plain original. Partial original output and decoder runs fed that output are excluded.

Several exact-text generator anchors and a private encoder-state field reference failed before execution. The corrected generator and harness are archived. Generator errors/compile failures are not native validation. The test-only copy and callback hooks are verified through final hashes and output sizes.

`final-comparison.json` checks prompt identity, input/array extents, finite values, instrumentation hidden hashes, repeated padding hidden hashes and free-decode results. Raw F32 exports are retained locally in `tmp/whisper-vad-parity-20261001`; a path/size/SHA256 manifest covers1817 artifacts totalling2,329,121,200 bytes, including labelled invalid exports. They are not committed. [Hashed repository evidence](../../benchmarks/speech-foundations/whisper-common-state-20261001/) preserves metrics, score reports, original source-copy hooks, Go harness, shader/source/SPIR-V, generators, wrapper, failed/successful logs and provenance sufficient to reconstruct the experiment from pinned assets.

The full copied original source retains its upstream provenance. Generated source does not enter production builds. Diagnostic-only Go code is removed from test discovery. The original library, vendor tree, runtime shader inventory and all production Go implementation files are unchanged.

The retained tree passes `make model-layout-check host-build host-vet host-test docs-check`, accepted decoder-opt-in affected race, and marked ARM64/RISC-V builds. Corrected diagnostic shader validation, host affected tests/vet and final native common-input attention execution pass. Final documentation checks follow this report.

A read-only supplied-facts judge confirms the single-fixture attribution limits and notes that padding is not the sole remaining numerical difference. It did not audit source or raw tensors. No independent source/acoustic approval is recorded.

Fresh @llama isolation uses CPU4/8GiB/no-swap, physical Intel GPU, heap4GiB for Go, Qwen-idle and host-available-memory≥6GiB guards. Native deadlines stay≤120s. Actual native/build/container drain precedes release; no services, resource allocations, Qwen LAN or Gemma state change.

## Qualification required

A proposed explicit original-key-extent compatibility mode must pass fresh multilingual and VAD/word-output comparisons, native padding/lifetime/cancellation tests, long-form/resume/fault gates and matched request measurements before adoption. GELU compatibility is a separate candidate with different arithmetic and its own gates. Neither is enabled here. The overall original-speed and independent quality objective remains unmet.
