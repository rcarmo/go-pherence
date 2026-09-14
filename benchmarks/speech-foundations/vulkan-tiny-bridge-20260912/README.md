# Trained Tiny encoder and checked PCM bridge

The resident F32 encoder passed pinned Whisper Tiny weight tests, full encoder shape execution and short-input cross-KV/decoder-logit comparisons on Iris Xe. Checked PCM transcription now accepts an optional caller-owned resident encoder. These tests do not establish speech accuracy, WER, long-file quality or whole-job realtime factor.

## Checkpoint provenance

Public model: `openai/whisper-tiny`, revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. Download URL: `https://huggingface.co/openai/whisper-tiny/resolve/169d4a4341b33bc18d8881c4b69c2e104e1cc0af/model.safetensors`.

- Weight file: 151,061,672 bytes, SHA256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`. The download hash matched the repository's linked object hash; the native test rehashes the file before use.
- Safetensors header: 19,104 bytes; 167 F32 tensors, metadata `format=pt`. `LoadModelSourceChecked` validated all tensor metadata and copied owned finite weights; the source mmap was closed before native inference.
- Config, tokenizer and generation JSON match the existing [generation manifest](../../../docs/speech-generation-manifest.json) hashes. The trained test pins model and config hashes; tokenizer/generation are provisioned but not used for the short feature/logit comparison.
- The pinned model card declares Apache-2.0. Consulted model-card SHA256: `57a3bbbbf1e79369e4d1ced790812a1723b43a256c1b43fc70cc2f3339ae9881`. No weights are committed or attached. Local cache: `projects/models/whisper-tiny-169d4a4` outside the Go repository.

## Trained numerical qualification

Three final repeats pass nine test/subtest events, no failures/skips. Short-input comparisons total 54 cases and 760,545 values. The encoder retains Tiny's trained width 384, four layers, six heads of width 64 and FFN width 1536, using a shortened 34-frame synthetic feature input for bounded scalar comparison.

| Check | Maximum absolute difference | Fixed budget |
|---|---:|---|
| Encoder boundaries/final | 2.384185791015625e-5 | `1e-3 + 2e-4*abs(reference)` |
| Decoder cross-KV | 1.4066696166992188e-5 | `2e-3 + 3e-4*abs(reference)` |
| Prompt logits | 8.7738037109375e-5 | `1e-2 + 5e-4*abs(reference)` |

Budgets were declared before the first trained run and were not relaxed. Encoder reference is the earlier scalar direct convolution/linear/norm and independent attention/erfc implementation. Cross-KV and logits use the existing Go decoder with CPU-reference versus GPU-encoder hidden states. The three forced prompt inputs are SOT, English and transcribe. Raw top-1 output tokens match at each step (`50362`, `50358`, `11`) across all repeats. This is a conditional prompt-logit comparison on synthetic features, not text decoding or multilingual quality.

### Full encoder shape

Each final repeat constructs the trained encoder at 3,000 input frames and runs three forwards. All nine forwards produce finite `[1500,384]` outputs, 576,000 values each. Outputs match bit-for-bit within and across repeats, with SHA256 `08125fb8596d86c8cd63da8f0187a41cc5ad173e0e9c80fcb31ad1035a0af488`.

Logical resident weight extent is 32,839,680 bytes and scratch extent 28,608,000 bytes; native allocation padding and pipeline/driver memory are separate. The graph has six plans/54 stages. Recorded wall times include each full Forward's input upload, layer submissions/waits, output download and finite scans; construction is recorded separately. Raw values are in `metrics.json`. The initial run was about 294/261/260 ms for three forwards, but no balanced old/new encoder benchmark or ASR realtime claim is made.

The full-shape input is synthetic sinusoidal features, with no scalar full-shape oracle. These runs establish execution, finiteness and repeatability for trained full-size Tiny, not reference accuracy at the full length. No large-v3/turbo checkpoint ran.

## Checked PCM integration

`PCMTranscribeOptions.VulkanEncoder` selects the resident encoder for every exact-feature window. Nil preserves the existing CPU path. The resident encoder must have the configured `MaxLength` and matching mel/model/layer/head/FFN geometry; it must be paired with a decoder from the same checkpoint. Geometry admission cannot prove weight identity.

The caller creates and owns the resident encoder. The PCM path never creates/closes it and never silently falls back to CPU. The host `w.Encoder` can be released or nil after native construction. The decoder and model config remain required and validated. Both normal and resident paths use the same frontend, cross-KV construction, decoder, timestamp policy and window/callback mapping.

A caller must exclude `Close` and other use of the resident encoder for the transcription call. Timeouts/cancellation can retain native work; errors preserve `ErrVulkanInFlight`, and the caller must drain before reuse/close. Previously emitted windows remain valid; a failed window is not emitted. No retained native work is represented as successful output.

Native bridge fixtures pass three repeats (27 total native encoder-suite pass events). They process two windows through exact frontend → Vulkan encoder → CPU cross-KV/decoder → checked callbacks. Output is a toy EOT/empty-transcript case, not trained speech quality. They verify:

- CPU and GPU callback/window results match.
- Poisoning host encoder weights, then setting `w.Encoder=nil`, does not affect resident execution; the nil/default CPU path fails on poisoned data.
- Callback errors propagate, and a closed resident encoder rejects before audio reads.
- Host decoder validation still rejects missing/malformed tensors. Model/lifetime/geometry checks occur before source access.

Source review identified that the first bridge version still required full host encoder storage. The validation split now checks only decoder/config in resident mode, while `validatePCMModel()` retains full checking for legacy/loader callers. A follow-up review found no scoped decoder-check, nil-dereference or legacy-validation regression. Weight identity remains the caller's responsibility.

## Verification and resources

Offline focused selection: nine top-level tests/29 events; 30 shuffled repeats: 870 events. Existing speech/frontend/checked-loader/generation, affine/media and Vulkan regressions pass. Affected vet and amd64 builds pass. Full-tree/backend compilation and Whisper arm64 FFT errors match the prior baseline. Race build cannot start because `gcc` is absent. Default-disabled native tests skip before any model/GPU load.

Compute and LLM-stop authorisation remain in effect. The LLM and both speech services stayed inactive/MainPID zero. Before/after swap counters did not increase. Model assets were downloaded separately; no private audio was used. Resource snapshots are not continuous telemetry. No service restart, push or deployment occurred.

## Reproduction

Provision the pinned public files outside the repository and verify the hashes. The test performs no download and uses no credentials.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TINY_DIR=/path/to/verified/whisper-tiny-169d4a4
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-trained-tiny-check

# Include three full-shape forwards in addition to short reference comparisons:
GO_PHERENCE_TEST_VULKAN_FULL_TINY=1 GOMAXPROCS=2 CGO_ENABLED=0 \
  make speech-vulkan-trained-tiny-check
```

[Evidence](evidence.json) and raw numerical/resource logs are included. Real PCM speech transcription/WER, external-framework parity, turbo/full-size quality, quantisation, device-loss recovery and end-to-end job performance remain unfinished. Strict SincNet still has four failures.
