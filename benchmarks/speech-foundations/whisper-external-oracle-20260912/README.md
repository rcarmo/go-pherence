# Pinned Transformers oracle comparison

Go and Transformers 4.57.1 generated identical tokens and timestamp segments for JFK, three MINDS-14 Portuguese/French clips and five seconds of digital silence. Three reference repeats were identical. The reference reproduces the poor Tiny transcripts and silence hallucination recorded in the earlier [corpus diagnostics](../vulkan-corpus-20260912/README.md).

This is a separate offline qualification tool. No Python, PyTorch or Transformers inference was added to the Go runtime, decoder or service path. The comparison uses CPU Go exports; earlier native tests independently established CPU/Vulkan output equality on these fixtures. No new GPU qualification is implied by this oracle run.

## Pinned inputs and reference environment

- Model: multilingual `openai/whisper-tiny`, revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. The existing weights/config/tokenizer/generation SHA256 checks are retained.
- Added feature-extractor asset: `preprocessor_config.json` from that revision, SHA256 `9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666`. The extractor loads this local file; it does not rely solely on library defaults.
- Transformers 4.57.1, tokenizers 0.22.2, PyTorch 2.14.0+cpu, NumPy 2.5.3; these exact versions are enforced. Hugging Face Hub 0.36.0 provides local loading only. PyTorch uses F32, eager attention, two compute threads, one interop thread and deterministic algorithms.
- Three Python wheels were downloaded from PyPI and checked against the published SHA256 values before extraction into the isolated `projects/whisper-stt/runtime/whisper-oracle-4.57.1` directory. No existing Python package/runtime was overwritten. The existing Python 3.12.14 environment supplies CPU PyTorch/NumPy. Wheel URLs/hashes and implementation file hashes are recorded in `environment.json`.
- `HF_HUB_OFFLINE=1`, local-only model loading and a fresh report path prevent accidental downloads/overwrites during the oracle run. The oracle test reads no secrets.

Public speech source provenance and references remain in [MINDS fixtures](../vulkan-corpus-20260912/fixtures.json) and [JFK evidence](../vulkan-public-speech-20260912/README.md). Canonical WAV is produced once by the Go-owned temporary FFmpeg adapter, then read by both implementations. Five-second silence is deterministic s16 zero PCM. Private audio and model weights are not included in the repository evidence.

Transformers is an Apache-2.0 diagnostic dependency; model-card licensing and source references are retained in prior reports. This work imports the installed reference package rather than copying a neural implementation into Go.

## Comparison method

`TestSpeechOracleExport` requires explicit opt-in and a new caller-named directory. It writes canonical PCM, Go exact log-mel features, full 1500×384 CPU encoder output, raw incremental decoder logits and metadata for five pinned fixtures. Logits are copied before Go's timestamp/suppression processors mutate them. Metadata retains the prompt, every decoder input, generated tokens including EOT, suppression policy and timestamp segments.

`scripts/whisper_pinned_oracle.py` independently:

1. Recomputes features from the same canonical PCM using the pinned HF extractor.
2. Runs the HF encoder on its own features and again on Go features, distinguishing frontend from encoder differences.
3. Computes both uncached teacher-forced and stepwise cached logits for the exact Go token prefix, from HF encoder output and from the exact Go encoder output.
4. Generates greedily with HF's timestamp and suppression processors from its own encoder and again from Go's encoder. It obtains suppression/initial-timestamp policy from the hash-pinned generation file and verifies Go's exported policy matches it.
5. Compares every generated token, EOT, and parsed `(start,end,text,tokens)` segment. A missing fixture, incomplete EOT or token/segment mismatch fails after preserving the report. Numerical errors are recorded as diagnostics; no newly invented cross-framework numeric budget is asserted.

All paths use the explicit language token, transcribe task, timestamps and a 96-token cap. High-level Whisper no-speech decisions, temperature fallback and beam search are not enabled in either path. This matches the checked Go policy; it does not benchmark an unrestricted production Transformers pipeline.

## Results

Three final Python runs produce byte-identical JSON reports. Across five fixtures and two greedy-decoder encoder inputs per run, all **30 generated-sequence and timestamp-segment comparisons match**. This is 92 generated tokens per five-fixture set, including EOT; the cached logit comparison uses 102 decoder input calls across the five fixtures. Raw timestamp token equality is checked, not inferred from text stripping.

| Fixture | Reference generated text | Token/segment match |
|---|---|---|
| JFK | And so my fellow Americans ask not what your country can do for you, ask what you can do for your country. | Exact in both reference paths |
| MINDS Portuguese row 0 | Bom dia, estou ligado por que os dados de informações sobram como eu posso ir depois de estar dinheiro e não me aconhe. | Exact in both reference paths |
| MINDS Portuguese row 1 | Com faspore transfridneir pra minha conta. | Exact in both reference paths |
| MINDS French row 0 | Josuait changer mon adresse. | Exact in both reference paths |
| Digital silence | you | Exact in both reference paths |

Portuguese WER remains 80% / 77.78%, French 40%, and default silence still hallucinates. Exact reference reproduction supports attributing these observed outputs to Tiny under the matched policy and input, rather than a Go-specific backend/decoder divergence. It does not establish model quality or exclude all implementation errors beyond these fixtures.

Maximum absolute differences across all five fixtures:

| Numeric comparison | Maximum absolute error |
|---|---:|
| HF vs Go log-mel | 1.2874603271484375e-5 |
| HF full encoder vs Go | 0.0029959678649902344 |
| HF encoder on Go features vs Go encoder | 0.003490447998046875 |
| HF cached logits on Go token prefix vs Go | 0.00026607513427734375 |
| HF cached logits on Go encoder output vs Go | 0.00014734268188476562 |

The encoder maximum is larger than the logit maximum; RMS/mean/99th percentile and per-fixture values are retained. These are actual differences, not bit-exact tensor parity. The uncached teacher-forced results are kept with explicit names and must not be confused with incremental cached decoding.

## Review and reproduction checks

The initial review identified that the first raw-logit comparison was uncached/full-prefix, feature-extractor defaults were not asset-pinned, versions were only partly enforced, greedy decoding did not isolate Go encoder input, and text-only reporting was insufficient. The final tool adds cached per-step comparisons, local preprocessor hashing, exact version gates, both decoder inputs and explicit timestamp segments. Independent generation policy is checked against the pinned file instead of trusting exported Go values.

Three helper unit tests cover numerical summaries, mismatched/nonfinite arrays, first token difference, valid timestamp segmentation, missing EOT and malformed timestamp spans. They pass without model loading. All three full reference repeats pass. A second Go export via the make target reproduced all **25 files byte-for-byte**, including metadata, canonical PCM and logits. A Go export run passes one opt-in test; default-disabled export skips before reading models. Focused Go regressions/vet/amd64 builds pass; full-tree/backend and Whisper arm64 FFT errors match the baseline, and race compilation lacks `gcc`.

No services were started. The LLM and both speech units stayed inactive/MainPID zero; before/after swap counters did not change. This phase used CPU inference, not a performance comparison or a new GPU test. Timings and thermal/energy claims are outside scope.

## Run the tools

Export requires a new output directory, verified local Tiny assets, MINDS source cache and pinned JFK file. The make target performs only Go inference and FFmpeg conversion.

```sh
export GO_PHERENCE_WHISPER_TINY_DIR=/path/to/verified/whisper-tiny-169d4a4
export GO_PHERENCE_MINDS_FIXTURE_DIR=/path/to/MINDS-cache
export GO_PHERENCE_WHISPER_JFK_PATH=/path/to/pinned/samples/jfk.wav
export GO_PHERENCE_ORACLE_EXPORT_DIR=/path/to/new/exports
GOMAXPROCS=2 CGO_ENABLED=0 make speech-oracle-export-check

# Use the isolated, exact-version Python environment described above.
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
/path/to/python scripts/whisper_pinned_oracle.py \
  --model "$GO_PHERENCE_WHISPER_TINY_DIR" \
  --exports "$GO_PHERENCE_ORACLE_EXPORT_DIR" --output /path/to/new/report.json
/path/to/python scripts/test_whisper_pinned_oracle.py
```

[Evidence](evidence.json), three reference reports, input hashes, environment provenance and command logs are included. Large exported tensors remain outside the repository; the input manifest allows regeneration checks. Turbo/larger-model quality, faithful no-speech policy, diarization, quantisation, recovery and whole-job qualification remain unfinished. Nothing was pushed, deployed or restarted.
