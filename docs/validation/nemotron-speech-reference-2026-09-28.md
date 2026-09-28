# Nemotron ASR and diarisation: bounded CPU reference

Two user-approved NVIDIA checkpoints ran with a pinned Hugging Face Transformers CPU reference against the repository's 11-second mono JFK WAV. This establishes local loading and one-recording execution. It does not establish native Go inference, transcription accuracy, diarisation error rate, streaming-cache correctness or production readiness.

## Assets and licence

| Role | Repository revision | Local safetensors file | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| Transcription | [`nvidia/nemotron-3.5-asr-streaming-0.6b@ea30d66debe3740a08b573244286791d423d6b3e`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e) | `checkpoints/nemotron/asr/model.safetensors` | 2,552,062,944 | `9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174` |
| Speaker attribution | [`nvidia/Nemotron-3-Diarization@a435e9867d79e789e90053f9b6d6834053af564a`](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/a435e9867d79e789e90053f9b6d6834053af564a) | `checkpoints/nemotron/diarization/model.safetensors` | 396,954,592 | `c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6` |

Both files match the pinned Hub response's linked SHA-256 identifier and published length. Configs, processors and the ASR tokenizer were fetched from the same revisions. All downloaded files are under the repository's ignored `checkpoints/` directory and are not distributed in Git. Both model cards identify OpenMDW-1.1; retain applicable licence, copyright and origin notices before any checkpoint redistribution. The ASR Hub API exposes `license:other` while its card names OpenMDW-1.1, so check the complete terms for any deployment.

ASR config identifies `Nemotron3_5AsrForRNNT`, a 24-layer FastConformer encoder and RNNT decoder. Diarisation config identifies `Nemotron3DiarizationForAudioFrameClassification`, 31 layers and an eight-speaker head. Both processors expect 16 kHz input and 128 mel features. The diarisation processor has a 160-sample feature hop and offers `low_latency`, `very_low_latency` and `ultra_low_latency` modes; the run below used offline processing. These config fields do not verify numerical frontends or streaming cache behaviour.

## Reference execution

The isolated local environment used Python 3.13.14, CPU-only PyTorch `2.9.1+cpu`, and Transformers `5.18.0.dev0` built from [`huggingface/transformers@2ceac527a04265689dc17111619b4e99b9b4cd9a`](https://github.com/huggingface/transformers/tree/2ceac527a04265689dc17111619b4e99b9b4cd9a). The revision includes both models; released Transformers 5.17 includes the ASR model but did not contain the diarisation modelling module at inspection time. The environment and probe source were kept outside Git under `/workspace/tmp/nemotron-reference/` and `/workspace/tmp/nemotron-reference-probe.py`. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` forced local CPU model loading and inference.

Input: `testdata/jfk.wav`, SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`, 176,000 samples of 16 kHz mono audio (11 s). ASR used `language="en-US"`; both models used `AutoProcessor.from_pretrained(local_path, local_files_only=True)`, the model-specific `AutoModel` loader, `.eval()` and `torch.inference_mode()`.

- ASR decoded: “And so my fellow Americans ask not what your country can do for you.  Ask what you can do for your country. ” The processor output is transcribed text only; this run did not measure timestamps or streaming partials. Reported in-process load plus inference was 1.298 s; process peak RSS was 3,013,684 KiB.
- Diarisation produced logits of shape `[1, 1101, 8]`; `processor.extract_speaker_dict(logits, attention_mask)[0]` returned speaker 0 spans `[0.28, 2.28]`, `[3.27, 4.56]` and `[5.36, 10.63]` seconds. Reported in-process load plus inference was 0.407 s; process peak RSS was 893,744 KiB. These times include warm page-cache conditions and are not controlled performance benchmarks.

The processor also emitted a nonfatal Transformers `image_like_kwargs` docstring diagnostic; ASR generation warned about its default `max_length`. The probe did not test other languages, multiple speakers, overlap, silence, chunk seams or cancellation. One output on a known recording cannot establish word error rate, speaker labels or diarisation error rate. Do not use the exact transcript string or model-output hashes as a generated-signal acceptance gate.

Raw local evidence: `/workspace/tmp/nemotron-{asr,diarization}-probe.log`. No Python/Torch subprocess was added to the Go runtime or speech job.

## Next gates

1. Pin small independent frontend and model-boundary fixtures with input provenance, numerical error distributions and calibrated tolerances; independently label a multi-speaker/overlap cohort for turn and timestamp scoring. Keep model-output hashes out of acceptance tests.
2. Verify ASR offline and streaming cache/partial-transcript behaviour and diarisation frame origin, lookahead, speaker-cache transitions and chunk seams against the pinned reference. Define explicit maximum audio length, language selection and eight-speaker limits.
3. Implement separate native Go scalar inference and checked SIMD dispatch only after the reference contract is reproducible. An opt-in speech-job provider must preserve plain transcript/VTT on speaker attribution failure; existing Whisper/Community-1 defaults must not change silently.
4. Measure independent transcription and diarisation quality, CPU/ARM64/RVV native execution where available, concurrency, cancellation, memory and end-to-end latency before considering production admission.
