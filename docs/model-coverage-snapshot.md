# Model coverage snapshot

| family | status | covered | pending | coverage | references | runtime | execution | parity | readiness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lfm2_moe | metadata_schedule_inspector_coverage | 63 | 2 | 96.9% | 12/0 (100.0%) | 19/2 (90.5%) | 0/2 (0.0%) | 2/0 (100.0%) | 5/0 (100.0%) |
| qwen3_tts | metadata_token_prompt_inspector_coverage | 66 | 5 | 93.0% | 9/0 (100.0%) | 20/5 (80.0%) | 0/5 (0.0%) | 2/0 (100.0%) | 5/0 (100.0%) |

# Runtime roadmap

## lfm2_moe runtime blockers

- [ ] P10 `cpu_generation_runtime` — implement the LFM2 CPU/reference generation path across embedding, conv, attention, and MoE stages _(package: `model/lfm2`)_ _(fixture: `model/lfm2/testdata/lfm25_8b_a1b_reference_placeholder.json`)_ _(validate: `cmd/lfm2inspect -require-ready`)_
- [ ] P90 `nvidia_runtime` — add NVIDIA acceleration after CPU/reference parity is established _(package: `backends/nvidia`)_ _(after: CPU/reference parity)_ _(validate: `cmd/qwen3ttsinspect -require-runtime / cmd/lfm2inspect -require-runtime`)_

## qwen3_tts runtime blockers

- [ ] P10 `cpu_talker_runtime` — implement the Qwen3-TTS CPU/reference Talker semantic-token path _(package: `model/qwen3tts`)_ _(fixture: `model/qwen3tts/testdata/customvoice_reference_placeholder.json`)_ _(validate: `cmd/qwen3ttsinspect -require-numeric-parity`)_
- [ ] P20 `cpu_code_predictor_runtime` — implement the Qwen3-TTS CPU/reference CodePredictor acoustic-code path _(package: `model/qwen3tts`)_ _(fixture: `model/qwen3tts/testdata/customvoice_reference_placeholder.json`)_ _(after: cpu_talker_runtime)_ _(validate: `cmd/qwen3ttsinspect -require-numeric-parity`)_
- [ ] P30 `decoder12hz_runtime` — implement the Qwen3-TTS 12Hz decoder and WAV/PCM output path _(package: `model/qwen3tts`)_ _(fixture: `model/qwen3tts/testdata/customvoice_reference_placeholder.json`)_ _(after: cpu_code_predictor_runtime)_ _(validate: `cmd/qwen3ttsinspect -require-ready`)_
- [ ] P90 `nvidia_runtime` — add NVIDIA acceleration after CPU/reference parity is established _(package: `backends/nvidia`)_ _(after: CPU/reference parity)_ _(validate: `cmd/qwen3ttsinspect -require-runtime / cmd/lfm2inspect -require-runtime`)_
- [ ] P100 `streaming_runtime` — add streaming execution after CPU/reference parity is established _(package: `model/qwen3tts`)_ _(fixture: `model/qwen3tts/testdata/customvoice_reference_placeholder.json`)_ _(after: CPU/reference parity, nvidia_runtime where applicable)_ _(validate: `cmd/qwen3ttsinspect -require-ready`)_

