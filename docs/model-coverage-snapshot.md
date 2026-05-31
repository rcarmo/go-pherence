# Model coverage snapshot

| family | status | covered | pending | coverage | references | runtime | execution | parity | readiness |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lfm2_moe | metadata_schedule_inspector_coverage | 58 | 2 | 96.7% | 11/0 (100.0%) | 14/2 (87.5%) | 0/2 (0.0%) | 2/0 (100.0%) | 5/0 (100.0%) |
| qwen3_tts | metadata_token_prompt_inspector_coverage | 61 | 5 | 92.4% | 8/0 (100.0%) | 15/5 (75.0%) | 0/5 (0.0%) | 2/0 (100.0%) | 5/0 (100.0%) |

# Runtime roadmap

## lfm2_moe runtime blockers

- [ ] `cpu_generation_runtime` — implement the LFM2 CPU/reference generation path across embedding, conv, attention, and MoE stages _(validate: `cmd/lfm2inspect -require-ready`)_
- [ ] `nvidia_runtime` — add NVIDIA acceleration after CPU/reference parity is established _(after: CPU/reference parity)_ _(validate: `cmd/qwen3ttsinspect -require-runtime / cmd/lfm2inspect -require-runtime`)_

## qwen3_tts runtime blockers

- [ ] `cpu_talker_runtime` — implement the Qwen3-TTS CPU/reference Talker semantic-token path _(validate: `cmd/qwen3ttsinspect -require-numeric-parity`)_
- [ ] `cpu_code_predictor_runtime` — implement the Qwen3-TTS CPU/reference CodePredictor acoustic-code path _(after: cpu_talker_runtime)_ _(validate: `cmd/qwen3ttsinspect -require-numeric-parity`)_
- [ ] `decoder12hz_runtime` — implement the Qwen3-TTS 12Hz decoder and WAV/PCM output path _(after: cpu_code_predictor_runtime)_ _(validate: `cmd/qwen3ttsinspect -require-ready`)_
- [ ] `nvidia_runtime` — add NVIDIA acceleration after CPU/reference parity is established _(after: CPU/reference parity)_ _(validate: `cmd/qwen3ttsinspect -require-runtime / cmd/lfm2inspect -require-runtime`)_
- [ ] `streaming_runtime` — add streaming execution after CPU/reference parity is established _(after: CPU/reference parity, nvidia_runtime where applicable)_ _(validate: `cmd/qwen3ttsinspect -require-ready`)_

