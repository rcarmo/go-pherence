# Community-1 packed Vulkan LSTM weights — 13 September 2026

Column-major packed LSTM weights reduced the pinned 30-second Community-1 hybrid median from 27.531 seconds to 21.624 seconds on Intel Iris Xe. The complete result stayed byte-identical to the previously qualified output.

## Change

`NewVulkanLSTM` now transposes each row-major PyTorch IFGO input and recurrent matrix once during resident-owner construction. The packed buffer keeps its checked logical `[rows, columns]` shape but stores `packed[column*rows+row] = source[row*columns+column]`.

The sequence shader reads adjacent output rows from adjacent addresses. Each lane still visits columns in ascending order and keeps the existing separate input projection, recurrent projection and bias composition. `VkLSTMSequenceF32.Stage` retains row-major behaviour for existing callers; the Community owner selects `StagePacked` explicitly. No service, model, tolerance or default changed.

## Baseline decomposition

A temporary diagnostic measured the existing full 21-window trained hybrid:

| Stage | Total | Per window | Share |
|---|---:|---:|---:|
| CPU SincNet | 7.122573s | 339.170ms | 25.84% |
| Vulkan LSTM | 12.285845s | 585.040ms | 44.58% |
| CPU head | 0.128889s | 6.138ms | 0.47% |
| Vulkan embedding plus CPU tail | 7.387860s | 351.803ms | 26.81% |
| Other orchestration/postprocess | 0.635661s | — | 2.31% |
| Total | 27.560827s | — | 100% |

The diagnostic source was removed after the run. `baseline-stage-timing.log` has SHA-256 `df47e41d5975fd2fe4d81ff0590be2528876eb821c4a24485474d4367d8cbc53`.

## Native results

Environment:

- model revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`;
- device `Intel(R) Iris(R) Xe Graphics (RPL-P)`;
- `/usr/share/vulkan/icd.d/intel_icd.x86_64.json` selected through both Vulkan ICD variables;
- `GO_PHERENCE_DISABLE_NVIDIA=1`, `GOMAXPROCS=2`, `CGO_ENABLED=0`;
- pinned 480,000-sample public WAV and the existing segmentation, embedding and PLDA assets;
- physical-device runs bounded to 180 seconds.

The synthetic LSTM comparison passed with maximum absolute errors of `5.22e-8` for output, `2.98e-8` for hidden state and `7.45e-8` for cell state.

A saved trained SincNet feature trace isolates the four-layer bidirectional LSTM. Twelve calls took 4.283016 seconds, or 356.918ms per call. The earlier full-run stage diagnostic measured 585.040ms per call, so the recurrent stage improved by `1.639×`.

Three complete trained runs measured:

| Run | Construct | 21-window run | Close |
|---|---:|---:|---:|
| 1 | 33.338ms | 21.695295s | 0.717ms |
| 2 | 39.128ms | 21.623503s | 0.980ms |
| 3 | 35.957ms | 21.463590s | 0.643ms |

The new run median is 21.623503 seconds. The three retained pre-change measurements were 27.520213, 27.531066 and 27.560827 seconds, with a 27.531066-second median. Median whole-run speedup is `1.273×`, a 21.46% latency reduction.

Each trained run completed all 21 windows with 37 clustering rows, two clusters, 13 full turns, 12 exclusive turns and 84 ambiguous frames. Tracked Vulkan memory was unchanged at 87,219,248 bytes in two allocations and returned to the initial state after close. The newly serialised result and the retained baseline are byte-identical:

```text
111e98ad0a6d1fb5dc61bcf6db4ff0d3d44485cc66c09efcabf7736b7d8f4ee3
```

This identity includes segmentation, masks, support data, embeddings, clustering diagnostics and reconstructed turns. Existing one-sample DER values and the `qualified=false` decision therefore remain unchanged. The default tie policy still rejects ambiguity; this diagnostic uses explicit `LowestIndexTies`.

## Verification

- packed-layout checks compare every stored matrix element with its row-major source;
- row-major and packed push-constant paths pass mock ABI/admission tests;
- all focused Vulkan and Community model-free tests pass;
- all 23 checked-in shaders pass stored/embedded identity, `spirv-val`, rebuild validation and normalised rebuild comparison;
- affected Community, speech-job and server packages pass;
- `go vet` and `git diff --check` pass;
- synthetic native LSTM parity and three complete trained runs pass.

`make` was unavailable in the runtime, so the exact commands from `speech-vulkan-community-check` and `speech-vulkan-static-check` were executed directly. No service ran or changed. No corpus, model asset, deployment setting, tolerance or default changed.
