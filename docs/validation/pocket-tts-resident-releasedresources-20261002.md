## Released-size resident admission blocker

The released Pocket TTS configuration requires **2,866,551,644 logical resident bytes**, exceeding the CLI maximum of **2,147,483,648 bytes** by **719,067,996 bytes**. The current resident training path rejects this configuration before device allocation. No runtime ceiling or host guard was changed.

The audit uses `model/pockettts/testdata/english-upstream.yaml`, synthetic positive-zero host arrays, the existing graph builders and pure resource validators. It loads no checkpoint/assets and executes no model, signal, project binary or native GPU operation. Host fixtures share immutable backing arrays by extent to keep audit allocation bounded; each named parameter retains its full independent logical device charge.

Source parent: `a90aec30567d485784f2e04d1803c543c9f2817f`; inference index remains eight paths / 469 insertions. Training source is uncommitted. All GPU/reset/recovery/model/audio/listening/publication holds remain.

## Resident inventory

The released configuration has flow depth 6 / width 512, transformer width 1024 / six layers / feedforward scale 4, vocabulary 4000 and latent dimension 32. The independent extent inventory includes every trainable FlowLM, flow head and default `2→32→32→32→1` weighting parameter. Nontrainable latent statistics and time frequencies are excluded.

`TestTrainingReleasedDeviceInventoryMapping` constructs synthetic models with those dimensions and compares every name/extent against the actual `fullParameterMap`. It does not construct a trainer or evaluate the model. Both inventories contain 131 arrays and 89,448,706 elements.

`ResidentBytes` charges eight F32 buffers per parameter element: parameter, gradient, gradient scratch, branch upload, zero, M, V and EMA. Per-parameter norm statistics contain three elements per 256-element chunk. Global norm/device scale/state checks add twelve F32 elements, or 48 bytes.

| Quantity | Value |
|---|---:|
| Trainable arrays | 131 |
| Parameter elements | 89,448,706 |
| Parameter F32 bytes | 357,794,824 |
| Norm-stat elements | 1,048,251 |
| Eight parameter-sized buffers + norm statistics | 2,866,551,596 bytes |
| Global buffers | 48 bytes |
| Total logical resident bytes | 2,866,551,644 bytes / 2.669684 GiB |
| CLI default | 268,435,456 bytes / 256 MiB |
| CLI maximum | 2,147,483,648 bytes / 2 GiB |
| Excess over maximum | 719,067,996 bytes |

`TestTrainingReleasedDeviceResidentRejectBeforeAllocation` calls the existing state-check constructor with a fake driver at both current ceilings. Both return `resident byte ceiling` with a nil state and zero allocations, uploads or tape constructions. No native loader is called.

`@default` independently recomputed 131 arrays, 89,448,706 parameter elements and 1,048,251 norm-stat elements from YAML and parameter collectors, without project execution or assets. Its direct scoped source review confirms the CLI and resident constructor rejection paths. The shared note is `/workspace/exports/pocket-devicehandoff-independent-review-20261002.md`.

## Released graph admission

The audit builds released-dimension forward/backward metadata using synthetic finite parameters. It performs no graph execution or numerical quality comparison. Every complete source passes the existing 4096-tensor/command ceilings and exact logical byte admission; one byte less rejects. Pruning and zero/lifetime storage planning use protected parameter IDs and declared roots.

| Graph | Tensors | Commands | Complete logical bytes | Pruned zero/lifetime slot bytes |
|---|---:|---:|---:|---:|
| Diagonal forward | 1,040 | 857 | 119,009,440 | 39,088,316 |
| Distillation forward | 2,074 | 1,712 | 238,018,112 | 39,090,764 |
| Diagonal backward | 1,041 | 858 | 119,009,328 | 78,349,032 |
| Distillation backward | 2,115 | 1,737 | 238,020,776 | 78,353,716 |

Each graph has 66 flow parameter bindings. Planned slots include borrowed parameter source extents; they are conservative accounting, not owned native allocations or device peak. Synthetic zero values can share immutable zero constants according to the existing planner. Real parameter values are unmeasured.

The retained diagonal forward/backward pair charges **238,018,768 bytes**. The distillation pair charges **476,038,888 bytes**, already exceeding the default 256 MiB ceiling. These are strict live-source lower bounds: they exclude transformer/EOS parent, row-condition child, native plan overhead and host graph/snapshot storage. The lease layer intentionally charges every live complete source plus the complete consumer before pruning.

## Storage design floor

Removing branch-upload and zero arrays alone would leave six parameter-sized buffers (P, two gradient ping-pong destinations, M, V, EMA), norm statistics and globals. That hypothetical layout still charges **2,150,961,996 bytes**, exceeding 2 GiB by **3,478,348 bytes**, before tapes. It is an arithmetic floor for that proposal; no such storage change is implemented.

Resolving released-size admission needs a reviewed storage or memory-policy design. Possible designs must preserve unused-parameter admission, explicit gradient readiness, repeated/scaled/compact accumulation, exact optimiser/EMA state, snapshots and failure retention. Reusing gradient destinations, compressing state, excluding buffers, changing logical ceilings or relying on a larger GPU cannot be assumed equivalent. Combined resident/live-tape/native/host overhead must remain explicit. No ceiling widening was authorised.

## Scoped review corrections

The independent reviewer found no additional concrete lifetime, admission-order or gradient-scaling defect in the retained-device coordinator and private diagonal/distillation helper paths with host downloads/guards retained. This is scoped source review, not full-repository or native approval.

Five stale helper comments said there was no production caller or CLI scope. They now identify the private experimental resident-devicehandoff coordinator and explicitly preserve host downloads/guards and native qualification requirements. Only comments changed in those reviewed runtime files. Targeted existing metadata-freeze/lifecycle tests passed after the comment corrections; the completed broad gates were not repeated.

## Isolated final gates

Window: **2026-10-02 16:36:03--16:37:40 UTC**. Go network and GPUs disabled; `GOMAXPROCS=2`, `-p=1 -parallel=1`.

| Gate | Top-level passes | Subtests | Failures | Skips |
|---|---:|---:|---:|---:|
| Three audit repetitions | 18 | 12 | 0 | 0 |
| Audit race run | 6 | 4 | 0 | 0 |
| Audit + affected CLI identity/factory checks | 8 | 4 | 0 | 0 |

Vet, host build, Linux ARM64/RISC-V test compilation, model-layout and diff checks passed. Foreign-architecture tests were compiled only. No kernels changed or native compilation/execution repeated. The initial successful audit window is preserved under `initial-pass/`. Fixture field-name errors were corrected before passing checks.

Evidence: `/workspace/exports/pocket-gpu-training-releasedresources-20261002`; script: `/workspace/tmp/pocket-releasedresources-gate.sh`. Source tests: `model/pockettts/training_device_resources_test.go` and `training_device_resources_linux_test.go`.

## Qualification still required

Resource/admission reconciliation precedes native qualification. Separately authorised PTX and Vulkan runs must cover cutoff neighbours/signs/NaN/Inf/signed zero/subnormal/extreme logvar and FTZ behaviour; zero-weight finite witnesses; actual aliases/lifetimes/device-complete ownership/failures; all numerical gradients/updates/clipping/EMA/resume against independent references; and transfer-inclusive repeated timing plus device/host peak/retention. Forward download removal is a separate policy after those gates. Released voice training, audio/listening, incident clearance and inference publication remain separate decisions.
