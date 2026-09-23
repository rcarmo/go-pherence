# Pocket TTS native training validation — 2026-09-22

The native training gate covers aligned-manifest admission, exact EOS and LSD-diagonal loss reductions, F32-owned `SimpleMLPAdaLN` reverse-mode gradients, analytic gradients for a frozen-backbone affine topology, one AdamW step, EMA and deterministic checkpoint resume.

## Source and fixture

- Upstream: `kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac`.
- Reference files: `training/dataloader/loader.py`, `training/modules/model.py`, `training/modules/samplers.py`, `training/checkpointing.py` and `training/args.py`.
- Frozen-topology fixture: `model/pockettts/testdata/training_tiny_pytorch.json`, SHA-256 `67cc11d62d35c0acb844f6bab548e1d8f1661fef846b743bbe43460c3c8907de`.
- Flow-head/LSD fixture: `model/pockettts/testdata/flow_head_backward_pytorch.json`, SHA-256 `61e39435037f081a8b7657328ed28d8327fe12b7a3c8cf64f7c1943efe172e00`.
- Transformer fixture: `model/pockettts/testdata/transformer_backward_pytorch.json`, SHA-256 `05f0100620f498a2a01b55c2c4367bad8b222049179b1ff84f740ba549bd8b42`.
- FlowLM layout fixture: `model/pockettts/testdata/flowlm_training_pytorch.json`, SHA-256 `fb823cd9140be2fa6d0355ee694dd5e4bc410ef56f8900e958f77e1f0f12462b`.
- Complete-step fixture: `model/pockettts/testdata/training_step_pytorch.json`, SHA-256 `2774178aba94bfcfea47932d557c53c5aaf2a43ba32a12fbe1d7a463070d22ed`.
- Oracle: PyTorch `2.13.0+cu130` in the pinned upstream checkout's virtual environment, using `torch.float32` and no Go code.

The fixture has four frozen hidden rows, two latent channels and the mask `[true,true,false,false]`. It records the normalized LSD diagonal loss, EOS loss, weighted total, all trainable gradients, one AdamW update and the resulting EMA shadow.

## Implemented boundary

`model/pockettts/training_manifest.go` admits JSONL rows only when they contain:

- a non-empty audio path and transcript;
- finite positive duration and non-negative start;
- at least two finite, monotonic word alignments within the utterance;
- at least one midpoint between adjacent words that leaves one second on both sides.

The parser bounds lines to 8 MiB and requires callers to set a positive row limit. It ignores extra dataset metadata and does not open audio or latent files.

`EOSLossAndGradient` follows `training/modules/model.py`:

- EOS target one on the first invalid frame;
- target zero on valid frames;
- active positions are valid frames plus that first invalid frame;
- position zero is never an EOS target;
- mean reduction over active positions.

`LSDDiagonalLossAndGradient` follows the diagonal branch in `training/modules/samplers.py`. With normalization enabled, each row is

```text
sum((prediction - velocity)^2) * exp(log_variance) / latent_channels - log_variance
```

and the function returns the mean over selected valid rows. `FlowMatchingLossAndGradient` also implements the upstream optimal-transport objective for caller-supplied noise and time samples.

`TinyTrainingModel` concatenates frozen hidden state, interpolated latent and time, then applies a trainable affine flow projection. Its EOS head is affine over the same frozen hidden state. `ForwardBackwardTinyTraining` returns analytic gradients for all parameters. `DefaultTinyTrainingConfig` supplies the upstream weights `p_equal=0.75` and `eos_loss_weight=0.1`; explicit zero weights remain valid.

`TinyTrainer` applies decoupled AdamW with `DefaultAdamWConfig`, updates EMA after the optimiser step, and stores parameters, moments, EMA, optimiser settings and step in a versioned JSON checkpoint. Explicit zero weight decay remains valid. Saves use a temporary file, file `fsync`, atomic rename and parent-directory `fsync`. Reloaded state produces the same second update as uninterrupted state.

`FlowHeadCPU.ForwardBackward` owns a single-row reverse-mode tape for F32 weights. It differentiates affine projections, SiLU, affine and non-affine LayerNorm, the upstream unbiased-variance timestep norm, AdaLN modulation, residual gates, both time embeddings, the latent input and the FlowLM condition. `ForwardTimeJVP` propagates an exact forward-mode tangent for either time condition through the same operations. `ForwardTimeJVPBackward` applies reverse-over-forward AD so objectives depending on both `v` and `dv/dt` receive exact mixed parameter/input derivatives. All reject BF16 inference storage because training must own mutable F32 parameters.

`LSDDistillForwardBackward` composes the normalized upstream `s→t` row: `x_s`, primary velocity and time JVP, `x_t`, `dx/dt`, endpoint velocity, squared residual and learned log-variance weight. Its endpoint discards direct parameter, condition and time gradients while retaining the input VJP, matching `f_grad_x_only` and `stopgrad_type="minimal"`. It returns gradients for every primary flow-head parameter, condition, noise, target, `s`, `t` and log variance.

The flow fixture instantiates the pinned upstream `SimpleMLPAdaLN` module with a reviewable four-channel topology and stores both PyTorch time-JVP/mixed-gradient cases and a complete `f_grad_x_only` LSD row. Separate central differences cover every native parameter and input. The generator verifies the exact clean upstream revision, `mlp.py`, `samplers.py`, `utils.py` and imported module paths before writing.

`TransformerCPU.ForwardBackward` covers the stateless F32 causal block: affine pre-norms, packed QKV, adjacent-pair RoPE, scaled dot-product attention and softmax, finite context windows, residual layer scales, tanh-GELU FFN and final LayerNorm. The context-two fixture records output, full sequence gradient and every parameter from the pinned upstream `StreamingTransformer`; all match within `1e-5`. It also exposed and fixed a stateless inference bug where `TransformerCPU.Forward` ignored finite `context`, unlike upstream and the streaming path. Checked shape arithmetic rejects overflow before allocation.

`FlowLMTrainingCPU.ForwardBackward` implements the one-row `build_sequences_with_conditions` boundary with dropout disabled: `[bos_before_voice, voice, text, input_linear(bos,audio[:-1])]`, followed by stateless transformer/out-norm execution, gathered audio conditions and EOS projection. It returns gradients for the text LUT, both BOS values, speaker/input/EOS projections, transformer, voice latents and normalized audio inputs. The last target latent receives no direct shifted-input gradient. The pinned upstream fixture matches the assembled sequence, `z`, EOS, both caller-input gradients and every parameter within `2e-5`. Every derived allocation uses checked arithmetic and token admission does not narrow `uint32` IDs.

`PocketTrainingStep` calls the same native components with one shared explicit noise tensor and sampled diagonal/`s→t` times. It means valid-row flow losses, applies `p_equal=0.75`, `1-p_equal`, and EOS weight `0.1`, accumulates one shared flow-head gradient and one `dZ` before a single backbone VJP, and backpropagates both learned log-variance leaves through the shared upstream ReLU `w_s_t` MLP. A deterministic `LSD` subclass fixes only sampled times; the fixture calls pinned `TrainableTTS.forward` directly with dropout probabilities zero. Raw flow metrics, normalized flow loss, EOS, total loss, log-variance leaves, caller inputs and every parameter match within `8e-5`.

`FullTrainer` applies decoupled AdamW then EMA and matches PyTorch after one complete step. Its versioned checkpoint stores all trainables, optimizer settings/moments, EMA, mutable latent mean/std and fixed timestep frequencies. Buffer tensors are restored but excluded from AdamW/EMA. Atomic publication uses file `fsync`, rename and parent-directory `fsync`; a resumed second update is identical to uninterrupted execution, and malformed state is rejected without mutation.

## Numerical results

Against the independent fixture:

- normalized LSD diagonal loss: `0.1699705868959427`;
- EOS loss: `0.7662174105644226`;
- weighted total: `0.20409968495368958`;
- accepted absolute tolerance for losses: `2e-7`;
- accepted absolute tolerance for frozen-topology gradients, AdamW values and EMA values: `3e-7`;
- accepted absolute tolerance for flow-head forward/backward and time-JVP parity: `5e-6`;
- accepted absolute tolerance for reverse-over-JVP mixed gradients: `1e-5`;
- accepted absolute tolerance for the normalized LSD minimal-stop-gradient row and all gradients: `3e-5`;
- accepted absolute tolerance for stateless transformer output, input and all-parameter gradients: `1e-5`;
- accepted absolute tolerance for FlowLM layout outputs, caller-input gradients and all parameters: `2e-5`;
- accepted absolute tolerance for direct complete-step metrics, input/leaf/all-parameter gradients: `8e-5`;
- accepted absolute tolerance for full-model AdamW/EMA one-step values: `4e-6`.

Checks run:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/pockettts -count=10
go test ./model/pockettts \
  -run '^(TestTinyTrainingPyTorchParity|TestFlowHeadBackward|TestFlowHeadTimeJVP|TestLSDDistill|TestTransformerBackward|TestFlowLMTraining|TestPocketTrainingStep)' -count=100
go test -race ./model/pockettts \
  -run 'TestTraining|TestEOS|TestFlowLoss|TestTinyTraining|TestFlowHead|TestLSDDistill' -count=10
go vet ./model/pockettts
```

All passed on the amd64 development host.

## Open boundary

This gate freezes the FlowLM hidden rows. The flow head now has reverse-mode gradients, but the transformer does not.

The normalized LSD diagonal and `s→t` self-distillation terms now have exact native loss and flow-head gradients. The `s→t` path includes its time JVP, mixed derivatives and minimal stop-gradient endpoint rule.

## Allocation-first optimisation

The correctness graph was frozen at `961c645958cdaeb690cfbbef74186b80698d6dd5`. Measurements used Go `1.26.3`, Linux/amd64, an Intel i7-12700 constrained to six single-threaded cores, `GOMAXPROCS=1`, `GO_PHERENCE_DISABLE_NVIDIA=1`, the three-frame/two-valid-frame complete-step fixture, and ten benchmark samples.

Baseline warm results:

| Phase | Time range | Bytes/op | Allocs/op |
|---|---:|---:|---:|
| forward/backward | 61.7–77.4 µs | 66,296 | 1,184 |
| AdamW + EMA | 32.7–37.3 µs | 26,865 | 123 |
| full step | 98.9–115.0 µs | 93,167 | 1,307 |

The baseline CPU profile spent 30.9% cumulative time in `runtime.mallocgc`; the allocation profile recorded 129,363 objects over 100 steps. `newFlowHeadGradients` and repeated FlowLM/dual forwards were the largest bounded orchestration costs.

Changes, in measured order:

- retained the first FlowLM forward tape for the single seeded backbone VJP;
- retained the primary dual flow tape for reverse-over-JVP;
- accumulated diagonal and distill rows into shared step-owned flow gradients, with a separate reusable discard tree for endpoint input-only VJPs;
- made flow-head admission allocation-free;
- cached immutable trainer parameter topology and used indexed live-gradient bindings, eliminating warm optimizer map/string construction;
- added request-owned `TrainingWorkspace` buffers and `PocketTrainingStepInto`; results alias the workspace until its next non-concurrent call;
- added topology and rebinding guards, checked workspace arithmetic and warm allocation regression tests.

Definitive warm results before the final audit guards were 48.2–51.8 µs, 29,321 B/op and 707 allocs/op for the full step; the guards preserve the same 29,320 B/op and 707 allocs/op in the final quick check. Warm AdamW+EMA is `0 B/op, 0 allocs/op`. Relative to baseline, the full step uses 68.5% fewer allocated bytes and 45.9% fewer allocations, with about half the latency on this tiny workload.

Setup is reported separately:

- workspace: about 2.3 µs, 5,408 B and 86 allocations;
- trainer: about 15.9 µs, 24,735 B and 128 allocations;
- fixture/model construction: about 0.75–0.84 ms, 201,457 B and 1,518 allocations.

The 10,000-step maximum RSS moved from 96,640 KiB to 93,696 KiB. The final CPU profile still spent 26.6% cumulative time in `runtime.mallocgc`; the remaining 70,251 allocation objects over 100 steps are distributed across higher-order dual/tape vectors (`linearForwardTraining`, `linearForwardDual`, `linearBackwardDual`, `zeroDualAdjoint`, block tapes and clones). No reusable arithmetic kernel dominates the tiny-shape profile, so SIMD promotion is not justified at this gate. A production-size workload may justify arena-backed tapes and SIMD after frozen Mimi latent preparation supplies representative shapes.

Evidence files are under `/workspace/tmp/pockettts-training-opt/`: `before.txt`, `after-definitive.txt`, CPU/memory profiles, allocation/in-use top reports and RSS logs. They are bounded external evidence and are not committed.

## Frozen Mimi latent-cache interchange

The pinned upstream contract is implemented without claiming a native raw-audio encoder:

- `<source>_latents.jsonl` preserves each audio row and adds `latents_file`;
- the exact relative shard path is `latents/<mimi_hash[:8]>/<source>_<index:08d>.safetensors`;
- `<source>_latents.meta.json` contains `stitch_frames`, `noise_floor`, `frame_rate`, `weights_path` and the full SHA-256 `mimi_hash`;
- each shard contains exactly one finite F32 tensor named `latents`, row-major `[frames,channels]`;
- admission requires explicit row/frame/total-element ceilings, exact 12.5 Hz metadata and expected Mimi hash, real-path root confinement, checked arithmetic and a content digest retained in private immutable cache state;
- each load rechecks dtype, shape and digest before returning owned F32 values;
- native shard publication fsyncs the temporary file, atomically renames it and fsyncs the parent directory.

Latent-mode manifest admission follows upstream rather than the stricter raw-audio split gate: at least one aligned word record is required, null boundary timestamps are allowed, and rows with no eligible one-second cut fall back to frame zero and the full transcript.

Stitching preserves upstream's cold-start correction. The loader freshly encodes a fixed, zero-padded `stitch_frames` audio prefix at the selected cut, discards `min(stitch_frames,target_frames)` cached frames, appends the cached tail and uses `target_frames` as the valid-mask length. Therefore the physical tensor can be longer than its valid target when a target is shorter than the fixed overlap.

The deterministic native writer was independently loaded with the pinned checkout's `safetensors.safe_open`: one `torch.float32` `latents` tensor, shape `[2,2]`, and exact values `[[1.25,-2.5],[3.75,4.0]]`. Repeated and race tests cover round-trip, short-target stitching, no-cut/null-boundary fallback, malformed metadata/path/dtype/shape/limits, post-admission replacement, non-finite data and failure non-publication.

## Production-shape admission

`PlanTrainingShape` is allocation-free and requires explicit target, prompt, text, total-sequence and resident-byte ceilings. The current exact native graph remains one row; effective batches are represented honestly through gradient accumulation, and a flow-batch multiplier other than one is rejected until native batching exists.

The upstream defaults map to 375 target frames (30 s × 12.5 Hz, exact) and 62 prompt frames (`int(5 s × 12.5 Hz)`). With an explicit 512-token native text ceiling, the worst admitted released row is:

- sequence rows: `1 + 62 + 512 + 375 = 950`;
- exact trainables: `89,449,730` (`341.22 MiB` F32), including the tokenizer padding row and default normalized-LSD `2→32→32→32→1` weighting MLP;
- parameters + equal-size gradients + two Adam moments + EMA: five parameter copies;
- conservative current activation/tape ceiling: `1,870.05 MiB`;
- conservative resident ceiling: `3,576.17 MiB` (`3.49 GiB`).

The activation ceiling counts retained transformer tapes, dense `[heads,rows,rows]` attention probabilities, backward scratch, FlowLM/workspace buffers and simultaneous primal/dual flow-row tapes, then applies a 2× safety factor for short-lived clones and allocator rounding. It is an admission upper bound, not a measured RSS claim.

A 24-layer teacher plan contains exactly `316,015,874` trainables and is admitted under an explicit 64 GiB ceiling. Every compound add/multiply is checked before use. `NewAdmittedTrainingWorkspace` binds a private plan snapshot and reruns execution-grade F32 topology/storage validation before allocation: vocabulary/embedding and BOS shapes, transformer heads/layers/FFN/layer scales, exactly two time embeddings, flow blocks/final projections, default weighting topology, forbidden biases/BF16 side storage and exact aggregate parameter elements. Mutating exported report fields cannot change allocation authority.

## Released-format export

`ExportPocketSafetensors` implements pinned `training/checkpointing.py::export_pocket_safetensors` semantics:

- begin with the raw native FlowLM state, including mutable `emb_mean`/`emb_std` and fixed timestep-frequency buffers;
- overlay only FlowLM parameter names present in the EMA shadow; untracked/frozen parameters retain raw values;
- prefix every FlowLM key with `flow_lm.` and exclude the normalized-LSD training-only `flow.w_s_t` network;
- emit two time conditions for LSD and one for FlowMatching;
- append exactly the registered frozen Mimi module state under `mimi.*`, not arbitrary source-prefix tensors;
- write every exported tensor as canonical F32, matching upstream's F32 FlowLM state and F32 Mimi module after loading source weights.

The independent `mimi_state_shapes_pytorch.json` fixture was generated from `build_mimi(config.mimi).state_dict()` at the pinned revision. It freezes 87 released Mimi names/shapes and has SHA-256 `84d0460044d02598042e20b622f6964d5f47caaacd7d683e1f6324fdcba62994`. Native topology-derived inventory matches it exactly. Missing or extra Mimi entries, wrong shapes/dtypes, non-finite values, unsupported FlowLM topology and absent requested EMA are rejected before publication.

The safetensors header and payload are lexical and deterministic. The existing destination directory is synced before temporary-file creation; the complete file is flushed, file-synced and atomically renamed; the directory is synced again. Ordinary errors are pre-publication and destination-preserving. If the rename succeeds but the final sync fails, the typed `PocketExportPublishedError` identifies the complete published path while reporting durability uncertainty. An independent upstream `safetensors.safe_open` check loaded the full synthetic export: 55 tiny FlowLM + 67 tiny-config Mimi tensors, partial-EMA values, live buffers, F32-widened Mimi data and no `w_s_t`.

The deterministic one-row F32 correctness, allocation optimisation, latent-cache interchange, production-shape admission and released-format export gates are complete. Native raw-audio Mimi encoding still requires the approved encoder bundle/fixture. Representative production-row profiling, 24-layer teacher to six-layer depth/CFG distillation and long CPU training remain open.
