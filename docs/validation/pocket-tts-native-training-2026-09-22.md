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

## Correctness-gate boundary

The first frozen-hidden-row fixture isolates the Flow/EOS arithmetic. Later fixtures in this record cover the stateless transformer backward, FlowLM conditioning gradients and the complete one-row training step. The normalized LSD diagonal and `s→t` self-distillation paths include the time JVP, mixed derivatives and minimal stop-gradient endpoint rule.

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

## Frozen Mimi raw-audio encoding and latent-cache interchange

The native encoder covers the released SEANet stack, projected finite-context transformer and replicate-padded 16× downsampling. The independent pinned fixture uses 30,721 samples, 272 transformer rows and 17×32 output latents; native output matches upstream within `3e-5`. The same latents pass unchanged through the cache loader.

`EncodeInto` processes at most 16 latent frames (30,720 samples and 256 transformer rows) per chunk while retaining convolution state and transformer K/V across chunks. Workspace setup for 375 frames allocates 91,912,120 bytes in 66 allocations. Reusing that workspace encodes 30 seconds in 4.29–4.77 seconds on an Intel i7-12700 with `GOMAXPROCS=1`, `GO_PHERENCE_DISABLE_NVIDIA=1`, zero bytes and zero allocations per warm call. The convenience path allocates 91,961,272 bytes in 67 allocations. CPU profiling attributes the work to the existing SGEMM/FMA kernels; no new SIMD kernel was added.

Input, written output and workspace storage must be disjoint. The implementation detects aliases before reset or output mutation; tests cover input/output, input/workspace and output/workspace overlap. Checked capacity calculations use the largest intermediate tensor rather than the first convolution shape.

The pinned upstream cache contract is also implemented:

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
- conservative current activation/tape ceiling: `3,740.10 MiB`;
- second persistent flow-gradient scratch: `37.22 MiB`;
- conservative resident ceiling: `5,483.45 MiB` (`5.35 GiB`).

The activation ceiling counts retained transformer tapes, dense `[heads,rows,rows]` attention probabilities, backward scratch, FlowLM/workspace buffers and simultaneous primal/dual flow-row tapes, then keeps four copies of that calculated live set for transient clones, allocator rounding and one garbage-collector cycle of dead tapes. The earlier 2× factor admitted 3.49 GiB, but a ten-update released-topology run reached 4,664,672 KiB process RSS, so that factor was removed. The ceiling is an admission upper bound, not a measured RSS claim.

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

## Depth and CFG distillation

`DepthDistillForwardBackward` follows the pinned `TrainableTTS.forward` distillation branch for one native row:

1. run the student on the fully conditioned sequence;
2. run the frozen teacher on the same full condition;
3. run the teacher force-null with no voice or text rows (`[bos_before_voice,audio]`);
4. form `target = z_null + cfg_coef * (z_conditioned - z_null)` under stop-gradient;
5. compute hidden-dimension mean-squared error per frame, then mean over `shifted_mask = [mask[0],mask[:-1]]`;
6. backpropagate only through the student conditioning/backbone.

The independent `depth_distill_pytorch.json` fixture calls the pinned conditioner and transformer modules directly with a one-layer student, two-layer teacher, CFG coefficient `2.0` and original mask `[true,false,false]` (shifted `[true,true,false]`). SHA-256 is `c39e8cb65d7a0b2ce0acde845029acfbc5919a4da57ff374ba4a3e945228d856`. Native loss, student output, conditioned/null teacher outputs, guidance target, audio/voice input gradients and every active student parameter gradient match within `3e-5`; EOS gradients are exactly zero.

Teacher seeding implements upstream's champion `ends` selection. `24→6` retains layers `[0,1,2,21,22,23]`, remaps them to `[0…5]`, copies every non-layer tensor unchanged and owns all copied values. Multi-digit indices and malformed/overflowing state names are covered.

`DistillTrainer` registers only text embedding, BOS values, speaker/input projections, transformer and final norm. EOS, flow head and `w_s_t` are absent from AdamW moments, weight decay and EMA, so teacher-calibrated heads remain byte-for-byte unchanged. Step validation is transactional; the active gradient subset is copied before mutation to handle hostile parameter aliases. Trainer/student pointers and layer topology are immutable, while same-model same-shape slice rebinding follows live values rather than stale arrays.

## Released production-row profile

`LoadFlowLMTrainingCPU` and `LoadFlowHeadTrainingCPU` materialise the pinned released checkpoint into mutable owned F32 storage, including voice conditioning and latent statistics. The training-only normalized-LSD `w_s_t` network is not present in the exported inference checkpoint; `NewDefaultLSDWeightMLP` constructs upstream's exact `2→32→32→32→1` topology with every affine parameter zero-initialised.

The representative row uses 375 target frames, 62 prompt frames, 32 deterministic text tokens and finite deterministic latent/noise/time arrays. This produces 470 transformer rows (`1 + 62 + 32 + 375`). Measurements used the Intel i7-12700 host, CPU-only mode and one benchmark iteration:

| boundary | time | bytes/op | allocs/op |
|---|---:|---:|---:|
| admitted workspace setup after gradient-tree reuse | 8.68–11.21 ms | 400,074,048 | 363 |
| forward/backward before production SIMD | 104.19 s | 1,880,279,232 | 378,533 |
| forward/backward after checked primal/dual SIMD, batched transformer linears and reusable gradient trees, one CPU | 22.94 s | 1,561,300,104 | 353,101 |
| same forward/backward with `GOMAXPROCS=6` | 19.65 s | 1,561,391,352 | 353,846 |
| warm AdamW + EMA | 594.6 ms | 0 | 0 |

The original CPU profile spent 92.9% of samples in scalar primal/dual linear forward and backward loops. Existing checked SIMD dispatch was sufficient; no new assembly kernel was required. Batching transformer forward, input-gradient and weight-gradient matrices preserves scalar-reference results in a dedicated 17-row differential test and keeps the independent PyTorch/finite-difference gates unchanged.

The remaining allocation profile is dominated by transformer and per-frame primal/dual flow tapes. Reusing the FlowLM gradient tree reduced allocated bytes by about 319 MB and retained the tiny warm gate at 670 allocations against the existing ceiling of 720. The next stage must address per-frame backward temporaries; a whole-utterance scratch would retain memory proportional to the number of frames.

A ten-update released-topology run completed in 3m50s with finite parameters, Adam moments and EMA. Loss was `0.1108239` at step ten. External peak RSS was 4,664,672 KiB. A separate 100-update run passed under `GOMAXPROCS=6`, CPU-only mode and an explicit 45-minute Go test timeout. It took 33m29.856s; loss was `0.1108239` at step ten and `0.006236044` at step 100 (the step-90 value was lower, `0.005383019`). Post-GC live heap stayed between 1,831,802,032 and 1,831,847,704 bytes over the ten-step boundaries; the external peak RSS was 4,328,756 KiB. The test checked finite parameters, Adam moments and EMA after the final update. This qualifies finite repeated updates and bounded live heap on the released six-layer topology for one deterministic synthetic 470-row workload; it does not test 512-token worst-case execution, released-target training quality, checkpoint/resume at released shape or the 24-layer teacher. The first 100-step attempt ended at step 30 under Go's default ten-minute timeout; the subsequent standalone binary with `-test.timeout=45m` exited successfully. The binary preceded the final loader and workspace-topology admission checks; the subsequent package tests cover those checks.

Evidence: `/dev/shm/pockettts-production-profile/soak-100-final.txt` (external log), plus the baseline and batched CPU/allocation profiles in the same directory. Run the opt-in gate from `model/pockettts` with `GO_PHERENCE_POCKETTTS_VOICE_MODEL` pointing at the pinned SHA-256 `fb0dc01b0d4d2e1c905b7a3e0676e3d9c96d5ae460e24e3ab94981805babf997` artifact and `GO_PHERENCE_POCKETTTS_RELEASED_TRAINING_STEPS=100`; use `go test -timeout=45m -run '^TestReleasedProductionTrainingSoak$' -count=1 -v`. Ordinary CI skips the released gate.

### Bounded flow-tape scratch — subsequent local slice

`TrainingWorkspace` now reuses separate request-owned tape buffers for each frame's primal/dual primary flow and endpoint flow. The primary scratch resets after the diagonal VJP and again before the next frame. Endpoint storage is separate because its VJP runs while the primary dual tape remains live. Standalone flow APIs still return owned results; the scratch is internal to `PocketTrainingStepInto`.

An opt-in differential ran the same pinned released model and deterministic 470-row inputs through the owned-tape reference and scratch paths. Loss components, every named training parameter gradient, both latent/voice input gradients and both learned log-variance gradient arrays matched exactly; neither scratch spilled beyond its bounded slots. The small PyTorch/finite-difference gates remain unchanged. A ten-update scratch soak completed in 2m57.5s with final loss `0.1108239`, finite parameters/moments/EMA, no spills and 4,314,776 KiB external peak RSS. Its final post-GC heap after release was 382,688 bytes. The earlier 100-update gate used the pre-scratch binary; the scratch path has ten-update evidence, not a new 100-update soak.

At one CPU, the first scratch use allocated 1,254,386,968 bytes in 221,003 objects. A subsequent warm step allocated 1,253,720,600 bytes in 220,726 objects (three samples). Before scratch, the equivalent step allocated 1,561,300,104 bytes in 353,101 objects: the warm reduction is 307,579,504 bytes (19.7%) and 132,375 objects (37.5%). Three timing samples overlap: pre-scratch 25.24–28.53 seconds, scratch 24.12–26.79 seconds; no latency improvement is claimed. Workspace setup now allocates 400,093,152 bytes in 367 allocations, 19,104 bytes and four allocations above the pre-scratch setup. Post-change allocation profiles place the remaining flat object cost in transformer backward, row-wise linear backward, `zeroDualAdjoint` and flow-head backward temporaries; their ownership and capacity need a separate bounded pass.

Evidence: `/dev/shm/pockettts-production-profile/tape-soak10.txt`, `tape-warm-bench.txt`, `tape-mem.pprof`, and `tape-objects-top.txt`. These local files are not committed.

### Primary dual-backward scratch — subsequent local slice

A third request-owned pool holds the primary dual VJP's row-local adjoints and linear input gradients. It is distinct from both live forward tapes and resets before each frame's distillation term. Standalone `ForwardTimeJVPBackward` and `LSDDistillForwardBackward` still allocate owned results. The released owned-vs-scratch differential remains exact for every metric and gradient; the one-step released gate and tiny warm allocation gate pass with no scratch spills.

On the same i7-12700 with `GOMAXPROCS=1`, three warm 470-row samples allocated 1,093,877,592–1,093,877,608 bytes and 151,351 objects per step, down about 160 MB and 69,375 objects from the first tape slice. Elapsed samples were 23.14–28.02 seconds, overlapping the earlier 24.12–26.79 seconds; no latency gain is claimed. Admitted workspace setup allocated 400,102,672 bytes in 369 allocations, an extra 9,520 bytes and two allocations over the first tape slice. The remaining unaddressed costs include transformer backward and ordinary flow-head/endpoint backward temporaries. This slice has released one-step parity and no-spill evidence, not a new ten- or 100-update soak.

Evidence: `/dev/shm/pockettts-production-profile/dual-backward-bench.txt`. The earlier ten-update tape soak predates this third pool.

### Ordinary and endpoint backward scratch — subsequent local slice

A fourth request-owned row-local pool reuses ordinary diagonal and endpoint VJP temporaries. Diagonal backward completes before distillation; the pool resets before endpoint backward. Its `dEndpointInput` stays live in this fourth pool while the primary dual VJP uses the third pool, so the minimal-stop-gradient endpoint and subsequent input-gradient accumulation do not alias.

The pinned released owned-vs-scratch 470-row differential matches every loss component, parameter gradient, caller-input gradient and log-variance gradient exactly. All four pools report zero spills. Three warm one-CPU samples allocated 946,324,104 bytes and 88,351 objects per step, down 147,553,488 bytes (13.5%) and 63,000 objects (41.6%) from the third-pool checkpoint. Setup rose by 9,552 bytes and two allocations to 400,112,224 bytes / 371 allocations. Warm times were 20.64–21.23 seconds versus 23.14–28.02 seconds for the earlier slice; these are three one-iteration samples on the same host, not a controlled distributional speed qualification.

A separate ten-update six-CPU soak passed in 3m4.755s with final loss `0.1108239`, finite parameters/moments/EMA, zero spills, 4,277,760 KiB external peak RSS and 368,504 bytes of post-GC heap after release. A subsequent **100-update four-pool** soak ran a binary built at `55c2f7fc79eef17db8f63734361fdbe4a4c3b5f3` with `GOMAXPROCS=6`, CPU-only mode, the pinned released voice model and an explicit 50-minute test timeout. It exited successfully after 33m47.172s: final loss `0.006236044` (step 90: `0.005383019`), finite parameters/Adam moments/EMA and zero spills checked at every update. Post-GC `HeapAlloc` at ten-step boundaries ranged from 1,833,117,512 to 1,833,155,520 bytes; after release it was 419,864 bytes. External peak RSS was 4,304,368 KiB. This qualifies repeated synthetic 470-row updates and stable retained memory for the four-pool path, not output quality or the 512-token/24-layer boundaries. The remaining allocation owners are transformer backward and non-arena row outputs; their lifetime and retained-memory costs need separate evidence.

Evidence: `/dev/shm/pockettts-production-profile/ordinary-backward-bench.txt`, `ordinary-backward-soak10.txt` and `fourpool-soak100.txt`. These local files are not committed.

### Transformer attention-gradient row reuse — subsequent local slice

The current four-pool allocation profile identified one `dProb` allocation per transformer query/head in backward attention. A function-local, layer-bounded probability-gradient row now resets between query/head iterations; the summation order, gradient ownership and public API are unchanged. The released 470-row owned-vs-scratch all-gradient differential and pinned tiny PyTorch transformer parity passed. An independent lifetime/read review found no material issue.

Three unprofiled warm one-CPU samples allocated 853,930,632 bytes in 43,237 objects per step, down 92,393,472 bytes (9.8%) and 45,114 objects (51.1%) from the four-pool baseline of 946,324,104 bytes / 88,351 objects. Times ranged from 22.08 to 32.34 seconds; these measurements do not qualify a latency change. A separate ten-update six-CPU released synthetic soak passed with final loss `0.1108239`, finite parameter/optimiser/EMA state, zero flow-pool spills, 4,204,276 KiB external peak RSS, and 368,216 bytes of post-GC heap after release. The four-pool 100-update result above predates this transformer change; this slice has ten-update evidence only. Remaining allocations include transformer norm backward and owned row outputs.

Evidence: `/dev/shm/pockettts-production-profile/fourpool-current-objects-top.txt`, `transformer-prob-row-objects-top.txt`, `transformer-prob-row-bench.txt` and `transformer-prob-row-soak10.txt`. These local files are not committed.

### Transformer norm backward row reuse — subsequent local slice

Transformer norm backward now reuses one normalized-base and one affine-gradient row while returning an owned full-sequence input gradient. The helper refactor preserves the original reduction order and nil-scratch behaviour elsewhere in flow training. A central-difference gate with zero affine norm weights covers the case where the pre-affine base cannot be recovered from the forward output. The released 470-row owned-vs-scratch all-gradient differential and ten repeated tiny gates pass; an independent read found no material lifetime or parity issue.

Three warm one-CPU samples allocated 778,957,448 bytes in 24,933 objects per 470-row step, 74,973,184 bytes and 18,304 objects below the attention-row checkpoint. Elapsed samples of 23.18–31.73 seconds are too variable to establish a latency change. A separate ten-update six-CPU synthetic soak passed in 3m15.335s with final loss `0.1108239`, finite parameters/Adam moments/EMA, zero flow-pool spills, 4,039,296 KiB external peak RSS and 389,880 bytes of post-GC heap after release. The 100-update result predates both transformer slices; this slice has ten-update evidence only. Further allocation work requires a new profile of owned row outputs and retained memory.

Evidence: `/dev/shm/pockettts-production-profile/transformer-norm-row-bench.txt` and `transformer-norm-row-soak10.txt`. These local files are not committed.

### LSD row-result scratch — subsequent local slice

The workspace path now uses its live primary flow scratch for one distillation row's temporary vectors, result fields and input-gradient fields. The standalone public method still passes nil scratch and returns owned results. Current workspace consumers finish reading the row before the next reset; the primary dual tape and endpoint input VJP remain in separate live pools. An independent lifetime and slot-count review found no material alias, spill or ownership issue.

The released owned-vs-scratch 470-row differential again matches every metric and gradient exactly, with zero spills in all four pools. Three warm one-CPU samples allocated 776,605,448 bytes in 18,183 objects per step, down 2,352,000 bytes and 6,750 objects from the norm-row slice. Elapsed samples were 22.21–22.65 seconds; they are too few to establish a speed change. A separate ten-update six-CPU synthetic soak passed in 3m20.275s with final loss `0.1108239`, finite parameter/Adam/EMA state, zero spills, 4,043,596 KiB peak RSS and 387,696 bytes of post-GC heap after release. The 100-update gate predates this slice. Scratch capacity and released-shape repeated training have ten-update evidence only here; trained-audio quality is still unmeasured.

Evidence: `/dev/shm/pockettts-production-profile/distill-row-scratch-bench.txt` and `distill-row-scratch-soak10.txt`. These local files are not committed.

### Transformer norm forward row reuse — subsequent local slice

`transformerNormForward` now reuses one pre-affine normalized row within each norm call. The tape still owns its input, affine output and inverse norms; standalone forward results remain owned. This removes one temporary allocation per sequence row and norm call without changing the reduction order. The zero-affine-weight central-difference gate, pinned tiny transformer parity and released all-gradient differential pass; an independent read found no alias/lifetime issue.

Three warm one-CPU samples allocated 751,632,136 bytes in 12,086 objects per 470-row step, 24,973,312 bytes and 6,097 objects below the preceding distillation-row slice. Elapsed samples of 23.50–35.96 seconds overlap earlier noisy measurements, so no latency change is accepted. A separate ten-update six-CPU synthetic soak passed in 3m54.185s with final loss `0.1108239`, finite parameters/moments/EMA, zero flow-pool spills, 4,246,592 KiB peak RSS and 386,704 bytes of post-GC heap after release. An independent 100-update soak of the exact `fbb6571906b2b7a73277615466733d08dfb854df` test binary (SHA-256 `018463e4f39765cf8305c255689178a610c11f956fee9ab6c9c75170994bf8fe`) then passed after a Piclaw process restart without restarting the test. With `GOMAXPROCS=6`, CPU-only mode and a 55-minute test timeout, it completed in 32m9.151s. Final loss was `0.006236044` (step 90: `0.005383019`); the test checked finite parameters, Adam moments and EMA after update 100, and all four scratch pools for zero spills at every update. Post-GC `HeapAlloc` at ten-step boundaries ranged from 1,833,119,312 to 1,833,156,544 bytes; after release it was 419,976 bytes. External peak RSS was 4,246,144 KiB; the process and service both exited successfully. This qualifies finite repeated synthetic 470-row updates and bounded live heap at `fbb65719`; later changes require their own long-run evidence. It does not establish speech quality on recorded training data or 24-layer teacher performance.

Evidence: `/dev/shm/pockettts-production-profile/transformer-norm-forward-row-bench.txt`, `transformer-norm-forward-row-soak10.txt` and `fbb65719-soak100.txt`. These local files are not committed.

### Transformer tape-output copy removal — subsequent local slice

Layer training forward now returns its owned tape output to the next layer, and norm training forward returns its owned affine output instead of making a second copy. Callers either discard the return or read it before any later mutation. `TransformerCPU.ForwardBackward` still returns an owned output; a two-layer repeat-call test checks that it does not alias caller input or change after a second call. An independent caller/lifetime review found no material issue, and the released 470-row owned-vs-scratch differential remains exact.

Three warm one-CPU samples allocated 715,054,856 bytes in 12,067 objects per step, 36,577,280 bytes and 19 objects below `fbb65719`. Elapsed samples of 21.63–29.55 seconds overlap prior samples; no latency gain is accepted. A separate ten-update six-CPU synthetic soak passed in 3m14.598s with final loss `0.1108239`, finite parameters/Adam moments/EMA, zero flow scratch spills, 4,299,456 KiB external peak RSS and 376,504 bytes of post-GC heap after release. A later independent **100-update** run used the exact `ecd5dc1b` test binary (SHA-256 `84931f86569aa85c1f0e2bfac5186a7649a5f61dbde42302e9956bf53ee674bc`) with 32 deterministic text tokens, six Go CPUs, CPU-only mode and a 55-minute timeout. It passed in 33m30.002s; final loss was `0.006236044` (step 90: `0.005383019`). All four scratch pools had zero spills on every update, and parameters, Adam moments and EMA were finite after update 100. Post-GC `HeapAlloc` ranged from 1,833,113,640 to 1,833,148,032 bytes across ten-step samples; after release it was 411,480 bytes. External peak RSS was 4,299,008 KiB. The test and service exited successfully. These are synthetic repeated-update and retained-memory results; recorded-speech quality and 24-layer teacher performance are still unqualified.

Evidence: `/dev/shm/pockettts-production-profile/transformer-tape-outputs-bench.txt`, `transformer-tape-outputs-soak10.txt` and `ecd5dc1b-soak100.txt`. These local files are not committed.

### Maximum admitted text length — repeated synthetic updates

The released opt-in soak accepts `GO_PHERENCE_POCKETTTS_RELEASED_TRAINING_TEXT_TOKENS` from 1 to 512 (default 32), leaving ordinary CI unchanged. With 512 deterministic text tokens, the admitted row contains 950 sequence positions (`1 + 62 + 512 + 375`). An initial one-update CPU check passed with loss `0.5955415`, finite parameters/Adam moments/EMA, zero flow scratch spills and 2,686,080 KiB external peak RSS; the concurrent 32-token long soak makes its 53.451-second step unsuitable for a latency comparison.

A separate ten-update run at `75e23f3a3b45662f5fe960a8c783aad79ec46723` used a pinned test binary (SHA-256 `b3687362b2949f9f1cd3300dec3a14577b03bdab9b2a7904eb0c91d4e9fb5e44`), `GOMAXPROCS=6` and CPU-only mode. It passed in 6m32.848s with final loss `0.1204249`, finite parameters/moments/EMA and zero spills on each update. Post-GC heap after update ten was 1,833,113,072 bytes; after release it was 374,600 bytes. External peak RSS was 5,333,608 KiB, within the conservative 5,483.45 MiB admission bound.

A pinned **100-update** run of source `f7544fdc` (test-binary SHA-256 `10467d21f5f153badffaa68257445472e228cb6f730aa99c88045c25e03d9655`) completed in 1h5m12.833s with `GOMAXPROCS=6`, CPU-only mode and an 85-minute test timeout. Final loss was `0.007436165` (step ten: `0.1204249`), with finite parameters, Adam moments and EMA after update 100 and zero flow-pool spills at every update. Ten-step post-GC `HeapAlloc` ranged from 1,833,142,344 to 1,833,150,208 bytes; after release it was 411,720 bytes. External peak RSS was 5,333,980 KiB, below the 5,483.45 MiB admission bound for this tested topology. Test and systemd service exited successfully. This qualifies one synthetic worst-admitted-text workload for finite repeated updates and bounded retained heap; it does not prove a universal RSS ceiling or recorded-speech training quality.

Evidence: `/dev/shm/pockettts-production-profile/released-max-text-step.txt`, `maxtext-ten.txt` and `maxtext100-f7544fdc.txt` (not committed).

### Released-shape resume boundary

The opt-in `TestReleasedTrainingInMemoryResume` loaded the pinned six-layer model, ran one 470-row update, copied the complete parameter/buffer/Adam/EMA state into a fresh released model and compared every state value before and after another update against the uninterrupted trainer. It passed in 90.11 seconds on the local CPU with no serialized checkpoint. The fixture uses deterministic synthetic latents, not recorded speech; this establishes exact released-shape in-memory restore/update equivalence, not training quality or file-based checkpoint/resume.

The existing JSON `SaveFullTrainingState` can emit a released state, but `ReadFullTrainingState` limits the entire JSON value to `512<<20` bytes. The released model alone has 89,449,730 F32 trainables; at step one parameters, two Adam moments and EMA contain at least 1,431,195,680 raw F32 bytes before JSON formatting, names and buffers. Thus a released checkpoint cannot fit the reader's limit. Tiny-graph round trips do not cover this. A production checkpoint needs its own bounded format/reader and measured end-to-end save/load/resume qualification; do not treat the in-memory differential as that gate.

A separate opt-in file-backed gate uses `SaveFullTrainingStateBinary` / `LoadFullTrainingStateBinary`, leaving the old JSON format intact for small fixtures. The binary stores named F32 tensors, metadata and a whole-file SHA-256; its reader requires an explicit caller byte ceiling, bounds each tensor and rejects corrupt/truncated/trailing data. The save path requires an existing directory, flushes and syncs a temporary file before rename and directory sync on the tested Linux filesystem. The released 1,431,217,112-byte file was published in `/dev/shm`, read under a 2 GiB limit and matched the in-memory state exactly; the next resumed update again matched uninterrupted state. It passed in 97.68 seconds with 13,152,592 KiB maximum process RSS during a concurrent four-pool soak. This peak is not covered by the ordinary training admission bound; released checkpointing must reserve additional memory and storage. The current path does not promise cross-platform atomic replacement; it requires an existing directory so newly created parent entries cannot be mistaken for crash-durable publication. A later opt-in test replaced the **same** 1,431,217,112-byte checkpoint after step two, read it into another fresh released model, compared the complete restored parameter/buffer/Adam/EMA state immediately and matched a third update against uninterrupted training. It passed in 136.24 seconds with peak RSS 10,406,672 KiB on the idle host. The earlier 13,152,592 KiB peak was measured under concurrent load, so these two peaks are not directly comparable. The released-size test covers successful repeated publication and restoration; the small-fixture suite covers malformed/over-limit checkpoint rejection and mutates loaded parameter, buffer, moment and EMA slices after restore to verify trainer ownership across garbage collection. Production checkpoint cadence has not been measured, and neither test establishes output quality on recorded audio.

Evidence: `/dev/shm/pockettts-production-profile/released-inmemory-resume.txt`, `released-binary-resume.txt` and `released-binary-resume-twice-final.txt` (not committed).

## Deterministic tiny-graph CPU soak

The opt-in `TestPocketTrainingDeterministicTinySoak` qualifies checkpoint/resume and retained-memory behavior on the complete exact **tiny** graph. It does not claim released-shape performance or memory qualification.

Two independent 100,000-step-per-run executions passed under `GOMAXPROCS=1` and CPU-only mode. Each test executed 200,000 complete updates: one uninterrupted run and one run that repeatedly replaced and reloaded the same checkpoint path every 10,000 steps. Explicit noise and all sampled times varied deterministically by step. At every boundary, resumed state matched uninterrupted state before save and after reload.

Both executions produced identical hashes:

- final logical state SHA-256: `a30108f97e6be944ec8cc881ebdfe70d7afd3576feb441e985585260da601344`;
- final checkpoint-file SHA-256: `9682dfb99b55015d03a6a51135c92945f5aa50735cb74943c8c50551b295ec7c`.

Wall times were `12.39 s` and `11.23 s`; external `/usr/bin/time -v` peak RSS was `98,048 KiB` and `97,408 KiB`. Separate continuous/resumed post-GC heap series include step zero. Every sampled boundary remains within 8 MiB for `HeapAlloc`/`HeapInuse` and 10,000 objects of its run baseline. This supports the narrow claim that no large retained-heap growth appeared during these tiny-graph runs; it does not measure transient peaks inside an update. The established warm allocation gates are rechecked in the same executable (`≤720` forward/backward allocations and zero optimizer allocations).

Evidence is under `/workspace/tmp/pockettts-training-soak/` as `soak-v2-100000-run{1,2}.json` and `.log`; it is bounded external evidence and is not committed.

The native raw-audio Mimi encoder, released-topology profiling, four-pool 100-update finite CPU soak and one released-shape binary file-backed resumed update are complete for their measured synthetic workloads. Full native training quality on recorded speech, transformer/row-output allocation, repeated production checkpoint scheduling and 24-layer teacher performance remain open.
