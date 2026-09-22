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

The deterministic one-row F32 correctness graph is complete with dropout disabled and explicit sampled inputs. Allocation, retained-memory and CPU profiling has not yet run on this frozen graph. Frozen Mimi latent precomputation, 24-layer teacher to six-layer depth/CFG distillation, released safetensors export and long CPU performance qualification also remain open.
