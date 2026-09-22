# Pocket TTS native training validation — 2026-09-22

The first native training gate covers aligned-manifest admission, exact EOS and LSD-diagonal loss reductions, analytic gradients for a frozen-backbone affine topology, one AdamW step, EMA and deterministic checkpoint resume.

## Source and fixture

- Upstream: `kyutai-labs/pocket-tts@0acce6b2f390150267557770d2098c5caa9a18ac`.
- Reference files: `training/dataloader/loader.py`, `training/modules/model.py`, `training/modules/samplers.py`, `training/checkpointing.py` and `training/args.py`.
- Fixture: `model/pockettts/testdata/training_tiny_pytorch.json`.
- Fixture SHA-256: `67cc11d62d35c0acb844f6bab548e1d8f1661fef846b743bbe43460c3c8907de`.
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

## Numerical results

Against the independent fixture:

- normalized LSD diagonal loss: `0.1699705868959427`;
- EOS loss: `0.7662174105644226`;
- weighted total: `0.20409968495368958`;
- accepted absolute tolerance for losses: `2e-7`;
- accepted absolute tolerance for gradients, AdamW values and EMA values: `3e-7`.

Checks run:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/pockettts -count=10
go test ./model/pockettts -run '^TestTinyTrainingPyTorchParity$' -count=100
go test -race ./model/pockettts \
  -run 'TestTraining|TestEOS|TestFlowLoss|TestTinyTraining' -count=10
go vet ./model/pockettts
```

All passed on the amd64 development host.

## Open boundary

This gate freezes the FlowLM hidden rows. It does not differentiate through the transformer or the released `SimpleMLPAdaLN` flow head.

The released LSD objective also has an `s→t` self-distillation term. Its gradient includes a JVP with respect to `t` and the upstream `minimal` stop-gradient rule: the endpoint target propagates into `x_t` while its flow-head parameter gradient is stopped. Native flow-head backward and this JVP boundary are the next loss-parity slice.

Frozen Mimi latent precomputation, full FlowLM backward, 24-layer teacher to six-layer depth/CFG distillation, released safetensors export and long CPU performance qualification have not run.
