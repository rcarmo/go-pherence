# Vulkan LSTM cell foundation

`VkLSTMCellF32` adds a model-free recurrent primitive for one unprojected PyTorch-IFGO state update. Its input is a caller-precombined `[4,hidden]` affine gate tensor plus prior `[hidden]` cell state; outputs are next hidden and cell state. Projection, bias addition, sequence order, layers and bidirectionality remain owner responsibilities. No model/default selects it.

## Contract

Hidden size is 1–256. Exact `CellOut`/`CellIn` alias is supported for recurrent in-place state. Gates must be disjoint from both outputs; hidden output must be disjoint from cell input/output. Shape, storage, overlap and grid constraints fail before command recording. Input contents must be finite and produce finite state; there is no host scan or fallback.

The shader uses branch-stable sigmoid and tanh formulas so finite large-magnitude gate values do not overflow the exponential. Arithmetic follows the standard update:

```text
i=sigmoid(IFGO[0]); f=sigmoid(IFGO[1]); g=tanh(IFGO[2]); o=sigmoid(IFGO[3])
cell=f*cell+i*g
hidden=o*tanh(cell)
```

## Offline verification

- Six hidden sizes from 1 through 256, four shuffled invocation schedules and 5,280 hidden/cell comparisons pass the source schedule model.
- Extreme finite activation inputs from `-MaxFloat32` to `+MaxFloat32` remain finite in the model.
- Tests cover four descriptor ranges, push/grid ABI, exact cell alias, forbidden partial/output aliasing, malformed shapes/storage, cancellation, plan retention through drain, constructor limits and nil close.
- All 20 embedded shader contracts pass. The shader is 4,100 bytes with stored/embedded SHA-256 `d41bfe8134dee00c628ae5d0ea5e97494e657aea349890a5f38604099f3b6e60`.

No Vulkan device, LSTM checkpoint, sequence owner, trained inference, service, deployment, push or performance run was used. Vulkan `Exp` accuracy and subnormal behavior are device-dependent and unqualified. A complete recurrent owner still needs checked projection/bias composition, forward/reverse scheduling, layer state/output ownership, cancellation and CPU/native comparison.
