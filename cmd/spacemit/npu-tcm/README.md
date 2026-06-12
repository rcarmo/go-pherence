# npu-tcm — TCM substrate validation

Validates the pure-Go Tightly-Coupled-Memory substrate against the SpaceMIT
A100 cores: opens `/dev/tcm`, prints block geometry, acquires AI cores, and
round-trips data through TCM.

## go-pherence packages used
- `npu` (TCM device wrapper)

## Kernels / SIMD to migrate
- None inline. This is the canonical exerciser for the `npu`/TCM layer; keep it
  as the integration test for whatever absorbs the `ime2run` AI-core handshake
  (`backends/spacemit/tcm/`).

## Status
Hardware-dependent (needs `/dev/tcm` + A100 cores 8–15). Keep as a board-side
integration probe.
