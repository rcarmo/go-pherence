# ime2test — IME2 backend smoke test

Minimal smoke test that exercises `backends/spacemit/ime2` directly (load,
repack, single matmul) without the full decode loop. Fast sanity check that the
IME2 dispatch and `vmadot` path are live on the current host.

## go-pherence packages used
- `backends/spacemit/ime2`

## Kernels / SIMD to migrate
- None inline. Consumes the `ime2` package; should become an example/test in
  `backends/spacemit/ime2` once kernels are consolidated there.
