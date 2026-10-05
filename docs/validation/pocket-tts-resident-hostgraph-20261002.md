## Pocket TTS composition metadata allocation

Distillation backward and forward composition now allocate final-sized tensor, command and output metadata slices once. Complete source extents are known and validated before reservation, so repeated append growth is avoidable. Command/tensor order, IDs, nested read-only sources, owned remapped command inputs, arithmetic, public results and replay scopes are unchanged.

The source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference is unchanged at eight paths and 469 insertions. NVIDIA/Vulkan were disabled. No native GPU, released-model/audio, recovery or publication run occurred.

## Profiled cost and reservation

The complete warmed private LSD mock row allocated about 2.85 MB/op before this change. A bounded `-memprofilerate=1` allocation-attribution run identified two composition costs: backward composition and forward remapping each accounted for about 13% of allocated bytes. Most of those bytes came from cloning the first source slice, growing it for the next source and repeatedly growing command storage.

After each source passes its existing tape/binding admission, the composer sums source tensor/command lengths and rejects a combined count above 4096. Independently admitted source lengths are at most 4096, so the two-/three-source sums cannot overflow an int. The forward reservation includes its two advance/desired vector commands and tensors. Missing sources/boundary positions still fail before indexing. Output capacity includes all ordered parameter roots.

The destination metadata has independent backing arrays; source constants/shapes remain read-only exactly as before. Remapped command input lists still have owned copies. Composition does not mutate source headers, bindings or edge IDs on success or late budget failure. The conservative full-source byte ceiling still includes dead placeholders and duplicate constants. No graph pruning, value folding or mutable cache was added.

## Alternating before/after samples

Preserved before/after test binaries ran alternately on the same host, five samples per path with 100 mock rows each. Preparation/warm-up was outside measurement. The complete row includes graph construction, admission, pruning, interpretation, downloads and accumulation. `GOMAXPROCS=2`, offline Go, `-p=1 -parallel=1` and both GPU disable flags were fixed.

| Path | Before median B/op | After median B/op | Before median allocations | After median allocations |
| --- | ---: | ---: | ---: | ---: |
| Public tiny row | 2,687,755 | 2,313,243 | 8,934 | 8,912 |
| Private tiny row | 2,854,013 | 2,479,471 | 9,001 | 8,978 |

The private row allocates about 374.5 KB less per call (**13.1%**); the public row shares the same composition implementation and falls by 13.9%. Allocation-count samples vary by one allocation due to surrounding metadata, with about 22 fewer per row.

Public timing medians were 1.290 ms before and 1.289 ms after; ranges 1.253--1.309 / 1.267--1.386 ms overlap. Private medians were 1.345 / 1.310 ms; ranges 1.330--1.447 / 1.296--1.402 ms also overlap. An initial non-alternating after batch was slower and is retained. The alternating comparison establishes the byte reduction without a clear timing regression; it does not establish a reliable latency gain or native speedup.

### Allocation re-profile

Bounded 20-row attribution used the same private fixture before/after. Exhaustive allocation profiling is separate from normal timing samples.

| Caller, flat allocated bytes | Before | After |
| --- | ---: | ---: |
| Backward composition | 7,896.94 KB | 3,775.95 KB |
| Forward remapper | 7,896.98 KB | 829.47 KB |
| Forward composer | 104.33 KB | 3,246.55 KB |

Forward capacity is now allocated by the composer, so remapper and composer must be considered together: **8,001.31 -> 4,076.02 KB**, about 49% less. Backward composition falls about 52%. The whole attribution run allocated 62,254.82 -> 54,199.94 KB; setup and fixture creation are included in those totals.

`PruneTape`, initial graph command growth and tensor construction are now the largest flat callers. Post-GC in-use profiles measured 49.95 / 43.02 KB, with no observed retained increase in this bounded fixture. Those values are not peak process RSS, released-size residency or native allocator peaks. CPU time profiles and released-model/native performance have not run.

## Offline verification

Evidence directory: `/workspace/exports/pocket-gpu-training-hostgraph-20261002`.
Gate script: `/workspace/tmp/pocket-hostgraph-gate.sh`.
Final window: **2026-10-02 08:27:11--08:28:08 UTC**.

| Gate | Passed |
| --- | ---: |
| Selected tests, ten repetitions | 100 top-level + 320 subtests |
| Selected tests, race detector | 10 top-level + 32 subtests |
| Affected CPU/mock regressions | 448 top-level + 778 subtests |

Final logs contain zero failures/skips. New tests admit each source independently, then reject the oversized combined metadata before reservation. Existing composition tests cover inventory/extents, seed/input IDs, missing sources, exact byte ceilings and source immutability. Independent CPU/F64 and pinned PyTorch mixed-AD/FlowLM/full-step references pass unchanged tolerances. Objective endpoints, padding, all gradients/updates, clipping, checkpoint/resume and terminal retention pass. Tiny device graph/download accounting is unchanged.

Both modified distillation source files retain **100% scoped statement coverage**. Vet, host `go build ./...`, Linux ARM64/RISC-V pockettrain cross-builds, formatting, layout and diff checks pass. Native code and replay identities did not change.

### Review and limits

A mathematical judge found no obvious alias/lifetime/arithmetic issue and asked about absent sources and integer overflow in reservation arithmetic. Complete-source limits and boundary-count checks cover those conditions. The judge did not inspect source or execute native code. Independent source review, native resources/throughput and voice/listening acceptance are open. No hold was released.

## Next work

Pruning and initial graph command/tensor growth still dominate measured host allocations. Audio-hidden transformer/flow handoffs, prediction admission, dense embedding VJP, EOS scalar/diagnostic synchronisation and full snapshots/live copies remain. Continue measured, bounded offline changes before authorised native qualification.
