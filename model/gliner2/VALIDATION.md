## GLiNER 2.5 validation

Host: Intel Core i7-12700, linux/amd64. Measurements and tests collected on 2026-09-18 against `fastino/gliner2.5-base-v1` and pinned upstream revision `d7c727458bf6929bc9ef5ee04e13c3f717a7c455`.

## Published checkpoint checks

The retained fixtures use float32 upstream execution and the checkpoint's actual weights. They cover one request per task, not arbitrary schema compatibility.

| Path | Checked result after Plan 9 SGEMM dispatch |
| --- | --- |
| Entities | Exact token IDs, candidate spans and masks; max logit delta 6.68e-6 |
| Classification | Exact token IDs; choice logit deltas below 1e-5 |
| Relations | Exact token IDs and pair set; max relation logit delta 9.54e-7 |
| Natural-anchor records | Exact token IDs, candidate spans and masks; max assignment delta 1.72e-5 |
| Optional record decoding | Selected fields/spans and capped confidence match the retained upstream fixture |
| Mixed entity/classification | Exact combined tokens/candidates; extraction max delta 3.82e-6 |
| Mixed entity/relation | Exact combined tokens and pair set; scores within 2e-3 tolerance |
| Mixed record/entity | Exact combined tokens/masks; assignment max delta 1.77e-5 |
| Described/example schemas | Exact upstream token IDs and structural query/text positions |

Tiny reference tests also cover DeBERTa attention and the complete encoder, three record anchor modes, and Unigram normalisation/segmentation. Required/exclusive decoding uses deterministic hand cases and brute-force assignment checks. No claim is made that unspecified tie ordering matches another implementation.

Package/CLI tests, vet and race tests pass. amd64, arm64 and riscv64 CLI builds pass with cgo disabled. Cross-compilation is not runtime validation on ARM64 or RISC-V.

## Whole entity pipeline

The benchmark includes schema tokenisation, DeBERTa, boundary and candidate scoring, and final entity decoding. Model loading is excluded and one scoring request warms the model first. Input: `Ada Lovelace lived in London.`, labels `person`, `location`. CPU affinity: cores 0-5; `GOMAXPROCS=6`.

A CPU profile attributed 98.21% of samples to `simd.GemmRows`: despite its package name, the multi-row implementation was scalar. Replacing that call with checked `SgemmNTTo` preserves output overwrite semantics by clearing the destination before the accumulating assembly kernel.

| Version | Time per request |
| --- | --- |
| Scalar batch dispatch, one timed iteration | 36.030 s |
| Scalar batch dispatch with CPU features disabled, one iteration | 35.772 s |
| Plan 9 SGEMM, three samples of three iterations | 556.598, 566.959, 545.770 ms |

The post-change median is 556.598 ms, approximately 64.7 times faster than the single baseline sample. This is a narrow workload, not a throughput guarantee across documents or schemas. Allocations remain about 53.2 MB and 6,001 allocations per request. All four retained published-model parity tests passed after the dispatch change.

Raw results are in `testdata/benchmarks/`. Reproduce with:

```sh
GLINER_MODEL_DIR=/path/to/model GOMAXPROCS=6 taskset -c 0-5 \
  go test ./model/gliner2 -run '^$' \
  -bench BenchmarkPublishedEntityPipeline -benchtime=3x -count=3
```

## Final supported-path checks

The final run enables both `GLINER_MODEL_DIR` and `GLINER_TOKENIZER_JSON`, so published-checkpoint and described/mixed-schema tests execute rather than skip. All retained published fixtures pass, along with ordinary tests, race tests, focused vet, CPU-features-disabled tests and cgo-disabled tests. Both native CLIs cross-compile for linux/amd64, linux/arm64 and linux/riscv64.

Repository-wide `go test ./...`, `go build ./...` and `go vet ./...` were also attempted. They do not pass on this host: SpaceMIT packages require missing IME2 symbols or a RISC-V C compiler, live NVIDIA Whisper attention runs out of device memory, and unrelated model fixtures fail (truncated GGUF, an MTP QKV reduction delta and Qwen3.5 tests). Vet also reports existing unsafe-pointer warnings and the DiffusionGemma Q6 assembly return-width diagnostic. The failing source/test paths are unchanged by these ports. The native Jevlike/GLiNER packages and affected frozen-encoder tests pass; this is not a claim that the entire repository is green.

## Compatibility boundary

Entities, classification, relations and record groups work individually or in shared mixed contexts. Ordered descriptions/examples are supported, and record fields include optional/required scalar/list cardinalities plus exclusive assignment. Automatic chunking, selection-field DSLs, user callback validators and arbitrary checkpoint architectures remain outside the native surface. The fixture corpus is deliberately small: broader real-document evaluation is appropriate before deployment on a new task. Do not treat these results as parity for the complete upstream Python API.
