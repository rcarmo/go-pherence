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

## Remaining checks

Mixed schemas, descriptions/examples, automatic chunking and arbitrary checkpoint variants remain unsupported. Broader real-document parity and stable load-controlled throughput samples are still needed. Do not treat a passing short fixture as completion of the entire upstream API port.
