## Jevlike validation record

Host: Intel Core i7-12700, linux/amd64. Tests and measurements collected on 2026-09-18.

Passed: package and CLI tests, package/CLI vet, race tests and cgo-disabled tests. ARM64 and RISC-V test binaries cross-compile; this is not runtime validation on either architecture.

## Whole-workload timings

The order was native, CPU-features-disabled, CPU-features-disabled, native. Each command used `taskset -c 0`, `GOMAXPROCS=1`, `-benchtime=1s` and `-count=1`. The disabled runs used `GODEBUG=cpu.all=off`; this affects more than one kernel and is not a single-kernel attribution experiment.

| Workload | Native samples | Features-disabled samples |
| --- | --- | --- |
| Eight-example scoring, including byte batching and probabilities | 0.938, 1.109 ms | 3.305, 3.656 ms |
| One training epoch, 32 examples, including initialisation and validation | 187.763, 238.408 ms | 218.802, 231.893 ms |

Scoring consistently improves with native CPU dispatch in this small sample. Training ranges overlap; no reliable training speedup is claimed. The high allocation counts (4,287 per scoring batch and 112,978 per training epoch) leave room for buffer reuse before adding more assembly.

Raw outputs are in `testdata/benchmarks/native.txt` and `testdata/benchmarks/portable.txt`.

```sh
GOMAXPROCS=1 taskset -c 0 go test ./model/jevlike \
  -run '^$' -bench BenchmarkTiny -benchtime=1s -count=1
GODEBUG=cpu.all=off GOMAXPROCS=1 taskset -c 0 go test ./model/jevlike \
  -run '^$' -bench BenchmarkTiny -benchtime=1s -count=1
```

## Unfinished validation

Frozen decoder tests use small in-memory models and injected CLI encoders. There is no published full-size transformer parity result. Visual encoder tests use scalar convolution references, not a complete imported PyTorch policy. Those gaps must not be described as real-model parity or whole-port completion.
