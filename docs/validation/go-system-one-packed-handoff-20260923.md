# Go System One packed execution hand-off

This port brings the accepted standalone packed Gemma path to `go-pherence`. Source: [`go-system-one@ee905d0`](https://github.com/rcarmo/go-system-one/tree/ee905d0271a13569eda5449cfd03932e955337ca); target base: `3a3a629bea7bc4bc2a11ac3bb7f8e6c23251d6f8`. The already-upstream attention barrier fix is unchanged.

## Scope

* Segmented independent-context/tree attention, segmented RoPE, selected Q5 batch readout and bounded 512-row reusable prefill scratch.
* Multi-field/multi-token tree packing, per-entry oversized fallback and automatic NVIDIA packing. `-packed-token-rows=0` retains serial execution; automatic `-1` selects 512 on NVIDIA and 0 on SIMD.
* Q6 shared activation staging and Q5 512-column chunks, with development CUDA sources and PTX regeneration scripts. Production still needs only the NVIDIA driver.
* Tail, isolation, reorder, parity and CLI tests; the `scripts/decisioncompare` tool and its frozen cohort.

The TypeSafe API and playground are separate review slices. No trained head, model artifact, CUDA-toolkit runtime dependency or unrelated upstream source is imported.

## Measurements and numerical limits

[Standalone benchmark data and charts](https://github.com/rcarmo/go-system-one/blob/efe3d929a9112fbfb6ad0ebfbf753d222e4c21dc/docs/benchmarks/README.md) identify the measured binaries: 72.80 ms single-boolean median and 8.330 s for 100 two-field entries. Those are standalone measurements, not timings collected from this port.

[Packing precision](https://github.com/rcarmo/go-system-one/blob/efe3d929a9112fbfb6ad0ebfbf753d222e4c21dc/docs/performance/multifield-precision.md) is not universally bitwise identical: Q8 packed activations replace F32 in tiny serial branches. The 80-field study retained all winners but had four losing-rank swaps, one diagnostic 95% crossing and maximum candidate-probability movement of 0.16542037. Existing pinned-reference tolerances are unchanged.

The later Q/K activation-reuse probe was inconclusive and is excluded. Compatible Q4 gate/up fusion already exists in the base runtime.

## Validation in the port

The fresh temporary clone passed affected-package tests and vet, a whole-tree CPU-only race run, model-layout checks, AMD64 build and ARM64/RISC-V cross-builds. NVIDIA tests passed Q5/Q6 tails, segmented/branched attention and the softmax-race regression three times.

Released-model boolean and multi-field llama.cpp fixtures passed. Packed-versus-serial logits/probabilities matched exactly at 128, 256 and 512 rows on that pinned fixture. Run with:

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./...
make model-layout-check
go test ./backends/nvidia/runtime \
  -run 'Test(Segmented|Branched|Q5|Q6|CausalBatchAttentionShortSoftmaxRace)' -count=3
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B=/path/to/gemma-4-12b-it-UD-Q4_K_XL.gguf \
GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER=/path/to/tokenizer \
go test ./model/gosystemone \
  -run 'TestGoSystemOne(PackedReleased|NVIDIAMultiFieldReleased|NVIDIAReleased)' -count=1 -v
```

An initial unrestricted runtime run failed the unrelated `TestWhisperAttentionFullOnlineLiveParity` tolerance once. An isolated five-repeat rerun and subsequent full runtime run passed; the clean base also passed. The cause is unestablished, so this is recorded as an intermittent observation, not dismissed as a proven pre-existing defect. No Whisper code or tolerance was changed.

The active neighbouring checkout was never modified. Review and merge these changes through the PR; the standalone repository's source-import direction remains one way.
