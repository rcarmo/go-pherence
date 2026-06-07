# half

IEEE-754 half-precision (FP16) and bfloat16 (BF16) → float32 conversion.

A stdlib-only leaf package (zero import-cycle risk) consolidating conversions that
were previously duplicated across `loader/gguf`, `model`, and `model/ideogram4`.

```go
half.F16ToF32(bits uint16) float32   // IEEE-754 binary16 → float32
half.BF16ToF32(bits uint16) float32  // bfloat16 (high 16 bits of a float32) → float32
```

The three former FP16 implementations were proven bit-identical for every
finite/inf/zero value across all 65536 inputs (only NaN bit-payloads differed,
which is harmless), so this consolidation is behavior-preserving. See
`half_test.go`.
