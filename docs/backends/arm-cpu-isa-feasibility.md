# ARM CPU ISA feasibility for Gemma inference

The current Gemma CPU path has four pieces worth treating separately on ARM: Q4_0 x Q8_0 projection, Q8 activation preparation, the exact FP16 GELU lookup, and RoPE. They have different instruction and data-movement problems, so a single "NEON port" would leave a fair amount of performance on the floor.

This note describes the reusable ARM64 design. The Orange Pi 6 Plus and its CIX P1 implementation are covered separately in [CIX P1 and Orange Pi 6 Plus](cix-p1-orange-pi-6-plus.md).

## The useful ISA tiers

AArch64 guarantees floating point and Advanced SIMD in the normal application profile, but it does not guarantee the newer dot-product or matrix instructions. Runtime dispatch therefore needs several tiers:

| Tier | Architectural feature | Useful instructions | Role in this runtime |
|---|---|---|---|
| Baseline | FP + ASIMD (NEON) | vector integer operations, widening multiply/add, FP32 FMA, conversions | Portable ARM64 fallback, Q8 preparation and RoPE |
| Dot product | FEAT_DotProd | `SDOT`, `UDOT` | Broad Q4_0 x Q8_0 fast path |
| Int8 matrix multiply | FEAT_I8MM | `SMMLA`, `UMMLA`, `USMMLA` | Preferred fixed-width projection path |
| Scalable vectors | SVE/SVE2 | predication, indexed loads, scalable dot and arithmetic operations | Exact GELU gather, tails, and wider projection on CPUs with vectors wider than 128 bits |
| SVE matrix multiply | SVE-I8MM | scalable `SMMLA` family | Projection on implementations where SVE width or scheduling beats fixed-width I8MM |
| Matrix state | SME/SME2 | streaming mode, ZA tiles, integer outer products | Later large-tile projection experiment, not a baseline requirement |
| Reduced precision | FP16/FHM and BF16 | `FCVTL`, `FMLAL`, `BFDOT`, `BFMMLA` | Scale conversion and future FP16/BF16 dense kernels |

NEON is a 128-bit SIMD ISA with 32 architectural vector registers. I8MM augments those registers with small matrix operations; it is not the large stateful matrix engine provided by SME.

## Q4_0 x Q8_0 projection

The packed Q4 values can be decoded directly to signed bytes in the range `[-8, 7]`. That representation fits ARM's signed dot product better than the unsigned-byte arrangement used by AVX-VNNI:

```text
packed nibbles -> signed Q4 bytes -> SDOT/SMMLA with signed Q8 bytes
```

A signed decode removes the `8*sum(Q8)` correction required when unsigned Q4 nibbles are multiplied by signed Q8 values.

### DotProd

`SDOT` forms four groups of four signed byte products and accumulates them into four int32 lanes. It is the first useful accelerated tier because it is deployed much more widely than SVE2 or SME and does not require a matrix-specific packed layout.

The practical tile will probably be smaller than the amd64 eight-row/sixteen-token topology: ARM has the same number of vector registers, but each NEON register is half the width of a YMM register. A four- or eight-row tile over four or eight tokens is a better starting point, with the final shape selected by packing-inclusive benchmarks.

### I8MM

`SMMLA` accumulates a 2x2 int32 result over a K dimension of eight int8 values. A dedicated packed layout can pair weight rows and activation columns so that each instruction advances four output accumulators.

There are two valid organisations:

* Decode Q4 to signed bytes and use `SMMLA`. This avoids correction arithmetic and is the preferred design.
* Keep Q4 nibbles unsigned and use `USMMLA`, then subtract the Q4 zero-point correction. This may be useful if a measured packing layout makes the unsigned decode cheaper.

The kernel should keep several 2x2 output tiles live, reuse each decoded weight fragment across tokens, and convert block scales only once per fragment. Any repacking cost belongs in the projection benchmark; a fast inner instruction sequence does not justify another full activation traversal or a permanent duplicate of the model weights.

### SVE and SME

SVE becomes compelling when the implementation has vectors wider than 128 bits. On a 128-bit implementation it mostly buys predication, indexed memory operations and a different instruction-selection path, rather than more arithmetic per register.

SME's ZA state could hold a much larger output tile and consume unpacked int8 panels with integer outer products. It also adds streaming-mode transitions, a separate ABI boundary and OS-managed state. That makes it a hardware-specific follow-up after fixed-width I8MM has been measured, rather than part of the first ARM port.

## Exact Q8 activation preparation

NEON is enough for the bulk of Q8 preparation:

* vector absolute value and maximum reduction;
* one scalar scale calculation per block;
* `FCVTA` conversion, whose nearest-with-ties-away rule matches `math.Round` for finite float32 inputs;
* saturation and narrowing to int8;
* int32 byte-sum accumulation when an unsigned projection layout still needs correction.

The scale's FP16 encoding has to preserve the repository's conversion contract. Hardware FP16 narrowing normally uses IEEE round-to-nearest-even, which is not automatically interchangeable with the existing bit-level conversion. Keep the integer conversion until exhaustive boundary tests prove otherwise.

## Exact FP16 GELU lookup

The GELU path indexes a large FP16 table with input-derived values. NEON has table-lookup instructions for small register-resident tables, but no arbitrary vector gather from a table this large.

The baseline ARM path therefore has to issue scalar table loads in batches and then use NEON for FP16 widening and multiplication by the up-projection values. SVE changes that: indexed halfword loads can gather FP16 entries directly, after which the values can be widened and multiplied under a predicate.

This is the clearest reason to keep an SVE path even on a 128-bit SVE implementation. It provides the missing memory operation, although it only gathers four values into FP32-sized lanes at that width.

## RoPE and reductions

RoPE maps cleanly to NEON FP32 loads, multiplies and add/subtract pairs. SVE can simplify tails, but DotProd, I8MM and SME add nothing to this kernel. Complex multiply instructions are only attractive if the data is already interleaved as complex pairs; changing layout merely to use them would add shuffles elsewhere.

Softmax maximum/sum reductions can also use NEON or SVE. Neither ISA provides an exact vector exponential, so replacing the scalar exponential requires a separately approved numerical approximation. Argmax has similar tie-order and NaN details and should only move to assembly if a profile makes it material.

## Dispatch from Go

The module currently uses `golang.org/x/sys/cpu`. Its ARM64 feature structure exposes the useful first set:

```go
cpu.ARM64.HasASIMD
cpu.ARM64.HasASIMDDP
cpu.ARM64.HasI8MM
cpu.ARM64.HasSVE
cpu.ARM64.HasSVE2
cpu.ARM64.HasFPHP
cpu.ARM64.HasASIMDHP
cpu.ARM64.HasASIMDFHM
```

`HasASIMDDP` is the DotProd feature despite the misleading historical comment in older `x/sys` releases. The package does not currently distinguish SVE-I8MM, SVE-BF16, BF16, SME or SME2, so those need Linux `AT_HWCAP2` parsing before they are used.

SVE vector length is a runtime property. Linux exposes it through `prctl(PR_SVE_GET_VL)`; dispatch must not infer it from the CPU model. A useful selection order is:

```text
SME-specific kernel, if one is deliberately enabled and fully supported
SVE-I8MM, when measured faster for the current vector length
fixed-width I8MM
DotProd
baseline NEON
portable Go
```

The first implementation only needs the last four entries.

Go's Plan 9 ARM64 assembler supports ordinary NEON well, but support for newer SVE, I8MM and SME mnemonics varies with the Go toolchain. Where a mnemonic is missing, checked raw instruction words are viable for small leaf kernels. Generate and disassemble them with a pinned GNU or LLVM assembler, retain the readable source used to produce the words, and execute feature-gated runtime tests on real hardware. Cross-compilation proves encoding and linkage, not that dispatch is safe.

SVE and especially SME routines should be self-contained no-call leaves. SME additionally needs balanced streaming/ZA state transitions and OS support for saving that state across signals. None of this complexity applies to ordinary NEON or fixed-width I8MM.

## Validation and promotion

Each architecture tier needs the portable implementation as its oracle. Tests should cover odd rows, token tails, incomplete Q4 blocks, extreme finite FP32 values, rounding boundaries, NaNs where the public contract permits them, and state checkpoint/restore.

Runtime tests must also execute every dispatched instruction family on the target machine. Heterogeneous systems need tests under affinity to each core class because Linux advertises userspace features as a system-wide safe set, while throughput and preferred tile sizes can differ sharply between cores.

The first coherent ARM batch should contain:

* a signed-Q4 DotProd projection kernel;
* an I8MM projection kernel and the minimum packing it requires;
* a NEON Q8 quantiser;
* a NEON RoPE kernel;
* a scalar-gather/NEON GELU kernel, followed by SVE gather where the target provides it.

That sequence attacks the same measured Gemma hotspots as the amd64 work without making SME or a wider SVE implementation prerequisites.

## References

* [Linux arm64 ELF HWCAP documentation][hwcap]
* [Arm C Language Extensions feature-test macros][acle]
* [Go ARM64 assembler documentation][goasm]

[hwcap]: https://docs.kernel.org/arch/arm64/elf_hwcaps.html
[acle]: https://arm-software.github.io/acle/main/acle.html
[goasm]: https://go.dev/doc/asm
