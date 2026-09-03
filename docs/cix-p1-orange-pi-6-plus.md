# CIX P1 and Orange Pi 6 Plus CPU acceleration

The Orange Pi 6 Plus uses the CIX P1, sold as CD8180 and on some boards as CD8160. Orange Pi's product page describes a twelve-core 64-bit SoC and advertises 45 TOPS across CPU, GPU and NPU; those accelerator numbers should not be confused with instructions available to a Go process.

For the CPU path, this is a much more interesting board than the usual NEON-only SBC.

## CPU topology

The production CIX P1 combines:

* eight Cortex-A720 performance cores, split across several frequency domains with clocks up to 2.6GHz on production systems;
* four Cortex-A520 efficiency cores at up to 1.8GHz;
* 128-bit NEON and 128-bit SVE/SVE2 execution;
* approximately 12MiB of shared last-level cache.

Both core classes implement the instruction families needed by the proposed Gemma kernels. They do not have equal throughput: an A520 at 1.8GHz is a poor substitute for an A720 at 2.4-2.6GHz inside a barrier-synchronised projection tile.

Firmware revisions have exposed different logical CPU numbering, so code must identify cores from MIDR and frequency policy rather than assuming that CPU 0-7 are the A720 set. The architectural part numbers are `0xd81` for Cortex-A720 and `0xd80` for Cortex-A520.

## Features exposed by Linux

A Debian system using the CIX 6.1.44 vendor kernel on the same CD8180 silicon reports this common userspace feature set:

```text
fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics
fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop
sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat
ilrcpc flagm ssbs sb paca pacg dcpodp sve2 sveaes
svepmull svebitperm svesha3 svesm4 flagm2 frint
svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
```

The features that matter here are:

| Linux flag | Architectural facility | Available | Use in go-pherence |
|---|---|---:|---|
| `asimd` | NEON/Advanced SIMD | Yes | Baseline vectors, Q8 and RoPE |
| `fphp`, `asimdhp` | Scalar/vector FP16 | Yes | FP16 conversion and arithmetic |
| `asimdfhm` | FP16 multiply into FP32 | Yes | Future FP16 dense kernels |
| `asimddp` | DotProd | Yes | `SDOT` Q4_0 x Q8_0 fallback |
| `i8mm` | NEON I8MM | Yes | Preferred `SMMLA` projection |
| `bf16` | NEON BF16 | Yes | Future `BFDOT`/`BFMMLA` kernels |
| `sve`, `sve2` | Scalable vectors | Yes, 128-bit | Predication and exact GELU gather |
| `svei8mm` | SVE I8MM | Yes, 128-bit | Optional projection path |
| `svebf16` | SVE BF16 | Yes, 128-bit | Future BF16 kernels |
| `sme`, `sme2` | Streaming Matrix Extension | No | No ZA-based kernel on this SoC |

Linux arm64 HWCAP is a safe common feature set across the CPUs available to a process. Seeing `i8mm` and `svei8mm` here means that a feature-gated process can execute them on both the A720 and A520 cores, although that says nothing about equal performance.

The SVE width matters. CIX P1 implements 128-bit vectors, so SVE does not double or quadruple NEON arithmetic width. Its main value to this runtime is indexed memory access and predicated tails.

## What to build for this board

### Projection

The primary Orange Pi 6 Plus kernel should decode Q4 nibbles to signed int8 and use fixed-width `SMMLA`. That removes the unsigned Q4 correction used by the amd64 VNNI kernel and maps naturally onto I8MM's 2x2 output tiles over K=8.

A synthetic instruction-throughput test on CD8180 measured approximately 320 integer GOPS on one 2.5GHz A720 and 115 GOPS on one 1.8GHz A520. Those figures are execution ceilings rather than model throughput, but they confirm that I8MM is implemented and fast on the silicon. Once unpacking and block scales are included, memory traffic and packing will set the real limit.

Keep an `SDOT` kernel as the second tier. It provides a simpler parity target and a useful implementation for other ARM boards without I8MM.

SVE-I8MM is worth measuring after fixed-width I8MM works, but its 128-bit width means it should not be assumed faster. SME is not available.

### Exact GELU

SVE's indexed halfword load is the one architectural feature that solves a problem NEON cannot: gathering arbitrary FP16 entries from the GELU table. At 128 bits each gather supplies four FP32-sized lanes, so it is narrower than the eight-lane AVX2 path but should remove four scalar loads from each vector group.

The fallback should batch scalar table loads and use NEON for FP16 widening and multiplication. NEON `TBL` cannot address the full lookup table.

### Q8 and RoPE

Q8 preparation and RoPE should use NEON rather than SVE. Both vector ISAs are 128 bits on this SoC, and the operations do not need SVE gather. `FCVTA` is useful for Q8 rounding; the FP16 scale conversion must keep the repository's bit-exact software rule until hardware conversion has passed boundary tests.

## Scheduling the twelve cores

The existing six-worker projection topology should initially run entirely on A720 cores. Using A520 cores in the same synchronised batch can extend the critical path even if aggregate CPU utilisation looks better.

A reasonable measurement sequence is:

```text
6 fastest A720 cores
all 8 A720 cores
8 A720 + 4 A520 cores with asymmetric work allocation
```

The last case only makes sense if work stealing or static weighting gives the A520 cores smaller chunks. Equal chunks across all twelve cores are unlikely to minimise prompt latency.

Affinity needs to be established from MIDR/frequency data at startup or supplied by the operator. Linux scheduling may otherwise migrate workers between core classes, making projection timings noisy and invalidating tile-size comparisons.

## Runtime probing

The board image, kernel configuration and container policy still decide what a process can use. These commands capture the required state:

```bash
uname -a
lscpu
sed -n '1,/^$/p' /proc/cpuinfo
grep -H . /sys/devices/system/cpu/cpu*/regs/identification/midr_el1
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    printf '%s: ' "$p"
    cat "$p/related_cpus" "$p/cpuinfo_max_freq"
done
```

A tiny native probe should also call `prctl(PR_SVE_GET_VL)` and print the result. `/proc/cpuinfo` confirms that SVE is exposed, but not the active vector length inherited by the process.

Within Go, `golang.org/x/sys/cpu` is sufficient for NEON, DotProd, fixed-width I8MM, SVE and SVE2 dispatch:

```go
cpu.ARM64.HasASIMD
cpu.ARM64.HasASIMDDP
cpu.ARM64.HasI8MM
cpu.ARM64.HasSVE
cpu.ARM64.HasSVE2
```

The current `x/sys/cpu` API does not expose SVE-I8MM, SVE-BF16 or BF16 separately. Read `AT_HWCAP2` before dispatching those instructions. An Orange Pi-specific build flag is not an adequate substitute because binaries will otherwise fail with `SIGILL` on older ARM64 machines.

## GPU and NPU boundary

The CIX P1 also includes an Immortalis-G720-class GPU and a Zhouyi NPU, and Orange Pi advertises 45 TOPS of combined compute. Neither is reachable through ARM CPU instructions or Plan 9 assembly.

Using them requires a Vulkan/OpenCL GPU backend or CIX's NPU runtime, including model graph conversion, buffer ownership and driver-specific validation. They may eventually move entire projections or graphs off the CPU, but they do not change the CPU implementation order:

```text
I8MM projection -> SVE GELU gather -> NEON Q8/RoPE
```

## Known software constraints

CIX mainline kernel work is active, while current board images commonly use a vendor-derived 6.1 kernel. CPU HWCAP exposure is mature enough for these kernels, but GPU, NPU, media and IOMMU support varies substantially between vendor and mainline images.

A cross-compiled ARM64 binary proves that Go accepted the assembly and relocation model. Promotion still requires execution on the Orange Pi under each relevant core affinity, followed by a finite Gemma request and checkpoint/restore test. This is especially important for raw instruction encodings, since the Go assembler does not expose every I8MM or SVE mnemonic uniformly across toolchain versions.

## References

* [Orange Pi 6 Plus product page][orange]
* [CIX CD8180 CPU and Linux feature report][sbc]
* [CIX P1 instruction-throughput measurements][cpufp]
* [Cortex-A720 product information][a720]
* [Cortex-A520 product information][a520]
* [Linux arm64 ELF HWCAP documentation][hwcap]
* [General ARM CPU ISA feasibility](arm-cpu-isa-feasibility.md)

[orange]: http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html
[sbc]: https://github.com/ThomasKaiser/sbc-bench/blob/master/results/88LE.txt
[cpufp]: https://github.com/pigirons/cpufp/blob/master/benchmark_result/arm64/CIX_P1_CD8180.md
[a720]: https://developer.arm.com/Processors/Cortex-A720
[a520]: https://developer.arm.com/Processors/Cortex-A520
[hwcap]: https://docs.kernel.org/arch/arm64/elf_hwcaps.html
