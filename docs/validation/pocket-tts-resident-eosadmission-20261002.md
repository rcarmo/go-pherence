## Conservative device EOS admission primitive

`EOSAdmissionShape`, the PTX `training_eos_admission` kernel and Vulkan `training_eos_admission_f32` shader check a scalar absolute-dot bound over all hidden rows, including padding. The flag is zero only when the original host F64 projection guard is certainly satisfied. The primitive was first qualified independently; the [opt-in producer integration](pocket-tts-resident-residenteosadmission-20261002.md) now selects it in distinct scopes while retaining original host guards and downloads. Older replay identities are unchanged.

Source is uncommitted at parent `a90aec30567d485784f2e04d1803c543c9f2817f`. Staged inference remains eight paths / 469 insertions. Native GPU, model/audio/listening and publication holds are unchanged.

## Original guard and integer bound

`admitEOSProjection` checks each row's `abs(bias) + sum(abs(float64(hidden)*float64(weight)))` against `L = float64(MaxFloat32) * .125 / rows`. It checks padding too. Ordinary F32 accumulation could underestimate this nonnegative sum; overflow or FTZ could further change its admission.

The new primitive uses integer units `U = 2^95`, with `budget = floor(L/U) - 1`, for rows `1..1024` and width `1..16384`. It decodes each absolute F32 bit pattern as `m * 2^(e-150)`. Normal mantissas include the hidden bit; subnormals use `e=1` and their actual fraction. Exponent 255 rejects even when the other operand is zero.

For each product it rounds each mantissa upward to a multiple of 4096. The resulting integer product `p = ceil(m1/4096)*ceil(m2/4096)` is at most `2^24` and bounds the exact magnitude by `p * 2^(e1+e2-276)`. It rounds this upward again to units through guarded shifts and a low-bit remainder test. Terms too large for the remaining budget reject before shifting/adding. Each row starts with the bias times exact one and subtracts accepted units, so the running sum cannot overflow. A nonzero term below one unit still consumes one unit.

Two F32 values have an exact F64 product with at most 48 significant bits. The host sums at most 16,385 nonnegative terms; its F64 rounding error is bounded by `gamma_16385` times the exact sum. Since `L/U < 2^30`, that error is below .002 units. The one-unit reserve therefore covers possible upward rounding in the host's accumulation and division when the integer bound accepts. No device float multiplication or addition is used in admission; bit-preserving execution of the native shader still requires qualification.

## Conservative rejection

This predicate can reject host-safe values near the original limit. Positive normal mantissa overbound is below `(1+2^-11)^2 - 1`, about .0977%. Ceiling each term to a unit contributes at most `(width+1)*U`; the largest row/width geometry adds about 1.5626% of `L`, plus the reserved unit. Subnormals are handled conservatively by their decoded bits and may have larger relative mantissa overbound, but their absolute contribution is tiny compared with this EOS limit.

Boundary fixtures at rows 1024 and widths 1, 1024 and 16384 deliberately show rejection of original-host-safe near-limit inputs. In those same fixtures, a 2% margin passes. This is tested rejection behaviour, not exact admission equivalence or a relaxed threshold. Host checks remain authoritative, and scope integration must explicitly decide whether this conservative rejection is acceptable before removing any host guard.

## Independent checks

128 all-magnitude/ordinary F32 cases cover rows 1, 2, 7 and 1024; widths 1, 3, 64 and 16384. The independent original F64 absolute-dot calculation records 77 device-reference accepts and 51 rejects; all 51 rejects in that corpus also fail the host guard, and there are zero unsafe accepts.

100,000 independent exact-F64 product checks record 86,681 admitted product bounds and 13,319 rejected bounds, with no downward admitted bound. For products at least one unit, the maximum relative overbound is .999423603 -- ceiling a term just above one unit can nearly double it. That relative statistic is separate from the bounded whole-row absolute rejection cost above. Tests include unsafe final/padded rows, NaN/Inf even times zero, signs, subnormals, exhausted budget, malformed shapes, unused nonfinite source, pruning and a scalar tape interpreter fixture. Numerical acceptance uses the original inequalities and independent references; no generated-output hash is a gate.

PTX mock tests check four buffers and rows/width/budget argument ordering, extent/output aliases, missing/poisoned owner, launch/sync failure retention and blocked retry. Vulkan checks four descriptor bindings, 12 push bytes, local size 256, dispatch groups, output alias/rank/extent/context rejection, complete tape-kernel preflight and late constructor rollback. SPIR-V reflection passes the existing opcode contract without widening it.

## Constructor and cleanup cost

PTX training construction requires one additional entry, `training_eos_admission`. Vulkan training construction creates 29 pipelines instead of 28 and includes the new owner in close/rollback. All existing scopes pay that construction overhead even though none invokes the primitive yet. No construction-time or native memory/latency measurements were taken.

The cleanup pass also found that the existing compact-accumulation pipeline was missing from Vulkan normal close and rollback lists. Both lists now include it. The late 29th-pipeline construction test verifies cleanup of preceding owners. Uncertain native close still retains ownership; no reset/recovery or retry policy changed.

## Final offline gates

Window: **2026-10-02 13:22:35--13:24:06 UTC**. GPUs and Go network access disabled, `GOMAXPROCS=2`, tests `-p=1 -parallel=1`. All final commands exited successfully.

| Gate | Top-level passes | Subtest passes | Failures | Skips |
|---|---:|---:|---:|---:|
| Ten repetitions | 150 | 0 | 0 | 0 |
| Race detector | 15 | 0 | 0 | 0 |
| CPU/mock regression | 882 | 1,460 | 0 | 1 |
| Compiled-artifact follow-up | 15 | 0 | 0 | 0 |

`TestTrainingRandomMPFRComparisonDiagnostic` is the existing opt-in skip; it did not pass. Vet, host build, Linux ARM64/RISC-V builds, model-layout and diff checks passed. Offline `nvcc -ptx -arch=sm_80 --fmad=false`, `ptxas`, GLSL compilation and `spirv-val` passed. Shape, integer units/reference and PTX/Vulkan wrappers reached 100% scoped coverage; this does not qualify native arithmetic or execution.

Evidence: `/workspace/exports/pocket-gpu-training-eosadmission-20261002`; script: `/workspace/tmp/pocket-eosadmission-gate.sh`. Early wrapper fixture field mismatches and tape-validator helper names were corrected before final gates. The abstract mathematical delegate timed out after 25 seconds with no returned verdict. Independent source/mathematical approval was not obtained.

## Integration still needed

The distinct opt-in producer scope now appends the scalar flag while retaining the original hidden/EOS host guard, with flag ownership, failure ordering and pinned full-step/update/replay checks. Conservative rejection needs explicit scope documentation. Producer hidden, prediction and backward downloads/full snapshots remain; further device result lifetimes and diagnostics are unfinished. Native FTZ/bitcast, resources, throughput and new-voice listening require Rui/@default coordination.
