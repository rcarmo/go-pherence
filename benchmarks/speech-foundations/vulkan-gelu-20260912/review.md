# GELU source review record

The initial delegated read-only review of the shader, wrapper, tests and parser changes timed out after 180 seconds without returning findings. It supplies no approval.

A second, smaller judge review (github-copilot/gpt-5.4) inspected the supplied shader and wrapper. It returned no scoped functional issue, conditional on the documented nil-safe tensor binding/Close and checked dispatch helpers. It checked descriptor order, push count, ceil(N/256) geometry, guarded tails, exact in-place alias, lack of cross-element dependencies, rank/shape/extent checks, uint32 overflow, A&S coefficient ordering and the negative erfc branch.

The review flagged the wording that ±10 guards avoid overflow: 10 is a tail cutoff, not the F32 overflow threshold. The comment now states that the cutoff also bypasses arithmetic for extreme finite inputs. The original cutoff already prevented extreme-input overflow; no branch threshold changed.

After this review the parent extended `precise` to the final output arithmetic, rebuilt the shader and added regression assertions that all 20 basic floating operations carry `NoContraction`. The parent also added explicit rounding casts to the Go source model. The source-only review did not inspect this subsequent decoration extension, generated SPIR-V or surrounding pipeline barriers. Final static validation, exact rebuild and mock/source-model tests passed after these edits.

The parent checked core opcode arities and GLSL.std.450 values against the pinned Khronos grammars. Added tests reject malformed ordered-comparison arities, NoContraction with an operand, Tanh and unknown unary operations. The inspector remains a narrow metadata validator.

No device execution, GPU numerical comparison, race detector or trained-model assessment was performed by either reviewer.
