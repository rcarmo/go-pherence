# Offline Vulkan shader validation

All 11 exact embedded modules and all 11 GLSL rebuilds pass SPIRV-Tools 2023.1 validation for `vulkan1.3`. This did not load a Vulkan driver or execute a shader.

The checked-in `.spv` files match the embedded Go bytes exactly. Rebuilding with glslang 12.0.0 produces different bytes for all 11 modules. Every rebuild matches after ignoring only the generator header word and sorting the `OpDecorate`/`OpMemberDecorate` multiset; all other instructions, IDs, operands and their order must remain identical. This narrow comparison is not general semantic equivalence. Exact byte mismatches are retained in [verification.json](verification.json).

## Reproduce

Install or provide offline `spirv-val` and `glslangValidator`, then run from the repository root:

```sh
SPIRV_VAL=/path/to/spirv-val \
GLSLANG_VALIDATOR=/path/to/glslangValidator \
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-shader-report
```

The report directory must not exist. Missing tools fail with a failed report rather than skipping validation. The target runs six Bun tests with 45 assertions, then validates the exact embedded bytes, rebuilds GLSL and validates the rebuilt bytes. It does not alter shader sources, embedded arrays or stored binaries. Temporary shader files are removed on success and failure. The script carries checked `SCRIPT_JDOC` metadata.

[Checkpoint evidence](evidence.json) records script/source/tool hashes, versions, package provenance and regression scope. The tools were extracted locally under the workspace `tools/vulkan-offline` directory from Debian packages because this host lacks Homebrew/dpkg/apt. No system installation, service change or container execution was needed. The binaries are not committed in this repository. Package hashes identify the downloaded artefacts; this record is not a package-signature attestation.

## Semantic findings and unchanged holds

### No-scale RMSNorm

The source and [embedded disassembly](disassembly/rms_norm_no_scale_f32.spvasm) declare two buffers (`X`, `Out`) and push members `n`, `eps`. The legacy wrapper supplies one buffer and therefore still fails layout admission.

Source review found that exact input/output range aliasing is valid for a **single workgroup**: all invocations finish reading the reduction inputs before the final workgroup barrier, then each invocation reads and writes only its own output-loop elements. Shifted overlap is unsafe, and multiple workgroups process the same elements because the shader does not use `gl_WorkGroupID`. The existing wrapper dispatches one workgroup, but no alias-binding repair or numerical test was implemented in this checkpoint. The hold stays in place.

### Partial RoPE

The source and [embedded disassembly](disassembly/rope_partial_f32.spvasm) have an in-place paired read/write race. One invocation computes the first component while another computes the second; each reads both original components and writes one, without synchronization or staging those reads. Either can observe the other's updated value. Static SPIR-V validation does not detect this data race.

The host wrapper also disagrees with the shader:

| Field | Shader | Host wrapper |
|---|---|---|
| Descriptors | Q, Cos, Sin | X, interleaved frequencies |
| Push word order | HeadDim, RotHalf, Heads, Pos | Pos, Heads, HeadDim, RotHalf |
| Invocation indexing | One invocation per head component | Groups sized from head/rotation pairs |

The existing two-versus-three descriptor mismatch already rejects pipeline creation. Supplying a third binding would not fix the push layout, frequency representation, dispatch geometry or race. The shader and wrapper were left unchanged.

## Review and verification scope

A narrow delegated source review confirmed the RMS alias conditions and RoPE race. Its host-wrapper findings were conditional because only the two shader files were supplied; the parent checked the actual wrapper source separately. Another review found no false-success path in the checker but identified incomplete failure reporting and cleanup. Those issues were fixed, and a missing-tool/no-clobber regression test was added.

The focused Vulkan mock suite and speech affine/foundations/media targets pass. No runtime Go code changed in this checkpoint; the previous 81-top-level/383-pass-event plan suite remains the runtime baseline rather than a new test-count claim. Vulkan/board vet passes. Full-tree build failures and the missing race compiler are unchanged baseline limitations; neither was represented as fixed.

Passing `spirv-val` establishes static module validity for the selected validator and target. It does not prove race freedom, descriptor/parameter agreement, F32/BF16 semantics, model quality, device compatibility or speed. No GPU, trained checkpoint, private media, service, deployment or restart was used.
