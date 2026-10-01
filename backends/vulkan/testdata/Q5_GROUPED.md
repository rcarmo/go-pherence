# Q5 grouped decode fixtures

`q5-grouped-input.bin` and `q5-grouped-output.f32` are copies of the independent 64-block/2048-value oracle retained in [original-value bridge evidence](../../../benchmarks/speech-foundations/whisper-original-value-bridge-20261001/).

- Input: 64 original GGML Q5_0 blocks, 22 bytes each. SHA-256 `2f540db499b6e8c083b0d8fc92d0b1f9f249d322c52012642264dc05a3589c55`.
- Output: 2048 little-endian float32 values. SHA-256 `d44a59144d4ea8153421604d78e1d87d07fd72a50d4586e8b2587988ede1ee15`.
- Generator: retained `q5-reference.cpp`, linked against original whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177` GGML CPU decoder. The fixture is independent of the new Vulkan repacking and shader.

The offline test reconstructs every value from the aligned six-word blocks and compares against this independent result. The native identity projection covers the same 2048 values, plus tiled projection checks. Fixture hashes establish input provenance; exact numerical checks enforce the explicit stored-value contract and do not establish acoustic accuracy.
