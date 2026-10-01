# Legacy Whisper GGML storage

This loader reads the retained whisper.cpp GGML file format without inference or backend initialisation. It admits F32, F16 and Q5_0 tensors; quantised values widen through the existing GGML block decoder with no requantisation. This format is separate from GGUF.

`Open(ctx, path, sha256)` requires a regular file, a lowercase full-file SHA-256 pin and an immutable caller-owned path/inode. It hashes bounded chunks before parsing metadata. Bounds include 3 GiB file size, 100 million elements per tensor, 2,000 tensors and 4 GiB total logical F32 data (also bounded by host `int`). The header admits retained F32/F16/Q5_0 file types. Geometry, filter table, vocabulary lengths, tensor types/shapes, Q5 row alignment, duplicate names and trailing bytes are checked.

`Tensors`, `Filters` and `Vocabulary` return metadata copies. `Float32(ctx, name)` reads and decodes one tensor in bounded chunks, rejects nonfinite values and returns owned output. Errors and cancellation return no partial tensor. Calls serialize with `Close`; waiting for the mutex itself is not cancellable. File mutation after open is outside the contract; short/truncated reads and nonfinite data fail. The hash is checked at open, not repeated for every read.

Ordinary tests use tiny synthetic fixtures for all three types and exercise pinning, ownership, malformed geometry, duplicate/truncated records, nonfinite payloads, closure and every parser-read cancellation. Optional independent block/reference and retained-model tests require explicit paths and pins. No downloads or local model-cache access occur by default.

The private checked-model bridge in [Whisper](../../model/whisper/legacy_ggml_source.go) maps legacy names/shapes to logical HF-shaped F32 metadata for numerical qualification. It preserves original file/type provenance separately. It does not advertise Q5 storage as original F32, provide a packed-Q5 kernel or change production loading.

[Retained-model compatibility evidence](../../docs/validation/whisper-original-value-bridge-20261001.md).
