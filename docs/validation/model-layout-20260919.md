# Source and checkpoint layout validation, 2026-09-19

All model source now lives under `model/`. BERT, Whisper, speaker (including
Community-1) and OmniVoice moved from `models/<family>` to `model/<family>`;
344 tracked files moved without changing the native kernels. Downstream imports
must use `github.com/rcarmo/go-pherence/model/<family>`.

Repository-local downloaded assets now live under the git-ignored `checkpoints/`.
The local move retained inode, size and relative path for all 234 weight/metadata
files; it did not copy or download weights. External paths, including
`/workspace/models`, `/opt/models` and a sibling `llama.cpp/models`, are unchanged.
Historical benchmark files and earlier audit reports retain their recorded paths.

The downloader and MiniCPM discovery helper default to `checkpoints/` and accept
`--checkpoints-dir`; `--models-dir` remains an alias. Make uses `CHECKPOINTS_DIR`,
with `MODELS_DIR` accepted when the new variable is not supplied. Root `models/`
is still ignored to prevent accidental commits from older checkouts; it contains
no source or compatibility wrappers. `cmd/models/` and `docs/models/` retain their
command and documentation roles.

## Checks

* Before moving, all five source packages under `models/` passed host tests/vet.
* After moving, those packages, their direct importers, speech-job HTTP APIs and
  modified import-boundary tests pass host tests/vet. The moved packages also
  pass the race detector and CGo-disabled tests with `GODEBUG=cpu.all=off`.
* Host `go build ./...` passes, including with CGo disabled.
* Linux/ARM64 and Linux/RISC-V builds pass for the moved packages, MOSS and the
  Whisper, diarize-vtt, speakercheck and OmniVoice CLIs. Test binaries for all five
  moved packages compile for both architectures; none were executed.
* `make model-layout-check` checks source placement, ignored asset roots, old
  import removal, Make defaults/aliases and both Python helpers using disposable
  directories and mocks, without downloading weights.
* Documentation links, generated diagrams, coverage snapshot and the speech
  quality freeze checks pass. Frozen model revisions and tensor evidence remain
  unchanged.

## Existing failures remain separate

`make host-test` retains the same nine model/MTP/Qwen failures recorded in
issues [#3](https://github.com/rcarmo/go-pherence/issues/3),
[#4](https://github.com/rcarmo/go-pherence/issues/4),
[#5](https://github.com/rcarmo/go-pherence/issues/5),
[#6](https://github.com/rcarmo/go-pherence/issues/6) and
[#7](https://github.com/rcarmo/go-pherence/issues/7).
Whole-tree vet still reports the NVIDIA unsafe-pointer and DiffusionGemma amd64
return-width diagnostics tracked in issues
[#8](https://github.com/rcarmo/go-pherence/issues/8) and
[#9](https://github.com/rcarmo/go-pherence/issues/9).

Broader checks also found two failures reproduced in a detached, unchanged
`5919c8e7` baseline checkout:

* Linux/ARM64 `go build ./cmd/audio/speechjobserve` fails because Community-1
  configuration/profile symbols are Linux/amd64-only while callers compile for
  other platforms. Targeted portable command builds are not a claim that the
  entire audio command tree cross-builds. Tracked in
  [#10](https://github.com/rcarmo/go-pherence/issues/10).
* `go test -race ./runtime/speechjob -run '^TestVulkanJobFatalDrainErrorQuarantines$'
  -count=3` reports a race between test reads and asynchronous quarantine-state
  writes. The ordinary suite passes, but the broader speech-job race gate does
  not. Tracked in [#11](https://github.com/rcarmo/go-pherence/issues/11).
  No full race-suite pass is claimed.

[Asset migration guide](../guides/model-assets.md) | [Validation gates](validation-gates.md)
