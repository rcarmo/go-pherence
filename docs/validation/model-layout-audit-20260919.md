# Model layout follow-up audit, 2026-09-19

The follow-up checked source paths and asset defaults across the 4,214 tracked
files present at the start of the audit, including code, Make targets, generators,
fixtures, documentation examples and CI scaffolding. It found three live gaps:

* Two current Gemma4 command examples still used `$PWD/models`.
* The default Gemma4 MTP parity fixture still supplied executable `models/` paths.
  Those paths now use `checkpoints/`; tokens, logits and all other fixture values
  are unchanged.
* The layout check was standalone and only checked Git-tracked Go imports. It
  missed new/ignored files and was not part of the normal validation targets.

## Enforcement

`docs/model_layout_test.go` now walks the worktree, including new source,
foreign-platform files, templates and CI scaffolding. It checks old imports,
repository-local asset paths, joined paths, downloader defaults, live JSON
fixtures and fenced command examples. Twenty-three mutation cases cover regressions,
including files hidden by the legacy `/models/` ignore rule. The Go guard needs
no Git, Bun, Python, model weights or hardware.

`make model-layout-check` adds the mocked Python/default/alias tests.
`make docs-check` includes it, and `make host-check` runs it before build/vet/test.
The GitHub layout workflow runs `make docs-check` on pushes and pull requests
without weights. The [first hosted run](https://github.com/rcarmo/go-pherence/actions/runs/35436280379)
passed on a clean Ubuntu runner without checkpoints, in addition to the local
checks below.

Checkpoint payloads are not traversed. External stores, `cmd/models/`,
`docs/models/`, upstream Python paths, metadata keys and historical numerical
records retain their meanings. Three archived benchmark reproducers and two
DiffusionGemma reference reports have exact exceptions. New executable scripts
in benchmark, history or experiment directories are still checked.

## Verification

The Go guard and CLI fixture regression pass tests, vet and race checks. The
compiled guard also passes with `PATH=/nonexistent`, proving it does not depend
on external tools. Its Linux/ARM64 and Linux/RISC-V test binaries compile but were
not executed. The Gemma4 default fixture test, host build, generated diagrams,
coverage snapshot, speech quality freeze and Markdown link checks pass.

Whole-tree host testing retains the same nine failures tracked by issues
[#3](https://github.com/rcarmo/go-pherence/issues/3) through
[#7](https://github.com/rcarmo/go-pherence/issues/7); the layout tests pass within
that run. The NVIDIA and DiffusionGemma vet findings in
[#8](https://github.com/rcarmo/go-pherence/issues/8) and
[#9](https://github.com/rcarmo/go-pherence/issues/9) are unchanged. This audit does
not claim a green full repository or new runtime parity/performance results.

[Asset conventions](../guides/model-assets.md) | [Original migration validation](model-layout-20260919.md)
