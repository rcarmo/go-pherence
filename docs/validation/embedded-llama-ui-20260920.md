# Embedded llama.cpp UI validation — 2026-09-20

The upstream llama.cpp UI is now built with Bun and embedded in Go. This is a source port of commit `4a6735f1cf0594250958bcc839267696c7b998a4`, not a visual approximation. Both `cmd/llm/llmserver` and `cmd/diffusiongemmaserver` expose it when started with `-webui`; the default remains API-only.

The [package README](../../webui/README.md) records build instructions, attribution, changed upstream files and API limits. The MIT licence is included in the source and the embedded distribution. The generic static handler is separate from the frontend and the chat compatibility adapter.

## Evidence

| Check | Result |
|---|---|
| Pinned-source comparison | All 576 upstream files retained; eight files adapted, with local additions listed in the package README. No components, styling or icons redesigned. |
| Source checkout preservation | Before/after `git status --porcelain=v1` identical for the dirty llama.cpp checkout; copied with `git archive`. |
| Build | Bun 1.4.1, frozen lockfile; two successive builds produced identical SHA-256 hashes for every output asset. |
| Frontend diagnostics | Svelte check: zero errors, zero warnings. |
| Upstream unit tests | 13 files, 199 tests passed. |
| Chromium | Two synthetic Go-hosted scenarios passed three consecutive times (six executions). |
| Affected Go packages | Race tests passed for `webui`, `cmd/llm/llmserver`, `cmd/diffusiongemmaserver`. |
| Whole repository | NVIDIA-disabled `go test -race -p=2 ./... -count=1 -timeout=180s`: 111 passed, 54 no-test packages. |
| Static checks | `go vet ./...`, `go build ./...`, `git diff --check` passed. |
| Foreign builds | Linux ARM64 and RISC-V, CGo disabled: compilation passed, not native execution. |
| Script tests | 22 tests, 2647 assertions passed. |
| Documentation | `make docs-check`: layout/tests passed, 354 Markdown files checked, no broken local links. |

The browser tests load the actual embedded assets through Go, obtain `/props` and model metadata, submit a streamed chat request through the compatibility adapter, and verify the visible reply and IndexedDB persistence after a hard reload. They check desktop/mobile rendering, absence of horizontal overflow at 390px, synthetic server errors, direct settings hash navigation, saved sampling values, import/export screen navigation and a displayed unsupported-temperature error. They assert no uncaught page errors.

Go tests exercise asset allowlisting, MIME/cache/security headers, HEAD and methods, missing paths, source-map/traversal rejection, actual bundle references and embedded licence availability. Adapter tests cover the 1 MiB/256-message boundaries, malformed/trailing JSON, text arrays, unsupported tools/media/roles/controls, bounded token defaults, SSE flushing, request context/headers and preservation of the original request. Server integration tests cover opt-in routes, unchanged strict API behavior and live model/context metadata.

## Corrections during validation

The upstream C++ bundle-flattening plugin was removed from the build configuration so the Go handler serves Svelte's hashed assets. Direct loading of `/#/settings` exposed an upstream `replaceState` call before SvelteKit's root existed; waiting for `tick()` fixed that without changing the screen or its routing behavior.

Early browser checks failed on optional `backend_sampling:false` and timing requests; the adapter now explicitly accepts their neutral/optional forms. An early persistence test reloaded as soon as the last text chunk rendered, before the asynchronous IndexedDB write completed. It now polls for the durable message before reloading; repeated final runs passed. Settings tests use the upstream Save button, not an assumed autosave behavior.

The first full race attempt exceeded its outer tool timeout and is not counted. Subsequent complete runs passed. A frontend module boundary excludes Go examples inside npm dependencies from repository package discovery. Copied model-path examples were adjusted to satisfy the existing layout guard; it was not weakened. Three documentation links now point at the pinned upstream files outside the copied subtree. Git whitespace checks exclude generated bundles and three upstream Markdown-in-template story fixtures: trimming whitespace inside those strings would change their content.

An independent delegated review timed out and supplied no findings; it is not counted as review coverage.

## What this does not establish

The UI does not implement llama-server's inference stack. It advertises one text model, not router mode or multimodal support. Non-neutral unsupported controls fail explicitly. The existing strict OpenAI routes remain separate. UI title settings, browser-local functionality and other upstream controls are retained, but only the scenarios above were tested: this is not exhaustive interactive-feature qualification or a pixel-diff comparison against a running native llama-server.

There was no model load, inference smoke test, native-device execution, authentication deployment or service restart. The upstream Storybook/client suite and its old demo E2E test were not run; the Go-hosted browser scenarios are the integration gate. Cross-builds are not hardware-safety evidence.

Evaluation #2 remains frozen at **676/1,440**. Both frozen binaries, all **16** freeze file entries and all **676** record hashes were verified unchanged. No GPU recovery, replay or tuning was attempted. Audit #16 and platform issues #17–#21 remain open; this UI work does not close them.
