# Embedded llama.cpp UI

This is the llama.cpp Svelte UI, not a recreation. The component tree, CSS, interface icons, hash routes, browser storage, chat editor, settings and import/export screens come from upstream. The sidebar name/logo and browser favicon use go-pherence branding, as requested; the icon is a 64px rendition of `docs/icon-256.png` embedded in an SVG wrapper. Bun builds static assets; Go embeds and serves them. There is no production JavaScript server.

The frontend and the inference API are different things. This port preserves the upstream UI, but does **not** give go-pherence every llama-server capability. Unsupported inference controls return errors rather than silently changing the request.

## Use

Add `-webui` to either existing server invocation:

```sh
go build -o bin/llmserver ./cmd/llm/llmserver
go build -o bin/diffusiongemmaserver ./cmd/diffusiongemmaserver
# Add -webui to your normal model/backend flags, then open / on that server.
```

The flag defaults to false. Existing `/v1` routes are unchanged. These servers do not gain authentication from the UI's API-key field: bind to loopback or put an authenticated reverse proxy in front of **all** routes. Conversations and settings are stored in the browser, as upstream does; clearing site data removes them.

## Build and test

Generated `dist/` files are committed so `go build` works without Bun, npm, Node or network access. To change the frontend, use Bun 1.4.1 (the tested version):

```sh
make webui-build                 # frozen Bun lockfile, static build
make webui-check                 # Svelte diagnostics and upstream unit tests
cd webui/frontend
bun x playwright install chromium
cd ../..
make webui-test                  # Go race tests and synthetic Chromium scenarios
```

Set `PLAYWRIGHT_BROWSERS_PATH` if using a shared browser cache. The browser fixture binds only to `127.0.0.1:18181`, runs inside a Go test, and never loads a model. The frontend's nested `go.mod` excludes Go examples shipped in JavaScript dependencies from the parent Go package scan.

`make webui-build` consumes `frontend/bun.lock`; the original npm lockfile is retained for provenance, not used by that target. A fixed SvelteKit version string removes the wall-clock build ID. On updates, rebuild the assets and review the source diff together. Tests do not qualify any native backend.

## Reuse

`webui.NewHandler()` serves only the embedded index, favicon and Svelte assets. `NewHandlerFS` accepts a caller-owned filesystem for testing or other static builds. No source files, directory listings, source maps or wildcard SPA fallback are exposed; navigation uses hash routes. HTML is revalidated; hashed assets are immutable.

`webui.Register(mux, Config{...})` additionally installs `/props` and `/webui/v1/chat/completions`. Supply your existing `http.Handler` as `ChatHandler`; no network proxy is involved. The response writer and request context pass through, preserving SSE flushes and cancellation. Wrap the entire mux in any authentication middleware. `CurrentModel` optionally reports an API-side model switch; callbacks must synchronize their own state.

The generic asset handler is independent of this compatibility layer and the frontend source.

## Compatibility boundary

- `/props` advertises a single loaded text model, one slot, and the available context/default token bounds. Unknown context size is omitted. It does not pretend to be llama-server router mode or advertise vision, audio, video or a chat template.
- String content and text-only content arrays are forwarded with system, developer, user and assistant roles. Nonempty tools, tool calls, tool/function roles and media parts are rejected. Historical reasoning is not concatenated into visible prompt text.
- Requests are limited to 1 MiB and 256 messages. Omitted, zero and `-1` token limits use the configured bounded default, not unlimited generation. `llmserver` defaults/caps at 4096; DiffusionGemma uses `-max-new` as its default and the larger of that value and 4096 as the UI cap. Backend/context admission still applies.
- The adapter accepts only zero/omitted temperature and neutral sampling values. It removes neutral temperature because the DiffusionGemma API does not accept that field. DiffusionGemma continues to use its configured denoiser and seed: this does **not** expose an autoregressive sampler for it.
- Nondefault samplers/penalties, grammar, thinking budgets, prompt continuation, pre-encoding and unknown custom parameters return HTTP 400 with a displayed explanation. Requests for optional timings/progress/control handles are tolerated but do not manufacture those fields. Existing server SSE timing and cancellation behavior is unchanged; this does not make diffusion inference preemptible.
- Model load/unload, slots, completion control, built-in tools and the CORS proxy are not implemented by this package. The upstream single-model UI remains available; this is not a model router. MCP/settings controls remain upstream code, but backend tool execution is unsupported.
- Title generation keeps the upstream default and setting. No adapter override disables the control. Browser-local features remain upstream implementations; only the scenarios in the validation report have been exercised here.

## Provenance and licence

Source: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), commit [`4a6735f1cf0594250958bcc839267696c7b998a4`](https://github.com/ggml-org/llama.cpp/tree/4a6735f1cf0594250958bcc839267696c7b998a4/tools/ui), `tools/ui/`. Copied using `git archive`, not the source worktree; its pre-existing local changes were preserved.

Copyright (c) 2023–2026 The ggml authors. The upstream [MIT licence](LICENSE.llama-cpp) is retained and embedded at `/LICENSE.llama-cpp.txt` so binary distributions carry it too. Dependencies retain their own licences in their distributions.

Local changes to copied files:

- `vite.config.ts`: leave Svelte assets hashed instead of flattening them for C++; add the development compatibility-route proxy.
- `svelte.config.js`: output to the sibling embedded directory; fix the build ID for reproducibility.
- `src/lib/constants/api-endpoints.ts`: send chat to the separate compatibility route.
- `src/lib/constants/ui.ts`, `SidebarNavigation.svelte` and `static/favicon.svg`: go-pherence name and existing project artwork for the sidebar and browser tab. Storage keys stay unchanged, preserving existing conversations/settings.
- `src/routes/settings/[[section]]/+page.svelte`: defer the default-section `replaceState` until SvelteKit's root exists on direct hash loads. No layout or navigation redesign.
- `src/lib/utils/model-names.ts`, its unit test and `vitest-setup-client.ts`: repository-conforming example/fixture paths only.
- `README.md`: point three links at the pinned upstream documents outside this vendored subtree.

The Bun lockfile, Go module boundary, static licence copy, Go-hosted Playwright configuration/tests and this document are local additions. All other copied upstream files are unchanged. See [the validation report](../docs/validation/embedded-llama-ui-20260920.md) for tested behavior and exclusions.
