# Decision playground

This UI review slice requires the `/v1/systemone` API hand-off in #23 before production use. Its browser tests use a synthetic fixture.

The embedded playground at `/go-system-one` submits requests to both inference routes on the same loopback service. It opens with **TypeSafe questions**; the API selector also offers the existing **Batch decisions** interface.

## Question types

TypeSafe mode sends one `state` and a named `questions` object to `POST /v1/systemone`. Choose **JSON** for a structured object, array, quoted string or `null`, or **Plain text** to send the editor contents as a string. Each question has its own instructions and criteria; the [API contract](https://github.com/rcarmo/go-system-one/blob/ee905d0271a13569eda5449cfd03932e955337ca/docs/systemone-api.md) defines the supported shapes and limits.

* Noul shows the probability of yes and its complement, `1 - P(yes)`. It has no confidence field, threshold or selected boolean.
* Choice shows the selected label and all candidate probabilities in request order. The selected row is highlighted; local confidence is labelled separately.
* Score shows the fractional expected level, the numbered rubric and each level's probability. It does not highlight a level as the answer. Local confidence is labelled separately.

TypeSafe results show input/output token usage and **client HTTP** latency. The API returns no handler timing field; the browser measures the interval from `fetch` through JSON decoding. Candidate probabilities are constrained model probabilities, not calibrated correctness estimates, and confidence uses the documented local approximations.

Batch mode sends instructions, one context per line and a boolean/enum schema to `POST /v1/decision`. It retains automatic/tree controls, complete tree distributions, selected outcomes and handler timings. Greedy fallback explicitly reports when a full distribution is unavailable.

Switching API preserves each editor's draft and clears the previous result. Reset restores defaults for the selected mode. Request controls are disabled during inference to prevent a mode change from mislabelling an in-flight result. Validation and server errors clear stale results, and HTTP 429 permits a manual retry after the active request finishes.

## Appearance

The page uses the neutral OKLCH palette, system-font controls and monospace JSON editors. Appearance follows the operating-system `prefers-color-scheme` setting, with no theme toggle or persisted override. Mobile layout uses a single column.

### Light desktop

![Noul, Choice and Score results in the light desktop playground](images/go-system-one-light-desktop.png)

* OS preference: light; viewport: 1280 × 900; device pixel ratio: 1.
* Full-page output: 1280 × 1213.
* SHA-256: `a8dbdd5bb799063f4dae6ff886c909d7d8f888dfa8f858f4293f01dcbb64002d`.

### Dark mobile

![Noul, Choice and Score results in the dark mobile playground](images/go-system-one-dark-mobile.png)

* OS preference: dark; viewport: 390 × 844; device pixel ratio: 1.
* Full-page output: 390 × 2129; content width: 370 px, without horizontal overflow.
* SHA-256: `faa33baca87f91287fbed8e35c86bce1b85669d4adfdf1bf8b05d3db98fd9fd9`.

## Browser checks and screenshots

The model-free Playwright project under `webui/frontend/tests/go/` uses the real embedded page and a loopback-only Go fixture. The seven imported Chromium scenarios cover both API request bodies, typed answers and usage, the existing two-context boolean/enum results, OS themes, mobile overflow, text/null state, draft preservation, error recovery, in-flight control locking and HTML-safe labels/structured legends. Run the existing embedded Playwright suite as part of review; the upstream layout CI does not run it.

```sh
cd webui/frontend
PLAYWRIGHT_BROWSERS_PATH=/workspace/.cache/ms-playwright bun x playwright test --config playwright.go.config.ts
```

Regenerate both committed screenshots with:

```sh
cd webui/frontend
UPDATE_PLAYGROUND_SCREENSHOTS=1 \
PLAYWRIGHT_BROWSERS_PATH=/workspace/.cache/ms-playwright \
bun x playwright test --config playwright.go.config.ts
sha256sum ../../docs/images/go-system-one-{light-desktop,dark-mobile}.png
```

The capture tests submit the default TypeSafe request, assert all three answers, wait for request controls to unlock and save full-page PNGs. They replace the variable HTTP timing pill with **client HTTP: fixture timing**, clear hover/focus and disable transitions before capture. Two successive runs produced identical screenshot hashes on the recorded browser environment. The screenshots identify the synthetic backend. They were regenerated in this port and match the standalone hashes. No checkpoint or user data is loaded, and the screenshots are not performance or accuracy measurements. Update the dimensions and hashes above after regenerating them.

The original boolean/enum renderer came from [`go-pherence@c84a151dd8e7f952bc6b3aba35f1d309e79016f3`](https://github.com/rcarmo/go-pherence/commit/c84a151dd8e7f952bc6b3aba35f1d309e79016f3). The TypeSafe controls, rendering and screenshot tests are local changes in this repository; the source manifest retains the original import pin.
