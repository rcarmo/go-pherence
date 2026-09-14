# Opt-in speech-job browser interface

The embedded browser interface provides upload, paged job inspection, explicit run/cancel/delete, retained inventory and verified transcript downloads. It is disabled by default. No production services are enabled or restarted by adding these assets.

## Enable explicitly

For an application embedding `httpapi`, set `EnableUI:true`, an exact `Origin`, an allowlisted matching host/port and at least two ordinary request slots. For `speechjobserve`, set `http.enable_ui:true` and `http.origin` to the exact listener origin. Its TLS/plain-loopback policy must match the origin. Open `/ui` or `/ui/` on that origin. A token is still required for every API request. Normal CLI/HTTP-only operation is unchanged when UI is disabled.

The only unauthenticated endpoints are the fixed embedded `/ui`, `/ui/`, `/ui/app.js` and `/ui/style.css` assets. They contain no job data, profile configuration or credentials. Host, actual TLS scheme, Origin/Fetch Metadata, exact path/query/method, request admission and shutdown checks still apply. Responses inherit `Cache-Control:no-store`, `nosniff` and no-referrer, use self-only scripts/styles/connections, block framing/base/form/object use and disable camera/microphone/geolocation. There is no CDN, external library, inline script or CORS exception.

Authenticated `GET /v1/profiles` returns sorted profile IDs and the upload byte cap only, and is available only with UI enabled. Model paths, raw configuration and stage closures stay private. Profile definitions remain administrator-owned.

## Browser lifecycle

The token is held only in page memory and the password field is cleared after connection. Nothing is written to cookies, local/session storage or URL parameters. Requests omit cookies, require same-origin fetches, reject redirects, disable caching, and carry the bearer header. At most six ordinary fetches and one cancellation fetch are active from one page; the server imposes its own lower limits. A stale page/request epoch cannot update a new authentication session.

Forget token aborts owned requests, clears job/profile data and revokes download URLs. Authentication/Origin/Host rejection also clears connected state. Navigation/reload/pagehide clears credentials for privacy, including BFCache suspension. Background-tab visibility alone does not trigger pagehide, but mobile/browser lifecycle interruptions can cancel a request. There is no durable browser session or promise of background continuation. Before leaving an active run/upload, supported browsers show a navigation warning; after reconnecting, inspect durable status/inventory.

Runs are synchronous and explicitly clicked. Closing/reloading or forgetting credentials may cancel them; server callbacks still need to drain. The page uses manual status refresh rather than polling or inventing percentage progress. No automatic run, retry, queue or upload replay is implemented. A lost upload acknowledgement may leave a stored job; inventory supports explicit inspection/deletion instead of blind retry. Job deletion requires a confirmation dialog, including incomplete retained jobs.

**Wire correction:** application cancellation now returns HTTP **409** with error code `cancelled`, replacing 408. Chromium can retry a POST receiving HTTP 408, which violates explicit-run semantics. The real-browser test verifies cancellation returns to a retryable UI with exactly one server attempt. The API `POST /cancel` still returns 202 to acknowledge the signal only; the CLI already interprets stable error codes independently of HTTP status.

## Text and downloads

Uploaded names, job fields and profile IDs are rendered with `textContent`, never `innerHTML`. Results are not injected as markup. MIME types and artifact names are allowlisted; raw PCM/uploads, raw ASR windows, diarization internals and store manifests have no browser download link and remain forbidden by the API.

Before a download, the page fetches fresh authenticated status, checks job/artifact identity, size, MIME, ETag and exact length, then computes SHA256 with Web Crypto. A mismatch creates no Blob URL. After verification, a generated filename containing only job ID/allowlisted artifact name is handed to the browser. Browser download settings control final placement and collision handling; unlike the CLI, a web page cannot promise no-overwrite filesystem publication. The server and TLS origin remain trusted; server-provided hashes are not independent authenticity proof.

The page holds at most one completed verified Blob (up to 16 MiB) until the next completed download or session reset, avoiding premature timer revocation. In-flight content is bounded to 16 MiB/artifact or 4 MiB/JSON, with one download at a time. Reading/copying/hash/Blob construction can briefly retain several bounded byte buffers; no zero-allocation claim is made. New data/status views replace old DOM trees, and streams/controllers are cancelled/released on errors and resets.

## Verification

`make speech-job-ui-check` builds a gated model-free HTTP test server and runs `scripts/speechjob-ui-check.ts` in Chromium through Playwright. Set `SPEECHJOB_BROWSER_OUT` and, when Playwright is installed outside the repository, `PLAYWRIGHT_MODULE` to its absolute ESM entrypoint. Chromium must already be installed. The fixture uses a temporary store and random loopback port, fixed synthetic stage results and a test-only token; no trained model, production server or persistent service is used.

The browser scenario checks authentication, CSP/no-store, no credential storage, hostile-name rendering, upload/run/download, corrupted downloads, delete confirmation, retained text after later failure, cancellation without replay, source-media denial, lost upload acknowledgement/inventory, reload/forget/auth rotation and mobile layout. Go tests cover static opt-in and route/security headers, authenticated profile metadata, disabled API behaviour, matching server config and shutdown admission. Screenshots are synthetic fixture evidence, not a deployed UI.

The interface does not qualify model accuracy, trained resume/long-file performance, general overlap/word alignment, multi-tenant authorisation, persistent queueing or shared CPU/RSS/GPU admission. Only Chromium is currently exercised; Safari/Firefox/mobile lifecycle behaviour remains unqualified.
