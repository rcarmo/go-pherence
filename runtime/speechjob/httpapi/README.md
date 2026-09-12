# Authenticated speech-job HTTP handler

`httpapi.New` returns a single-tenant `http.Handler` around one caller-owned speech store and fixed administrator-supplied profiles. It opens no listener or model. Optional queue metadata is opened/recovered during construction; execution requires a separate `StartQueue(ctx)` call. This is a tested HTTP boundary, not a deployed speech service or a trained-model qualification.

## Request contract

Every API route requires `Authorization: Bearer <token>`. The optional [browser UI](ui/README.md), disabled by default, serves only constant static assets without authentication; all its job/profile calls still require the token. Configure a cryptographically random printable token of at least 32 bytes. The handler retains its SHA256 digest and compares digests in constant time; configured length does not establish token entropy. There is no cookie, URL-token, ambient browser or anonymous authentication.

Configure exact Host values, including the port. Forwarded headers cannot override them. A supplied Origin must exactly match the configured scheme/host/port origin; no configured origin means all Origin-bearing requests fail. Fetch-Metadata values other than `same-origin`, `none` or absent fail. No CORS headers/preflight, redirects, escaped-path aliases or noncanonical routes are provided. Non-browser clients can omit Origin; possession of the bearer token authorises this single tenant.

All responses use `Cache-Control: no-store`, `nosniff`, no-referrer and a restrictive CSP. Errors contain stable codes, never raw callback errors, model paths, settings, upload bytes or stack traces. Status fields include ID, inert upload name, known profile ID/availability, status, attempts, active stage, input byte count, timestamps and allowlisted artifact metadata. Original configuration and filesystem/blob names are omitted. Clients must render upload names as text, not HTML.

| Method | Route | Behaviour |
|---|---|---|
| POST | `/v1/jobs?profile=ID&name=NAME` | Raw `application/octet-stream` upload; no content encoding or multipart form. Returns 201 and Location only after durable queued acknowledgement. |
| GET | `/v1/jobs?after=ID` | Up to 100 status records in lexicographic random-ID order; `next` cursor when more directories exist. Empty `after` starts traversal. |
| GET | `/v1/jobs/ID` | Current persisted status. |
| POST | `/v1/jobs/ID/run` | Synchronous explicit run/retry using exact stored profile identity. Returns after the store and owned callback finish. |
| POST | `/v1/jobs/ID/cancel` | Signals the admitted run, or durably withdraws a pending queue ticket. The response snapshot distinguishes running from cancelled. |
| GET | `/v1/queue` | Queue mode only: bounded ticket snapshots (up to 128), without configuration or paths. |
| POST | `/v1/jobs/ID/enqueue` | Queue mode only: 202 after durable intent; pending/running duplicates reuse a ticket. |
| POST | `/v1/jobs/ID/retry-queued` | Explicit retry of failed/cancelled/interrupted work; new ticket/sequence. Complete media jobs reject retry. |
| DELETE | `/v1/jobs/ID/queue` | Forget terminal ticket metadata only; pending/running work rejects this operation. |
| DELETE | `/v1/jobs/ID` | Explicit irreversible deletion, including incomplete uploads; busy stores reject it. Repeated deletion is idempotent. |
| GET | `/v1/inventory` | Retained IDs, byte counts and manifest/deletion/corruption flags, including incomplete uploads. No raw manifest/config data. |
| GET, HEAD | `/v1/jobs/ID/artifacts/NAME` | Verified transcript download; only `transcript`, `vtt`, `speaker-transcript` and `speaker-vtt`. |

All non-upload requests require an empty body. Unknown/duplicate query keys fail. Downloads are attachments with generated safe filenames, exact content length and SHA256 ETag. Range requests return 416. Raw upload, PCM, ASR-window, diarization, manifests and journal files are never exposed. Artifacts over 16 MiB are refused. Each artifact hash is verified before headers/body; failures after streaming starts abort the HTTP response rather than appending an error document. The server must not swallow `http.ErrAbortHandler` in recovery middleware.

Profiles bind the ID, exact configuration bytes and full ordered stage name/version list into the stored configuration. HTTP clients can select a profile but cannot send model paths, code, stage lists or configuration. The handler copies config and stage slices; closure-owned models remain immutable and externally synchronised. Changed or unavailable profiles return `profile_available:false` on status and 409 on run. Existing artifacts and explicit deletion remain accessible to the same authenticated tenant.

## Lifetimes and admission

There is one mutation slot, matching the store's serial executor. Concurrent upload/run/delete mutations receive 409. One to 64 ordinary request slots are configured; overflow receives 503. One additional bounded cancel slot prevents ordinary requests from starving cancellation. These request/executor bounds do not enforce rate limits or shared CPU/RSS/GPU budgets.

In default synchronous mode, runs belong to the HTTP request. Client disconnect and `Handler.Shutdown` cancel them cooperatively. A long request keeps its connection open; HTTP 202 is not used to claim durable background execution. Request timeout may cancel a run, and retry remains explicit. Application cancellation returns **409 `cancelled`**, not 408: browsers can automatically retry POST requests after HTTP 408. The explicit cancellation endpoint still returns 202 for its signal acknowledgement. A completed job remains a verified no-op only for its original full stage list.

The handler exclusively owns store access during use. Call `Handler.Shutdown(ctx)` before closing its store or models. It refuses new requests, cancels admitted requests and waits for them to drain. If the shutdown context expires, work may still exist; keep all resources alive and retry shutdown. Cancellation closes and joins the in-flight upload body closer. Synchronous filesystem calls and arbitrary stage callbacks remain cooperatively cancellable; the handler cannot forcibly interrupt them. It does not close the caller's store or listener.

`Store.ListPage` retains at most 100 manifests, checks cancellation between entries and skips incomplete uploads. Pagination is live, not a multi-page snapshot: concurrent create/delete operations may change later pages. The next page can be empty if only incomplete directories remain. Inventory retains bounded metadata for the store's configured job count.

## Opt-in queue

Set `Config.Queue` with a separate private `Directory`, `MaxEntries` (1–128), `MaxBytes` (256 KiB–4 MiB), `JobTimeout` (positive, at most eight hours) and nonnil cooperative `Admission`. Then call `StartQueue(ctx)` separately. Nil options retain synchronous mode. Queue mode rejects `EnableUI` and returns 409 `queue_enabled` for `/run`; the current browser only implements synchronous operations.

Accepted enqueue work is independent of the originating request. Shutdown cancels the worker and joins admission/run/release before closing its journal. Queue cancellation has a reserved request slot. Pending tickets are durable; running-at-crash tickets become interrupted and require explicit retry. Read [queue recovery and ownership](../queue.md) before embedding.

The worker acquires shared admission then the handler mutation gate; uploads/deletes receive 409 while it holds that gate. Enqueue can record previously uploaded jobs during a run. Uploading during execution remains unsupported by the serial store. Ticket removal plus media deletion holds the queue lock to exclude concurrent enqueue. Normal release finishes before the mutation gate opens. Release panic stops the queue and keeps mutations blocked until inspected restart; status/download/shutdown remain available.

`SerialAdmission` is an optional in-process exclusion callback for cooperating owners. It supplies no weighted CPU/memory budget, RSS enforcement, fairness or unrelated LLM/GPU coordination. Queue deadline includes admission wait and is separate from HTTP request timeouts. List order is storage order; `sequence` determines FIFO execution.

## Embedding requirements

The application must provide:

- TLS or an equivalent trusted local transport boundary, network ACLs and protection against token disclosure;
- HTTP header limits, read-header/idle timeouts, upload/read/write deadlines and slow-client controls;
- secure token generation/rotation and no secret logging;
- a private local store, immutable published payloads and appropriately bounded store quotas;
- trusted model/backend profile construction and shared compute admission before neural work;
- explicit HTTP listener and handler shutdown ordering, followed by model/store teardown;
- explicit queue options/worker startup if needed, shared admission and additional client/progress policy. The CLI and opt-in static browser UI are separate implemented clients; neither starts jobs automatically.

The handler authenticates before reading upload bodies, but a network server/proxy may still buffer bytes before dispatch. Uploads exceeding an unknown/chunked length are stopped by `MaxBytesReader`; incomplete directories remain visible for explicit cleanup. No automatic retry/deletion occurs; recovered pending queue intents require explicit worker startup. Exactly-at-limit uploads are allowed; unexpected read failures are not acknowledged as durable uploads. A publication error returns `persistence_uncertain`; clients should inspect status/inventory before deciding on retry. Blind POST-upload retries can create duplicates because uploads have no idempotency-key contract yet.

## Verification

Run `make speech-job-http-check`. Tests use synthetic stages and `httptest`, including one real loopback HTTP server; no production listener or model runs. They cover authentication/Origin/Host rejection before body reads, upload caps/disconnect, request/mutation limits, cancellation control admission, shutdown drain, status sanitisation, profile drift/copy ownership, artifact allowlisting/integrity/headers, error mapping, pagination, reopening and explicit resume. A Unix child is SIGKILLed through an HTTP run after a text checkpoint commits; reopening serves that checkpoint and resumes without repeating it. Process-death tests do not establish power-loss durability.

The core HTTP/store package cross-compiles independently of Linux/amd64 model adapters; cross-builds are not execution. Windows store locking is unsupported. No race result, multi-tenant authorisation, trained-quality, long-job proxy behaviour or production security audit is established by these tests. Separate Chromium UI checks are documented in the browser contract.
