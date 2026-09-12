# Speech-job HTTP client

`speechjob` is a command-line client for an application embedding the [speech-job HTTP handler](../../../runtime/speechjob/httpapi/README.md). It does not start a server or load models. The current workspace's production speech services remain stopped; the examples below require a separately configured server.

## Commands

```sh
go build -o speechjob ./cmd/audio/speechjob
# Supply SPEECHJOB_TOKEN securely through your environment or secret manager.
export SPEECHJOB_URL=https://speech.example:8443

./speechjob upload --profile asr-pt --file recording.m4a
./speechjob list
./speechjob list --after JOB_ID
./speechjob get JOB_ID
./speechjob inventory
./speechjob --timeout 2h run JOB_ID
./speechjob cancel JOB_ID
./speechjob download --job JOB_ID --artifact vtt --out transcript.vtt
./speechjob delete --confirm JOB_ID
```

Global flags precede the command. `--url` overrides `SPEECHJOB_URL`. Tokens are read only from `SPEECHJOB_TOKEN`; there is no command-line token flag, credential URL or cookie support. The token must contain 32–256 printable non-space ASCII bytes and must be generated securely by the operator. The server's exact Host allowlist must include the endpoint host and port.

The timeout defaults to 30 seconds, includes the complete request/body and can be set from a positive duration to eight hours. `run` is synchronous and often needs a longer explicit timeout. SIGINT/SIGTERM cancels the request. Cancellation does not certify immediate server/model drain: inspect status before retrying. `cancel` only requests cancellation of an admitted matching run; HTTP 202 does not mean its durable status is already cancelled.

Upload requires a local regular non-symlink file of 1 byte to 512 MiB, an explicit server profile and optionally an inert display name. The default name is the local basename. The local file and its parent must remain stable during upload; this client does not create an upload snapshot or implement chunk resume. Server/profile limits may be lower. It sends exact declared bytes, never multipart or encoded media.

`list` retrieves one page of up to 100 jobs and prints the server's next cursor. There is no automatic polling or all-page scan. Delete requires `--confirm` and a job ID. No command performs application-level retries, and mutations are never marked replayable with an idempotency header or a body-replay function. Go's transport may retry idempotent reads after a stale pooled connection. A failed or lost upload acknowledgement can leave a retained job, so inspect inventory rather than blindly repeating the upload.

## Explicit queue commands

These commands require a queue-enabled server; synchronous `/run` is rejected in that mode:

```sh
./speechjob queue
./speechjob enqueue JOB_ID
./speechjob retry-queued JOB_ID
./speechjob forget-queued JOB_ID
```

`enqueue` acknowledges durable intent with a ticket; the operator must separately start the worker. Pending/running duplicates reuse their ticket. `retry-queued` explicitly creates a new ticket for failed/cancelled/interrupted work; complete jobs reject it. `forget-queued` removes terminal ticket metadata only and never deletes media. `cancel` withdraws pending intents durably or signals a running callback. Read the returned status and refresh manually; no automatic polling or retries occur. The request timeout only bounds the enqueue call, not accepted work.

Queue job deletion excludes concurrent enqueue; pending/running tickets must first be cancelled and drained. See [queue recovery](../../../runtime/speechjob/queue.md) for lost acknowledgements, interrupted claims and resource ownership. The current browser UI requires synchronous mode.

## Connection and output safety

HTTPS uses normal system trust and hostname validation; there is no insecure-TLS bypass. Plain HTTP is accepted only with `--allow-loopback-http` and a literal loopback IP such as `http://127.0.0.1:8090` or `http://[::1]:8090`. Hostnames including `localhost` do not receive this exception. Userinfo, endpoint paths other than `/`, query strings and fragments are rejected. Redirects and environment proxies are disabled; bearer credentials do not follow a Location header. The configured endpoint and host remain trusted operator inputs.

JSON responses are bounded to 4 MiB and printed as re-encoded JSON: terminal control characters and HTML are escaped, and numeric precision is retained. Error responses are limited to 64 KiB and reduced to HTTP status plus a fixed known error code. Unknown proxy/HTML/private error bodies and Location values are not printed. TLS/transport errors do not echo token or endpoint details. Output failure after an acknowledgement explicitly says the response was received; inspect status before retrying a mutation.

## Verified transcript downloads

Only `transcript`, `vtt`, `speaker-transcript` and `speaker-vtt` are accepted. The client first fetches job status and validates its ID and unique artifact size/SHA256. It then requires matching HTTP Content-Length, MIME type, identity encoding and SHA256 ETag, reads exactly the advertised bytes within 16 MiB, and verifies the computed digest. The authenticated server itself remains trusted; a hash delivered by a malicious server is not an independent provenance proof.

Downloads use a 0600 same-directory temporary file and never use a server-provided filename. After the complete digest and size match, the file is synced and published with a no-clobber hard link, followed by parent-directory sync. Existing files and symlinks are never replaced, including a destination created while a download is running. Success reports the saved path, byte count and SHA256 as JSON. No-clobber publication requires a local filesystem with hard-link and directory-sync support.

Before publication, failures and handled signals remove temporary files and leave the destination absent. SIGKILL/power failure can leave `.speechjob-download-*` scratch; cleanup is an explicit operator task. After publication, cancellation cannot retract the file. Directory-sync or stdout failure is reported as already-published output, preserving it for inspection. This is not a hostile same-UID directory sandbox, distributed-filesystem guarantee or power-loss qualification. Downloading experimental speaker VTT preserves its warning note; the client does not qualify speaker or speech accuracy.

## Verification

Run `make speech-job-cli-check`. Tests cover the real local HTTP handler lifecycle, endpoint/token/redirect/proxy policy, system TLS rejection, query escaping, upload byte identity, output bounds/redaction, SHA/length/ETag/MIME mismatch, partial downloads, cancellation, destination races, no-clobber publication, output-error acknowledgement, and SIGTERM cleanup in an actual child process. No trained models, public network requests, production listeners or service changes are needed. Test loopback listeners are temporary and close after each run.

ARM64/Windows cross-builds are compilation only. Filesystem publication is exercised on local Linux; the server store's Windows locking is unsupported. Queue commands and the separate synchronous browser UI are implemented for the tested synthetic scope. Weighted shared compute admission and the overall neural/quality/performance plan are unfinished.
