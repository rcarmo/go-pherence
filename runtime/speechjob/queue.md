# Explicit durable speech queue

`OpenQueue` opens a bounded local journal. Only `Start(ctx)` starts its single worker. Uploads and opening a store never enqueue work. Queue operation is opt-in; synchronous `Store.Run` and HTTP `/run` remain the defaults.

## Lifecycle

1. Open the private media store, then a separate private queue directory under an existing immutable administrator-owned parent. Both hold cooperating-process locks. The queue cannot reside in the media store.
2. Supply immutable `Resolve(Manifest)` stages and an explicit cooperative `Admission` callback. Neither resolver nor stage callbacks may re-enter queue methods. The owner coordinates all store mutations; direct concurrent `Store.Run`/deletion outside that owner is unsupported.
3. `Enqueue(ctx, id, false)` durably records input/configuration/ordered-stage hashes and expected attempt count. Pending/running duplicates return the same ticket. Acknowledgement means intent is durable; execution may still fail or never start.
4. `Start(ctx)` explicitly authorises execution of pending intents, including those recovered on open. The worker chooses the smallest pending sequence number. One worker exists per queue instance.
5. Admission precedes the durable running claim. Before running, verify the job hashes, expected attempts and current resolved plan. Changed jobs/profiles fail before inference. There is no automatic retry.
6. `Cancel` durably withdraws pending work. Running cancellation signals context and returns a running snapshot; it cannot promise a cancelled result when completion wins. The callback and admission release must drain before the worker moves on.
7. Failed/cancelled/interrupted tickets require `Enqueue(..., true)` to retry explicitly with a new ticket and sequence. Complete media jobs cannot run again through the queue, even when their queue status was interrupted before terminal publication.
8. `Forget` removes terminal ticket metadata only. `Delete` excludes concurrent enqueue while forgetting terminal metadata and deleting media; its caller also holds the store mutation gate. A failed media deletion can leave a job without a ticket. Inspect inventory and retry deletion explicitly.
9. `Shutdown` stops enqueue mutations, cancels running/waiting work and joins admission, stage and release callbacks. A timeout retains ownership. Retry shutdown before `Close`, then close store/models. Pending work waiting for admission at shutdown stays pending. Once claimed, cancelled work needs explicit retry.

## Recovery and failures

Each transition writes and fsyncs a private temporary JSON file, renames it over `queue.json`, then syncs the directory. Claim publication precedes execution. Reopening changes a running ticket to `interrupted`; it never automatically replays that claim. Pending tickets retain their identity but execute only after explicit worker startup. Completed tickets remain completed.

A crash after a media job commits but before its queue terminal transition leaves an interrupted ticket beside a complete media job. Inspect/download the verified artifacts, then forget the ticket. An explicit queue retry rejects the complete job.

After rename, an IO error means publication is uncertain. The queue instance stops new work and rejects mutations until the operator drains, inspects and reopens it. A worker claim/terminal persistence failure also stops the worker. An admission or release panic stops the queue because resource ownership is uncertain. In HTTP mode a release panic retains the mutation gate; restart only after inspecting resources. Ordinary admission errors become failed tickets and must release partial reservations before returning.

The tests use actual SIGKILL and injected IO failures. They do not simulate sudden power loss, hostile same-UID writers, broken fsync implementations or distributed filesystems. An acknowledged running cancellation is a cooperative signal, not durable withdrawal across process death; restart still marks its claim interrupted.

## Bounds and admission

- At most 128 entries; terminal entries count until explicitly forgotten.
- At most 128 KiB canonical JSON; directory byte budget 256 KiB–4 MiB including old journal, orphan files and temporary-publication headroom. This budget is separate from media-store bytes.
- At most 256 directory entries, counted in batches of 32. Writes reserve one temporary entry. Orphans are retained for explicit operator inspection/cleanup, never automatically removed on open.
- Each job deadline is positive and at most eight hours, including admission wait. File IO and callbacks are cooperative and may drain after the deadline.
- `List` returns a detached bounded snapshot; the 128-entry allocation regression checks one slice allocation. No inference-speed or RSS improvement is established by this measurement.

`SerialAdmission()` supplies a process-local exclusive gate. Sharing the same callback can exclude cooperating owners on different stores; separate callback instances do not coordinate. It has no weighted CPU/memory accounting, fairness, hard RSS/CPU enforcement, GPU admission or control over unrelated LLMs. The standalone server owns one serial gate, not a host resource controller. Model loading occurs before queue admission; its separate metadata/weight limits still apply.

## Interfaces

See the [HTTP queue routes](httpapi/README.md), [CLI](../../cmd/audio/speechjob/README.md) and [server configuration](../../cmd/audio/speechjobserve/README.md). Queue mode rejects the current synchronous browser UI. The trained-model, long-recording, shared-resource and numerical/performance gates are separate unfinished work.
