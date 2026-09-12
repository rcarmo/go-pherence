package speechjob

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var ErrQueueState = errors.New("speech queue state conflict")

// Admission reserves shared resources until its returned release function runs.
// Implementations must honour context and release partial reservations on error.
// The callback is cooperative; it cannot enforce OS memory/CPU or unrelated work.
type Admission func(context.Context) (release func(), err error)

// SerialAdmission creates a shared in-process gate. Pass the SAME callback to all
// cooperating job owners that must exclude one another. No process/global magic,
// memory enforcement, fairness or forced interruption is provided.
func SerialAdmission() Admission {
	gate := make(chan struct{}, 1)
	return func(ctx context.Context) (func(), error) {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		select {
		case gate <- struct{}{}:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		var once sync.Once
		return func() { once.Do(func() { <-gate }) }, nil
	}
}

// QueueConfig caps metadata separately from the Store's media quota. Directory
// must be a private sibling/outside the store, never inside it. Resolve supplies
// immutable trusted stages for the manifest's EXACT configuration; it runs no
// inference and must not re-enter Queue methods. Admission includes any shared
// compute and caller mutation gate. Stage callbacks must not re-enter the queue.
// Neither OpenQueue nor Enqueue starts a worker. Start is a separate consent.
type QueueConfig struct {
	Directory  string
	MaxEntries int
	MaxBytes   int64
	JobTimeout time.Duration
	Resolve    func(Manifest) ([]Stage, error)
	Admission  Admission
}
type QueueStatus string

const (
	QueuePending     QueueStatus = "pending"
	QueueRunning     QueueStatus = "running"
	QueueSucceeded   QueueStatus = "succeeded"
	QueueFailed      QueueStatus = "failed"
	QueueCancelled   QueueStatus = "cancelled"
	QueueInterrupted QueueStatus = "interrupted"
)

type QueueEntry struct {
	JobID            string      `json:"job_id"`
	Ticket           string      `json:"ticket"`
	Sequence         uint64      `json:"sequence"`
	Status           QueueStatus `json:"status"`
	InputSHA256      string      `json:"input_sha256"`
	ConfigSHA256     string      `json:"config_sha256"`
	PlanSHA256       string      `json:"plan_sha256"`
	ExpectedAttempts int         `json:"expected_attempts"`
	Updated          time.Time   `json:"updated"`
	ErrorCode        string      `json:"error_code,omitempty"`
}
type queueState struct {
	Schema  int          `json:"schema"`
	Next    uint64       `json:"next"`
	Entries []QueueEntry `json:"entries"`
}

// Queue owns one journal lock and at most one worker. Store/profile/model access
// must remain exclusive to its coordinating owner. All public state is detached.
// Persistence uncertainty poisons this instance: stop, inspect and reopen; it
// never silently runs work whose durable claim is uncertain.
type Queue struct {
	mu                        sync.Mutex
	store                     *Store
	root                      *os.Root
	lock                      *os.File
	cfg                       QueueConfig
	state                     queueState
	poison                    error
	started, stopping, closed bool
	ctx                       context.Context
	cancel                    context.CancelFunc
	active                    string
	activeCancel              context.CancelFunc
	wake                      chan struct{}
	done                      chan struct{}
	fault                     func(string) error
}

const maxQueueJSON = 128 << 10

func OpenQueue(store *Store, cfg QueueConfig) (*Queue, error) {
	if store == nil || cfg.Resolve == nil || cfg.Admission == nil || cfg.MaxEntries < 1 || cfg.MaxEntries > 128 || cfg.MaxBytes < 2*maxQueueJSON || cfg.MaxBytes > 4<<20 || cfg.JobTimeout <= 0 || cfg.JobTimeout > 8*time.Hour {
		return nil, fmt.Errorf("invalid queue configuration")
	}
	dir, e := filepath.Abs(cfg.Directory)
	if e != nil || cfg.Directory == "" {
		return nil, fmt.Errorf("queue directory required")
	}
	// Resolve parents to prevent an ordinary symlink alias into the store. Both
	// parents remain administrator-owned/immutable; not hostile same-UID defence.
	parent, e := filepath.EvalSymlinks(filepath.Dir(dir))
	if e != nil {
		return nil, e
	}
	dir = filepath.Join(parent, filepath.Base(dir))
	storeDir, e := filepath.EvalSymlinks(store.root.Name())
	if e != nil {
		return nil, e
	}
	rel, e := filepath.Rel(storeDir, dir)
	if e != nil || rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))) {
		return nil, fmt.Errorf("queue must be outside media store")
	}
	if e = os.Mkdir(dir, 0700); e != nil && !errors.Is(e, os.ErrExist) {
		return nil, e
	}
	st, e := os.Lstat(dir)
	if e != nil || !st.IsDir() || st.Mode().Perm()&0077 != 0 {
		return nil, fmt.Errorf("queue directory must be private")
	}
	p, e := os.Open(parent)
	if e != nil {
		return nil, e
	}
	if e = errors.Join(p.Sync(), p.Close()); e != nil {
		return nil, e
	}
	root, e := os.OpenRoot(dir)
	if e != nil {
		return nil, e
	}
	fail := func(e error) (*Queue, error) { root.Close(); return nil, e }
	if st, e := root.Lstat(".lock"); e == nil && !st.Mode().IsRegular() {
		return fail(ErrCorrupt)
	} else if e != nil && !errors.Is(e, os.ErrNotExist) {
		return fail(e)
	}
	lock, e := root.OpenFile(".lock", os.O_CREATE|os.O_RDWR, 0600)
	if e != nil {
		return fail(e)
	}
	if e = lockStore(lock); e != nil {
		lock.Close()
		return fail(e)
	}
	q := &Queue{store: store, root: root, lock: lock, cfg: cfg, wake: make(chan struct{}, 1), done: make(chan struct{}), state: queueState{Schema: 1, Next: 1, Entries: []QueueEntry{}}}
	if e = q.open(); e != nil {
		lock.Close()
		root.Close()
		return nil, e
	}
	return q, nil
}
func (q *Queue) open() error {
	if _, e := q.usage(); e != nil {
		return e
	}
	f, e := q.root.Open("queue.json")
	if errors.Is(e, os.ErrNotExist) {
		return q.save(q.state)
	}
	if e != nil {
		return e
	}
	b, e := io.ReadAll(io.LimitReader(f, maxQueueJSON+1))
	e = errors.Join(e, f.Close())
	if e != nil {
		return e
	}
	var state queueState
	if len(b) > maxQueueJSON || json.Unmarshal(b, &state) != nil {
		return ErrCorrupt
	}
	canonical, _ := json.Marshal(state)
	if !bytes.Equal(b, canonical) || state.Schema != 1 || state.Next == 0 || len(state.Entries) > q.cfg.MaxEntries || state.Entries == nil {
		return ErrCorrupt
	}
	ids := map[string]bool{}
	tickets := map[string]bool{}
	sequences := map[uint64]bool{}
	changed := false
	for i, entry := range state.Entries {
		if !validID(entry.JobID) || !validID(entry.Ticket) || entry.Sequence == 0 || entry.Sequence >= state.Next || sequences[entry.Sequence] || ids[entry.JobID] || tickets[entry.Ticket] || entry.Updated.IsZero() || !validHash(entry.InputSHA256) || !validHash(entry.ConfigSHA256) || !validHash(entry.PlanSHA256) || entry.ExpectedAttempts < 0 {
			return ErrCorrupt
		}
		ids[entry.JobID] = true
		tickets[entry.Ticket] = true
		sequences[entry.Sequence] = true
		switch entry.Status {
		case QueueRunning:
			state.Entries[i].Status = QueueInterrupted
			state.Entries[i].ErrorCode = "process_interrupted"
			state.Entries[i].Updated = time.Now().UTC()
			changed = true
		case QueuePending, QueueSucceeded, QueueFailed, QueueCancelled, QueueInterrupted:
		default:
			return ErrCorrupt
		}
		switch entry.ErrorCode {
		case "", "process_interrupted", "cancelled", "profile_changed", "job_changed", "job_failed", "admission_failed", "persistence_uncertain":
		default:
			return ErrCorrupt
		}
	}
	q.state = state
	if changed {
		return q.save(state)
	}
	return nil
}

// Bound metadata enumeration even if repeated crashes leave zero-byte orphans.
// Unknown regular files count against both limits and are never auto-deleted.
func (q *Queue) usage() (int64, error) { return q.usageReserved(0) }
func (q *Queue) usageReserved(reserve int) (int64, error) {
	d, e := q.root.Open(".")
	if e != nil {
		return 0, e
	}
	defer d.Close()
	var used int64
	count := 0
	for {
		entries, e := d.ReadDir(32)
		if e != nil && !errors.Is(e, io.EOF) {
			return 0, e
		}
		for _, entry := range entries {
			count++
			if count > 256-reserve {
				return 0, ErrLimit
			}
			st, e := entry.Info()
			if e != nil {
				return 0, e
			}
			if !st.Mode().IsRegular() || st.Size() < 0 || st.Size() > q.cfg.MaxBytes-used {
				return 0, ErrLimit
			}
			used += st.Size()
		}
		if errors.Is(e, io.EOF) {
			return used, nil
		}
	}
}
func (q *Queue) hit(point string) error {
	if q.fault != nil {
		return q.fault(point)
	}
	return nil
}
func (q *Queue) save(state queueState) error {
	b, e := json.Marshal(state)
	if e != nil || len(b) > maxQueueJSON {
		return ErrLimit
	}
	used, e := q.usageReserved(1)
	if e != nil {
		return e
	}
	if int64(len(b)) > q.cfg.MaxBytes-used {
		return ErrLimit
	}
	id, e := token()
	if e != nil {
		return e
	}
	name := ".queue-" + id
	f, e := q.root.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return e
	}
	defer func() { f.Close(); q.root.Remove(name) }()
	if e = writeFull(f, b); e == nil {
		e = f.Sync()
	}
	e = errors.Join(e, f.Close())
	if e != nil {
		return e
	}
	if e = q.hit("queue-synced"); e != nil {
		return e
	}
	if e = q.root.Rename(name, "queue.json"); e != nil {
		return e
	}
	// From rename onward the operation may have published, even if fsync fails.
	if e = q.hit("queue-renamed"); e == nil {
		var d *os.File
		d, e = q.root.Open(".")
		if e == nil {
			e = errors.Join(d.Sync(), d.Close())
		}
	}
	if e != nil {
		q.poison = errors.Join(ErrPersistence, e)
		return q.poison
	}
	q.state = state
	return nil
}
func (q *Queue) ready() error {
	if q.closed || q.stopping {
		return ErrClosed
	}
	if q.poison != nil {
		return q.poison
	}
	return nil
}
func (q *Queue) copyState() queueState {
	s := q.state
	s.Entries = append([]QueueEntry{}, s.Entries...)
	return s
}
func queuePlan(stages []Stage) (string, error) {
	if len(stages) < 1 || len(stages) > 64 {
		return "", ErrConfiguration
	}
	type version struct{ Name, Version string }
	versions := make([]version, 0, len(stages))
	seen := map[string]bool{}
	for _, s := range stages {
		if !validStage(s.Name) || !validHash(s.Version) || s.Run == nil || seen[s.Name] {
			return "", ErrConfiguration
		}
		seen[s.Name] = true
		versions = append(versions, version{s.Name, s.Version})
	}
	b, _ := json.Marshal(versions)
	return hash(b), nil
}

// Enqueue acknowledges only durable intent. A repeated request for an existing
// pending/running job returns that same ticket. Terminal entries require retry
// true explicitly; succeeded jobs cannot be enqueued again. No upload auto-run.
func (q *Queue) Enqueue(ctx context.Context, id string, retry bool) (QueueEntry, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	var zero QueueEntry
	if e := q.ready(); e != nil {
		return zero, e
	}
	if e := ctx.Err(); e != nil {
		return zero, e
	}
	index := -1
	for i, entry := range q.state.Entries {
		if entry.JobID == id {
			if entry.Status == QueuePending || entry.Status == QueueRunning {
				return entry, nil
			}
			index = i
			break
		}
	}
	if index >= 0 && !retry {
		return zero, ErrQueueState
	}
	if index < 0 && len(q.state.Entries) >= q.cfg.MaxEntries {
		return zero, ErrLimit
	}
	m, e := q.store.Get(id)
	if e != nil {
		return zero, e
	}
	if m.Status == Complete || m.Status == Running || m.Status != Queued && !retry {
		return zero, ErrQueueState
	}
	var stages []Stage
	e = callStage(func() error { var err error; stages, err = q.cfg.Resolve(m); return err })
	if e != nil {
		return zero, e
	}
	plan, e := queuePlan(stages)
	if e != nil {
		return zero, e
	}
	if q.state.Next == ^uint64(0) {
		return zero, ErrLimit
	}
	ticket, e := token()
	if e != nil {
		return zero, e
	}
	entry := QueueEntry{JobID: id, Ticket: ticket, Sequence: q.state.Next, Status: QueuePending, InputSHA256: m.Input.SHA256, ConfigSHA256: m.ConfigurationSHA256, PlanSHA256: plan, ExpectedAttempts: m.Attempts, Updated: time.Now().UTC()}
	next := q.copyState()
	next.Next++
	if index < 0 {
		next.Entries = append(next.Entries, entry)
	} else {
		next.Entries[index] = entry
	}
	if e = ctx.Err(); e != nil {
		return zero, e
	}
	if e = q.save(next); e != nil {
		return zero, e
	}
	select {
	case q.wake <- struct{}{}:
	default:
	}
	return entry, nil
}
func (q *Queue) List() ([]QueueEntry, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return nil, ErrClosed
	}
	return append([]QueueEntry{}, q.state.Entries...), q.poison
}

// Cancel durably withdraws pending work; for running work it signals cancellation
// and returns the running snapshot, not a false cancelled-state acknowledgement.
func (q *Queue) Cancel(ctx context.Context, id string) (QueueEntry, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if e := q.ready(); e != nil {
		return QueueEntry{}, e
	}
	if e := ctx.Err(); e != nil {
		return QueueEntry{}, e
	}
	for i, entry := range q.state.Entries {
		if entry.JobID != id {
			continue
		}
		if entry.Status == QueueRunning {
			if q.active == id && q.activeCancel != nil {
				q.activeCancel()
			}
			return entry, nil
		}
		if entry.Status != QueuePending {
			return entry, nil
		}
		next := q.copyState()
		next.Entries[i].Status = QueueCancelled
		next.Entries[i].ErrorCode = "cancelled"
		next.Entries[i].Updated = time.Now().UTC()
		if e := q.save(next); e != nil {
			return QueueEntry{}, e
		}
		if q.active == id && q.activeCancel != nil {
			q.activeCancel()
		}
		return next.Entries[i], nil
	}
	return QueueEntry{}, os.ErrNotExist
}

// Forget removes terminal queue metadata only, never job media. Pending/running
// entries must be cancelled/drained first. Use Delete for media deletion while
// enqueue callers may run concurrently; separate Forget/Store.Delete is racy.
func (q *Queue) Forget(ctx context.Context, id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.forget(ctx, id)
}

// Delete excludes concurrent enqueue while forgetting terminal metadata and
// deleting the media. Caller must also hold its store mutation admission. A
// failed media deletion can leave a job without a ticket; inspect then retry
// deletion explicitly. Neither a failed deletion nor reopening enqueues work.
func (q *Queue) Delete(ctx context.Context, id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if e := q.forget(ctx, id); e != nil {
		return e
	}
	return q.store.Delete(ctx, id)
}
func (q *Queue) forget(ctx context.Context, id string) error {
	if e := q.ready(); e != nil {
		return e
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	for i, entry := range q.state.Entries {
		if entry.JobID != id {
			continue
		}
		if entry.Status == QueuePending || entry.Status == QueueRunning {
			return ErrQueueState
		}
		next := q.copyState()
		next.Entries = append(next.Entries[:i], next.Entries[i+1:]...)
		return q.save(next)
	}
	return nil
}

// Start explicitly authorises one worker to process pending intents, including
// recovered pending work. Running-at-crash records were marked interrupted on
// open and are never automatically retried. Job deadline includes admission wait.
func (q *Queue) Start(ctx context.Context) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if e := q.ready(); e != nil {
		return e
	}
	if q.started {
		return ErrQueueState
	}
	if e := ctx.Err(); e != nil {
		return e
	}
	q.ctx, q.cancel = context.WithCancel(ctx)
	q.started = true
	go q.work()
	return nil
}
func (q *Queue) work() {
	defer close(q.done)
	for {
		q.mu.Lock()
		if q.ctx.Err() != nil || q.poison != nil {
			q.mu.Unlock()
			return
		}
		var entry QueueEntry
		for _, candidate := range q.state.Entries {
			if candidate.Status == QueuePending && (entry.Sequence == 0 || candidate.Sequence < entry.Sequence) {
				entry = candidate
			}
		}
		if entry.Sequence == 0 {
			q.mu.Unlock()
			select {
			case <-q.ctx.Done():
				return
			case <-q.wake:
				continue
			}
		}
		ctx, cancel := context.WithTimeout(q.ctx, q.cfg.JobTimeout)
		q.active = entry.JobID
		q.activeCancel = cancel
		q.mu.Unlock()
		q.execute(ctx, entry)
		cancel()
		q.mu.Lock()
		q.active = ""
		q.activeCancel = nil
		q.mu.Unlock()
	}
}
func (q *Queue) execute(ctx context.Context, entry QueueEntry) {
	// Admission must finish before the durable claim. Shutdown while waiting leaves
	// pending work intact; after claiming, shutdown cancels and requires retry.
	var release func()
	admissionReturned := false
	e := callStage(func() error { var err error; release, err = q.cfg.Admission(ctx); admissionReturned = true; return err })
	if !admissionReturned {
		q.mu.Lock()
		q.poison = fmt.Errorf("admission panicked; resources uncertain")
		q.mu.Unlock()
		return
	}
	if e != nil || release == nil {
		if release != nil {
			if err := callStage(func() error { release(); return nil }); err != nil {
				q.mu.Lock()
				q.poison = fmt.Errorf("admission release failed; resources uncertain")
				q.mu.Unlock()
				return
			}
		}
		q.finish(entry, QueueFailed, "admission_failed", true)
		return
	}
	defer func() {
		if e := callStage(func() error { release(); return nil }); e != nil {
			q.mu.Lock()
			q.poison = fmt.Errorf("admission release failed; resources uncertain")
			q.mu.Unlock()
		}
	}()
	q.mu.Lock()
	index := -1
	for i, x := range q.state.Entries {
		if x.Ticket == entry.Ticket && x.Status == QueuePending {
			index = i
		}
	}
	if index < 0 || q.ctx.Err() != nil || q.poison != nil {
		q.mu.Unlock()
		return
	}
	if ctx.Err() != nil {
		q.mu.Unlock()
		q.finish(entry, QueueFailed, "admission_failed", true)
		return
	}
	next := q.copyState()
	next.Entries[index].Status = QueueRunning
	next.Entries[index].Updated = time.Now().UTC()
	e = q.save(next)
	if e != nil {
		q.poison = errors.Join(ErrPersistence, e)
		q.mu.Unlock()
		return
	}
	q.mu.Unlock()
	if e = q.hit("queue-claimed"); e != nil {
		q.finish(entry, QueueInterrupted, "persistence_uncertain", false)
		return
	}
	e = callStage(func() error {
		m, e := q.store.Get(entry.JobID)
		if e != nil {
			return e
		}
		if m.Input.SHA256 != entry.InputSHA256 || m.ConfigurationSHA256 != entry.ConfigSHA256 || m.Attempts != entry.ExpectedAttempts || m.Status == Complete || m.Status == Running {
			return ErrQueueState
		}
		stages, e := q.cfg.Resolve(m)
		if e != nil {
			return ErrConfiguration
		}
		plan, e := queuePlan(stages)
		if e != nil || plan != entry.PlanSHA256 {
			return ErrConfiguration
		}
		_, e = q.store.Run(ctx, m.ID, []byte(m.Configuration), stages, nil)
		if e == nil {
			e = q.hit("queue-job-committed")
		}
		return e
	})
	status, code := QueueSucceeded, ""
	switch {
	case e == nil:
	case errors.Is(e, ErrPersistence):
		status, code = QueueInterrupted, "persistence_uncertain"
	case errors.Is(e, context.Canceled) || errors.Is(e, context.DeadlineExceeded):
		status, code = QueueCancelled, "cancelled"
	case errors.Is(e, ErrConfiguration):
		status, code = QueueFailed, "profile_changed"
	case errors.Is(e, ErrQueueState):
		status, code = QueueFailed, "job_changed"
	default:
		status, code = QueueFailed, "job_failed"
	}
	q.finish(entry, status, code, false)
	if errors.Is(e, ErrPersistence) {
		q.mu.Lock()
		q.poison = e
		q.mu.Unlock()
	}
}
func (q *Queue) finish(entry QueueEntry, status QueueStatus, code string, wasPending bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.poison != nil || wasPending && q.ctx.Err() != nil {
		return
	}
	for i, x := range q.state.Entries {
		if x.Ticket != entry.Ticket || wasPending && x.Status != QueuePending || !wasPending && x.Status != QueueRunning {
			continue
		}
		next := q.copyState()
		next.Entries[i].Status = status
		next.Entries[i].ErrorCode = code
		next.Entries[i].Updated = time.Now().UTC()
		if e := q.save(next); e != nil {
			q.poison = errors.Join(ErrPersistence, e)
		}
		return
	}
}

// Shutdown cancels a worker and waits for its admission/run/release to drain.
// It stops enqueue mutations but retains the journal lock until Close. Timeout
// does not permit store/model teardown; retry Shutdown before Close.
func (q *Queue) Shutdown(ctx context.Context) error {
	q.mu.Lock()
	q.stopping = true
	if q.cancel != nil {
		q.cancel()
	}
	started, done := q.started, q.done
	q.mu.Unlock()
	if !started {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (q *Queue) Close() error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return nil
	}
	if q.started {
		select {
		case <-q.done:
		default:
			return ErrBusy
		}
	}
	q.stopping = true
	q.closed = true
	return errors.Join(q.lock.Close(), q.root.Close())
}
