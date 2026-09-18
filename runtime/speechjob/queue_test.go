package speechjob

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func openQueueTest(t *testing.T, s *Store, dir string, admission Admission, resolve func(Manifest) ([]Stage, error)) *Queue {
	t.Helper()
	q, e := OpenQueue(s, QueueConfig{Directory: dir, MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: 5 * time.Second, Resolve: resolve, Admission: admission})
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := q.Shutdown(ctx); e != nil {
			t.Error(e)
		}
		if e := q.Close(); e != nil {
			t.Error(e)
		}
	})
	return q
}
func queueWait(t *testing.T, q *Queue, id string, status QueueStatus) QueueEntry {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		entries, e := q.List()
		if e != nil {
			t.Fatal(e)
		}
		for _, x := range entries {
			if x.JobID == id && x.Status == status {
				return x
			}
		}
		time.Sleep(time.Millisecond)
	}
	entries, e := q.List()
	t.Fatal("queue did not reach status", id, status, entries, e)
	return QueueEntry{}
}
func TestQueueExplicitStartFIFOAndDuplicate(t *testing.T) {
	s, _ := openTest(t)
	var mu sync.Mutex
	var order []string
	resolve := func(m Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(_ context.Context, in *Input, w io.Writer) error {
			mu.Lock()
			order = append(order, in.job.ID)
			mu.Unlock()
			_, e := io.WriteString(w, "result")
			return e
		})}, nil
	}
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), resolve)
	a, b := createTest(t, s), createTest(t, s)
	first, e := q.Enqueue(context.Background(), a.ID, false)
	if e != nil {
		t.Fatal(e)
	}
	duplicate, e := q.Enqueue(context.Background(), a.ID, false)
	if e != nil || duplicate.Ticket != first.Ticket || duplicate.Sequence != first.Sequence {
		t.Fatal(first, duplicate, e)
	}
	second, e := q.Enqueue(context.Background(), b.ID, false)
	if e != nil || second.Sequence <= first.Sequence {
		t.Fatal(e)
	}
	mu.Lock()
	if len(order) != 0 {
		t.Fatal("implicit worker")
	}
	mu.Unlock()
	if e = q.Start(context.Background()); e != nil {
		t.Fatal(e)
	}
	queueWait(t, q, b.ID, QueueSucceeded)
	mu.Lock()
	if len(order) != 2 || order[0] != a.ID || order[1] != b.ID {
		t.Fatal(order)
	}
	mu.Unlock()
	if _, e = q.Enqueue(context.Background(), a.ID, true); !errors.Is(e, ErrQueueState) {
		t.Fatal("complete retried", e)
	}
	if e = q.Start(context.Background()); !errors.Is(e, ErrQueueState) {
		t.Fatal("second worker", e)
	}
	if e = q.Forget(context.Background(), a.ID); e != nil {
		t.Fatal(e)
	}
}
func TestQueuePendingRecoveryAndRetryFailure(t *testing.T) {
	s, _ := openTest(t)
	dir := filepath.Join(t.TempDir(), "queue")
	var fails atomic.Bool
	fails.Store(true)
	var calls atomic.Int32
	resolve := func(Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(context.Context, *Input, io.Writer) error {
			calls.Add(1)
			if fails.Load() {
				return io.ErrClosedPipe
			}
			return nil
		})}, nil
	}
	q := openQueueTest(t, s, dir, SerialAdmission(), resolve)
	job := createTest(t, s)
	ticket, e := q.Enqueue(context.Background(), job.ID, false)
	if e != nil {
		t.Fatal(e)
	}
	q.Close()
	q = openQueueTest(t, s, dir, SerialAdmission(), resolve)
	entries, e := q.List()
	if e != nil || entries[0].Ticket != ticket.Ticket || entries[0].Status != QueuePending || calls.Load() != 0 {
		t.Fatal(entries, e)
	}
	q.Start(context.Background())
	queueWait(t, q, job.ID, QueueFailed)
	if _, e = q.Enqueue(context.Background(), job.ID, false); !errors.Is(e, ErrQueueState) {
		t.Fatal(e)
	}
	fails.Store(false)
	newTicket, e := q.Enqueue(context.Background(), job.ID, true)
	if e != nil || newTicket.Ticket == ticket.Ticket {
		t.Fatal(e)
	}
	queueWait(t, q, job.ID, QueueSucceeded)
	if calls.Load() != 2 {
		t.Fatal(calls.Load())
	}
}
func TestQueueCancelPendingAdmissionAndOwnedDrain(t *testing.T) {
	s, _ := openTest(t)
	admitted := make(chan struct{})
	releaseAdmission := make(chan struct{})
	var calls atomic.Int32
	admission := func(ctx context.Context) (func(), error) {
		close(admitted)
		select {
		case <-releaseAdmission:
			return func() {}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), admission, func(Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(context.Context, *Input, io.Writer) error { calls.Add(1); return nil })}, nil
	})
	job := createTest(t, s)
	q.Enqueue(context.Background(), job.ID, false)
	q.Start(context.Background())
	<-admitted
	x, e := q.Cancel(context.Background(), job.ID)
	if e != nil || x.Status != QueueCancelled {
		t.Fatal(x, e)
	}
	close(releaseAdmission)
	q.Shutdown(context.Background())
	if calls.Load() != 0 {
		t.Fatal("withdrawn work ran")
	}
	// Cancellation after claim waits until callback drains before release/close.
	started := make(chan struct{})
	drain := make(chan struct{})
	released := make(chan struct{})
	q2 := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), func(context.Context) (func(), error) { return func() { close(released) }, nil }, func(Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(ctx context.Context, _ *Input, _ io.Writer) error {
			close(started)
			<-ctx.Done()
			<-drain
			return ctx.Err()
		})}, nil
	})
	q2.Enqueue(context.Background(), job.ID, false)
	q2.Start(context.Background())
	<-started
	q2.Cancel(context.Background(), job.ID)
	if e = q2.Close(); !errors.Is(e, ErrBusy) {
		t.Fatal(e)
	}
	select {
	case <-released:
		t.Fatal("release before callback drain")
	default:
	}
	close(drain)
	queueWait(t, q2, job.ID, QueueCancelled)
	q2.Shutdown(context.Background())
	<-released
}
func TestQueueProfileChangeAndQuotaPoison(t *testing.T) {
	s, _ := openTest(t)
	job := createTest(t, s)
	version := "old"
	calls := 0
	resolve := func(Manifest) ([]Stage, error) {
		st := stage("transcript", func(context.Context, *Input, io.Writer) error { calls++; return nil })
		st.Version = hash([]byte(version))
		return []Stage{st}, nil
	}
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), resolve)
	q.Enqueue(context.Background(), job.ID, false)
	version = "new"
	q.Start(context.Background())
	entry := queueWait(t, q, job.ID, QueueFailed)
	if entry.ErrorCode != "profile_changed" || calls != 0 {
		t.Fatal(entry, calls)
	}
	q.Shutdown(context.Background())
	q.Close()
	q2 := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), resolve)
	q2.fault = func(point string) error {
		if point == "queue-renamed" {
			return io.ErrClosedPipe
		}
		return nil
	}
	if _, e := q2.Enqueue(context.Background(), job.ID, false); !errors.Is(e, ErrPersistence) {
		t.Fatal(e)
	}
	if e := q2.Start(context.Background()); !errors.Is(e, ErrPersistence) {
		t.Fatal("uncertain enqueue ran", e)
	}
	q2.fault = nil
	q2.Close()
	q3 := openQueueTest(t, s, q2.root.Name(), SerialAdmission(), resolve)
	entries, e := q3.List()
	if e != nil || len(entries) != 1 {
		t.Fatal(entries, e)
	}
	q3.cfg.MaxEntries = 1
	other := createTest(t, s)
	if _, e = q3.Enqueue(context.Background(), other.ID, false); !errors.Is(e, ErrLimit) {
		t.Fatal(e)
	}
	if e = os.WriteFile(filepath.Join(q3.root.Name(), "orphan"), make([]byte, q3.cfg.MaxBytes), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e = q3.Cancel(context.Background(), job.ID); e == nil {
		t.Fatal("unaccounted orphan")
	}
}
func TestQueueShutdownKeepsWaitingPendingAndSharedAdmission(t *testing.T) {
	shared := SerialAdmission()
	unlock, e := shared(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	s, _ := openTest(t)
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), shared, func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	job := createTest(t, s)
	q.Enqueue(context.Background(), job.ID, false)
	q.Start(context.Background())
	q.Shutdown(context.Background())
	entries, e := q.List()
	if e != nil || entries[0].Status != QueuePending {
		t.Fatal(entries, e)
	}
	unlock()
	unlock()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e = shared(ctx); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	release, e := shared(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	release()
}
func TestQueueLocationCorruptionAndResolverPanic(t *testing.T) {
	s, _ := openTest(t)
	cfg := QueueConfig{Directory: filepath.Join(s.root.Name(), "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: time.Second, Admission: SerialAdmission(), Resolve: func(Manifest) ([]Stage, error) { panic("resolver") }}
	if _, e := OpenQueue(s, cfg); e == nil {
		t.Fatal("queue within store")
	}
	cfg.Directory = filepath.Join(t.TempDir(), "queue")
	q, e := OpenQueue(s, cfg)
	if e != nil {
		t.Fatal(e)
	}
	job := createTest(t, s)
	if _, e = q.Enqueue(context.Background(), job.ID, false); e == nil {
		t.Fatal("panic escaped")
	}
	q.Close()
	if e = os.WriteFile(filepath.Join(cfg.Directory, "queue.json"), []byte(`{"schema":1,"schema":1}`), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e = OpenQueue(s, cfg); !errors.Is(e, ErrCorrupt) {
		t.Fatal(e)
	}
}

func TestQueueDeleteExcludesEnqueue(t *testing.T) {
	s, _ := openTest(t)
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	job := createTest(t, s)
	q.Enqueue(context.Background(), job.ID, false)
	if e := q.Delete(context.Background(), job.ID); !errors.Is(e, ErrQueueState) {
		t.Fatal(e)
	}
	q.Cancel(context.Background(), job.ID)
	entered, release := make(chan struct{}), make(chan struct{})
	s.fault = func(p string) error {
		if p == "delete-renamed" {
			close(entered)
			<-release
		}
		return nil
	}
	deleted := make(chan error, 1)
	go func() { deleted <- q.Delete(context.Background(), job.ID) }()
	<-entered
	enqueued := make(chan error, 1)
	go func() { _, e := q.Enqueue(context.Background(), job.ID, true); enqueued <- e }()
	select {
	case e := <-enqueued:
		t.Fatal("enqueue escaped deletion lock", e)
	default:
	}
	close(release)
	if e := <-deleted; e != nil {
		t.Fatal(e)
	}
	if e := <-enqueued; !errors.Is(e, os.ErrNotExist) {
		t.Fatal(e)
	}
	entries, e := q.List()
	if e != nil || len(entries) != 0 {
		t.Fatal(entries, e)
	}
}
func TestQueueConcurrentDuplicateAndCancel(t *testing.T) {
	s, _ := openTest(t)
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	job := createTest(t, s)
	var wg sync.WaitGroup
	results := make(chan QueueEntry, 32)
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x, e := q.Enqueue(context.Background(), job.ID, false)
			if e != nil {
				t.Error(e)
			}
			results <- x
		}()
	}
	wg.Wait()
	close(results)
	ticket := ""
	for x := range results {
		if ticket == "" {
			ticket = x.Ticket
		}
		if x.Ticket != ticket {
			t.Fatal("multiple tickets")
		}
	}
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x, e := q.Cancel(context.Background(), job.ID)
			if e != nil || x.Status != QueueCancelled {
				t.Error(x, e)
			}
		}()
	}
	wg.Wait()
}
func TestQueueAdmissionFailuresStopUncertainResources(t *testing.T) {
	for _, mode := range []string{"error", "nil", "panic", "release-panic", "error-release-panic"} {
		t.Run(mode, func(t *testing.T) {
			s, _ := openTest(t)
			var calls atomic.Int32
			var admissions atomic.Int32
			q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), func(context.Context) (func(), error) {
				admissions.Add(1)
				switch mode {
				case "error":
					return nil, io.ErrClosedPipe
				case "nil":
					return nil, nil
				case "panic":
					panic("test")
				case "release-panic":
					return func() { panic("test") }, nil
				default:
					return func() { panic("test") }, io.ErrClosedPipe
				}
			}, func(Manifest) ([]Stage, error) {
				return []Stage{stage("transcript", func(context.Context, *Input, io.Writer) error { calls.Add(1); return nil })}, nil
			})
			a, b := createTest(t, s), createTest(t, s)
			q.Enqueue(context.Background(), a.ID, false)
			q.Enqueue(context.Background(), b.ID, false)
			q.Start(context.Background())
			if mode == "error" || mode == "nil" {
				queueWait(t, q, b.ID, QueueFailed)
			} else {
				select {
				case <-q.done:
				case <-time.After(time.Second):
					t.Fatal("uncertain resource worker continued")
				}
			}
			q.Shutdown(context.Background())
			want := int32(0)
			if mode == "release-panic" {
				want = 1
			}
			if calls.Load() != want {
				t.Fatal(calls.Load())
			}
			if mode != "error" && mode != "nil" && admissions.Load() != 1 {
				t.Fatal(admissions.Load())
			}
		})
	}
}
func TestQueueClaimPersistenceFailureNeverExecutes(t *testing.T) {
	for _, point := range []string{"queue-synced", "queue-renamed"} {
		t.Run(point, func(t *testing.T) {
			s, _ := openTest(t)
			var calls atomic.Int32
			q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), func(Manifest) ([]Stage, error) {
				return []Stage{stage("transcript", func(context.Context, *Input, io.Writer) error { calls.Add(1); return nil })}, nil
			})
			job := createTest(t, s)
			q.Enqueue(context.Background(), job.ID, false)
			q.fault = func(p string) error {
				if p == point {
					return io.ErrClosedPipe
				}
				return nil
			}
			q.Start(context.Background())
			select {
			case <-q.done:
			case <-time.After(time.Second):
				t.Fatal("worker did not stop")
			}
			if calls.Load() != 0 {
				t.Fatal(calls.Load())
			}
			if _, e := q.List(); !errors.Is(e, ErrPersistence) {
				t.Fatal(e)
			}
		})
	}
}
func TestQueueDirectoryEntryBound(t *testing.T) {
	s, _ := openTest(t)
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	for i := 0; i < 257; i++ {
		if e := os.WriteFile(filepath.Join(q.root.Name(), fmt.Sprintf("orphan-%d", i)), nil, 0600); e != nil {
			t.Fatal(e)
		}
	}
	if _, e := q.usage(); !errors.Is(e, ErrLimit) {
		t.Fatal(e)
	}
}
func TestQueueSnapshotAllocationBound(t *testing.T) {
	s, _ := openTest(t)
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	q.state.Entries = make([]QueueEntry, 128)
	allocs := testing.AllocsPerRun(100, func() {
		entries, e := q.List()
		if e != nil || len(entries) != 128 {
			panic("snapshot")
		}
		entries[0].JobID = "detached"
	})
	if q.state.Entries[0].JobID != "" || allocs > 1 {
		t.Fatal("snapshot ownership/allocation regression", allocs)
	}
	t.Logf("128-entry detached snapshot: %.0f allocation(s); no timing assertion", allocs)
}

func TestQueueReservesTemporaryEntry(t *testing.T) {
	s, _ := openTest(t)
	q := openQueueTest(t, s, filepath.Join(t.TempDir(), "queue"), SerialAdmission(), func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	for i := 0; i < 254; i++ {
		if e := os.WriteFile(filepath.Join(q.root.Name(), fmt.Sprintf("orphan-%d", i)), nil, 0600); e != nil {
			t.Fatal(e)
		}
	}
	job := createTest(t, s)
	if _, e := q.Enqueue(context.Background(), job.ID, false); !errors.Is(e, ErrLimit) {
		t.Fatal(e)
	}
	entries, e := os.ReadDir(q.root.Name())
	if e != nil || len(entries) != 256 {
		t.Fatal(len(entries), e)
	}
}

func TestQueueRecoveredClaimDoesNotAutomaticallyReplay(t *testing.T) {
	s, _ := openTest(t)
	var calls atomic.Int32
	resolve := func(Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(context.Context, *Input, io.Writer) error { calls.Add(1); return nil })}, nil
	}
	dir := filepath.Join(t.TempDir(), "queue")
	q := openQueueTest(t, s, dir, SerialAdmission(), resolve)
	a, b := createTest(t, s), createTest(t, s)
	q.Enqueue(context.Background(), a.ID, false)
	q.Enqueue(context.Background(), b.ID, false)
	state := q.copyState()
	state.Entries[0].Status = QueueRunning
	if e := q.save(state); e != nil {
		t.Fatal(e)
	}
	q.Close()
	q = openQueueTest(t, s, dir, SerialAdmission(), resolve)
	q.Start(context.Background())
	queueWait(t, q, b.ID, QueueSucceeded)
	queueWait(t, q, a.ID, QueueInterrupted)
	q.Shutdown(context.Background())
	if calls.Load() != 1 {
		t.Fatal("interrupted claim replayed", calls.Load())
	}
}
func TestQueueSharedAdmissionAcrossStores(t *testing.T) {
	s1, _ := openTest(t)
	s2, _ := openTest(t)
	shared := SerialAdmission()
	entered, release := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	q1 := openQueueTest(t, s1, filepath.Join(t.TempDir(), "queue"), shared, func(Manifest) ([]Stage, error) {
		return []Stage{stage("transcript", func(context.Context, *Input, io.Writer) error { close(entered); <-release; return nil })}, nil
	})
	q2 := openQueueTest(t, s2, filepath.Join(t.TempDir(), "queue"), shared, func(Manifest) ([]Stage, error) { return []Stage{textStage("transcript", "ok")}, nil })
	a, b := createTest(t, s1), createTest(t, s2)
	q1.Enqueue(context.Background(), a.ID, false)
	q1.Start(context.Background())
	<-entered
	q2.Enqueue(context.Background(), b.ID, false)
	q2.Start(context.Background())
	queueWait(t, q2, b.ID, QueuePending)
	close(release)
	queueWait(t, q1, a.ID, QueueSucceeded)
	queueWait(t, q2, b.ID, QueueSucceeded)
}
