package httpapi

import (
	"context"
	"errors"
	"io"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/runtime/resourcebudget"
	"github.com/rcarmo/go-pherence/runtime/speechjob"
)

func budgetHTTP(t *testing.T, admission speechjob.Admission, queued bool, stage speechjob.Stage) *Handler {
	t.Helper()
	root := t.TempDir()
	s, e := speechjob.Open(filepath.Join(root, "store"), speechjob.Limits{MaxJobs: 8, MaxUploadBytes: 1024, MaxArtifactBytes: 1 << 20, MaxBytes: 8 << 20})
	if e != nil {
		t.Fatal(e)
	}
	cfg := Config{Store: s, Profiles: []Profile{{ID: "test", Configuration: []byte(`{}`), Stages: []speechjob.Stage{stage}}}, Token: testToken, Hosts: []string{"speech.test"}, MaxUploadBytes: 1024, MaxConcurrentRequests: 2, RunAdmission: admission}
	if queued {
		cfg.RunAdmission = nil
		cfg.Queue = &QueueOptions{Directory: filepath.Join(root, "queue"), MaxEntries: 8, MaxBytes: 1 << 20, JobTimeout: time.Second, Admission: admission}
	}
	h, e := New(cfg)
	if e != nil {
		s.Close()
		t.Fatal(e)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := h.Shutdown(ctx); e != nil {
			t.Error(e)
		}
		s.Close()
	})
	return h
}
func waitBudget(t *testing.T, b *resourcebudget.Budget, predicate func(resourcebudget.Snapshot) bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if predicate(b.Snapshot()) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("budget state", b.Snapshot())
}
func newHTTPBudget(t *testing.T) *resourcebudget.Budget {
	t.Helper()
	b, e := resourcebudget.New(resourcebudget.Config{Capacity: resourcebudget.Resources{CPUSlots: 4, MemoryBytes: 100}, MaxActive: 4, MaxWaiting: 8})
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := b.Shutdown(ctx); e != nil {
			t.Error(e, b.Snapshot())
		}
	})
	return b
}
func TestHTTPSharedWeightedAdmissionAndCancelDrain(t *testing.T) {
	b := newHTTPBudget(t)
	// Simulated cooperating LLM/resource owner, no real model/service. Memory
	// prevents both speech owners from entering even when CPU slots are available.
	llm, e := b.Acquire(context.Background(), resourcebudget.Resources{CPUSlots: 1, MemoryBytes: 80})
	if e != nil {
		t.Fatal(e)
	}
	defer llm.Release()
	admit, e := b.Admission(resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 30})
	if e != nil {
		t.Fatal(e)
	}
	entered, drain := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-drain:
		default:
			close(drain)
		}
	}()
	synchronous := budgetHTTP(t, admit, false, testStage("transcript", func(ctx context.Context, _ *speechjob.Input, _ io.Writer) error {
		close(entered)
		<-ctx.Done()
		<-drain
		return ctx.Err()
	}))
	var queuedCalls atomic.Int32
	queued := budgetHTTP(t, admit, true, testStage("transcript", func(context.Context, *speechjob.Input, io.Writer) error { queuedCalls.Add(1); return nil }))
	a, c := upload(t, synchronous), upload(t, queued)
	response := make(chan *httptest.ResponseRecorder, 1)
	go func() { response <- request(synchronous, "POST", "/v1/jobs/"+a.ID+"/run", nil) }()
	waitBudget(t, b, func(s resourcebudget.Snapshot) bool { return s.Waiting == 1 })
	if w := request(queued, "POST", "/v1/jobs/"+c.ID+"/enqueue", nil); w.Code != 202 {
		t.Fatal(w)
	}
	queued.StartQueue(context.Background())
	waitBudget(t, b, func(s resourcebudget.Snapshot) bool { return s.Waiting == 2 })
	if b.Snapshot().Used != (resourcebudget.Resources{CPUSlots: 1, MemoryBytes: 80}) {
		t.Fatal("partial reservation", b.Snapshot())
	}
	select {
	case <-entered:
		t.Fatal("ran before budget")
	default:
	}
	llm.Release()
	<-entered
	// Two admitted speech jobs can fit 4CPU/60bytes, but sync cancel must retain
	// its own 2CPU/30byte reservation until its callback actually returns.
	queuedState(t, queued, c.ID, speechjob.QueueSucceeded)
	waitBudget(t, b, func(s resourcebudget.Snapshot) bool { return s.Active == 1 && s.Used.CPUSlots == 2 })
	if w := request(synchronous, "POST", "/v1/jobs/"+a.ID+"/cancel", nil); w.Code != 202 {
		t.Fatal(w)
	}
	if b.Snapshot().Used != (resourcebudget.Resources{CPUSlots: 2, MemoryBytes: 30}) {
		t.Fatal("cancel released running reservation")
	}
	close(drain)
	if w := <-response; w.Code != 409 {
		t.Fatal(w)
	}
	waitBudget(t, b, func(s resourcebudget.Snapshot) bool { return s.Active == 0 && s.Waiting == 0 })
	if queuedCalls.Load() != 1 {
		t.Fatal(queuedCalls.Load())
	}
}
func TestHTTPSynchronousAdmissionWaitingCancel(t *testing.T) {
	b := newHTTPBudget(t)
	hold, e := b.Acquire(context.Background(), resourcebudget.Resources{CPUSlots: 4, MemoryBytes: 100})
	if e != nil {
		t.Fatal(e)
	}
	defer hold.Release()
	admit, _ := b.Admission(resourcebudget.Resources{CPUSlots: 1, MemoryBytes: 1})
	var calls atomic.Int32
	h := budgetHTTP(t, admit, false, testStage("transcript", func(context.Context, *speechjob.Input, io.Writer) error { calls.Add(1); return nil }))
	j := upload(t, h)
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	waitBudget(t, b, func(s resourcebudget.Snapshot) bool { return s.Waiting == 1 })
	if w := request(h, "POST", "/v1/jobs/"+j.ID+"/cancel", nil); w.Code != 202 {
		t.Fatal(w)
	}
	if w := <-done; w.Code != 409 {
		t.Fatal(w)
	}
	if calls.Load() != 0 || b.Snapshot().Waiting != 0 || b.Snapshot().Active != 1 {
		t.Fatal(calls.Load(), b.Snapshot())
	}
	stored, e := h.store.Get(j.ID)
	if e != nil || stored.Attempts != 0 || stored.Status != speechjob.Queued {
		t.Fatal(stored, e)
	}
}
func TestHTTPSynchronousAdmissionErrorsAndPanic(t *testing.T) {
	for _, mode := range []string{"error", "nil", "panic", "release-panic", "error-release-panic"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int32
			admission := func(context.Context) (func(), error) {
				switch mode {
				case "error":
					return nil, errors.New("private")
				case "nil":
					return nil, nil
				case "panic":
					panic("private")
				case "release-panic":
					return func() { panic("private") }, nil
				default:
					return func() { panic("private") }, errors.New("private")
				}
			}
			h := budgetHTTP(t, admission, false, testStage("transcript", func(context.Context, *speechjob.Input, io.Writer) error { calls.Add(1); return nil }))
			j := upload(t, h)
			w := request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil)
			expected := 503
			if mode == "release-panic" {
				expected = 200
			}
			if w.Code != expected {
				t.Fatal(w)
			}
			if mode == "release-panic" {
				if calls.Load() != 1 {
					t.Fatal(calls.Load())
				}
			} else if calls.Load() != 0 {
				t.Fatal("ran without admission")
			}
			if mode == "panic" || mode == "release-panic" || mode == "error-release-panic" {
				if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 503 {
					t.Fatal("uncertain mutation allowed", w)
				}
			}
		})
	}
}
func TestHTTPSynchronousReleaseDrainBeforeMutation(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	h := budgetHTTP(t, func(context.Context) (func(), error) { return func() { close(entered); <-release }, nil }, false, textStage("transcript", "ok"))
	j := upload(t, h)
	done := make(chan struct{})
	go func() { defer close(done); request(h, "POST", "/v1/jobs/"+j.ID+"/run", nil) }()
	<-entered
	if w := request(h, "DELETE", "/v1/jobs/"+j.ID, nil); w.Code != 409 {
		t.Fatal(w)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	if e := h.Shutdown(ctx); !errors.Is(e, context.DeadlineExceeded) {
		t.Fatal(e)
	}
	close(release)
	<-done
	if e := h.Shutdown(context.Background()); e != nil {
		t.Fatal(e)
	}
}
