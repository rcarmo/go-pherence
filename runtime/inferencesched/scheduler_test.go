package inferencesched

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

type testRequest struct {
	id  string
	seq uint64
}

func (r testRequest) ID() string         { return r.id }
func (r testRequest) ArrivalSeq() uint64 { return r.seq }

type eventLog struct {
	mu     sync.Mutex
	events []string
}

func (l *eventLog) add(event string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, event)
}

func (l *eventLog) snapshot() []string {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

type testWork struct {
	mu               sync.Mutex
	id               string
	prefillRemaining int
	decodeRemaining  int
	prefillQuantum   int
	decodeQuantum    int
	prefillErr       error
	decodeErr        error
	closeErr         error
	log              *eventLog
	prefillCalls     int
	decodeCalls      int
	cancelCalls      int
	closeCalls       int
	canceled         bool
	finished         bool
}

type testWorkState struct {
	prefillRemaining int
	decodeRemaining  int
	prefillCalls     int
	decodeCalls      int
	cancelCalls      int
	closeCalls       int
	canceled         bool
	finished         bool
}

func newTestWork(id string, prefillRemaining, decodeRemaining int) *testWork {
	w := &testWork{id: id, prefillRemaining: prefillRemaining, decodeRemaining: decodeRemaining}
	w.finished = prefillRemaining == 0 && decodeRemaining == 0
	return w
}

func (w *testWork) RemainingPrefill() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.prefillRemaining
}

func (w *testWork) Prefill(maxTokens int) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.prefillCalls++
	if w.log != nil {
		w.log.add("prefill:" + w.id)
	}
	if w.prefillErr != nil {
		return 0, w.prefillErr
	}
	used := minInt(maxTokens, w.prefillRemaining)
	if w.prefillQuantum > 0 && used > w.prefillQuantum {
		used = w.prefillQuantum
	}
	w.prefillRemaining -= used
	if w.prefillRemaining == 0 && w.decodeRemaining == 0 {
		w.finished = true
	}
	return used, nil
}

func (w *testWork) Decode(maxTokens int) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.decodeCalls++
	if w.log != nil {
		w.log.add("decode:" + w.id)
	}
	if w.decodeErr != nil {
		return 0, w.decodeErr
	}
	used := minInt(maxTokens, w.decodeRemaining)
	if w.decodeQuantum > 0 && used > w.decodeQuantum {
		used = w.decodeQuantum
	}
	w.decodeRemaining -= used
	if w.prefillRemaining == 0 && w.decodeRemaining == 0 {
		w.finished = true
	}
	return used, nil
}

func (w *testWork) Finished() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.finished
}

func (w *testWork) Cancel() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.cancelCalls++
	w.canceled = true
}

func (w *testWork) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closeCalls++
	return w.closeErr
}

func (w *testWork) snapshot() testWorkState {
	w.mu.Lock()
	defer w.mu.Unlock()
	return testWorkState{
		prefillRemaining: w.prefillRemaining,
		decodeRemaining:  w.decodeRemaining,
		prefillCalls:     w.prefillCalls,
		decodeCalls:      w.decodeCalls,
		cancelCalls:      w.cancelCalls,
		closeCalls:       w.closeCalls,
		canceled:         w.canceled,
		finished:         w.finished,
	}
}

func TestFCFSAdmissionByArrivalSequence(t *testing.T) {
	s, err := New(Config{MaxActive: 2, DecodeBudget: 0, PrefillBudget: 0})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	w2 := newTestWork("r2", 1, 0)
	w1 := newTestWork("r1", 1, 0)
	w3 := newTestWork("r3", 1, 0)
	if err := s.Add(context.Background(), testRequest{id: "r2", seq: 2}, w2); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), testRequest{id: "r1", seq: 1}, w1); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), testRequest{id: "r3", seq: 3}, w3); err != nil {
		t.Fatal(err)
	}

	report, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report.Stats.Admitted != 2 || report.Stats.Running != 2 || report.Stats.Waiting != 1 {
		t.Fatalf("unexpected stats: %+v", report.Stats)
	}
	got := resultIDs(report.Results)
	want := []string{"r1", "r2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("admission order=%v want %v", got, want)
	}
}

func TestDecodeFirst(t *testing.T) {
	log := &eventLog{}
	s, err := New(Config{MaxActive: 2, DecodeBudget: 1, PrefillBudget: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	wa := newTestWork("a", 0, 1)
	wa.log = log
	wb := newTestWork("b", 1, 0)
	wb.log = log
	if err := s.Add(context.Background(), testRequest{id: "a", seq: 1}, wa); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), testRequest{id: "b", seq: 2}, wb); err != nil {
		t.Fatal(err)
	}

	report, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report.Stats.DecodeUsed != 1 || report.Stats.PrefillUsed != 1 {
		t.Fatalf("unexpected usage: %+v", report.Stats)
	}
	if got, want := log.snapshot(), []string{"decode:a", "prefill:b"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("events=%v want %v", got, want)
	}
}

func TestBudgetsAndTailsAcrossSteps(t *testing.T) {
	s, err := New(Config{MaxActive: 2, DecodeBudget: 2, PrefillBudget: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	wd := newTestWork("decode", 0, 3)
	wp := newTestWork("prefill", 5, 0)
	if err := s.Add(context.Background(), testRequest{id: "decode", seq: 1}, wd); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), testRequest{id: "prefill", seq: 2}, wp); err != nil {
		t.Fatal(err)
	}

	report1, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report1.Stats.DecodeUsed != 2 || report1.Stats.PrefillUsed != 3 {
		t.Fatalf("step1 usage=%+v", report1.Stats)
	}
	if got := wd.snapshot().decodeRemaining; got != 1 {
		t.Fatalf("decode tail=%d want 1", got)
	}
	if got := wp.snapshot().prefillRemaining; got != 2 {
		t.Fatalf("prefill tail=%d want 2", got)
	}

	report2, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report2.Stats.DecodeUsed != 1 || report2.Stats.PrefillUsed != 2 {
		t.Fatalf("step2 usage=%+v", report2.Stats)
	}
	if !findResult(report2.Results, "decode").Finished {
		t.Fatalf("decode request not finished: %+v", findResult(report2.Results, "decode"))
	}
	if !findResult(report2.Results, "prefill").Finished {
		t.Fatalf("prefill request not finished: %+v", findResult(report2.Results, "prefill"))
	}
}

func TestCancellationWaitingAndRunning(t *testing.T) {
	s, err := New(Config{MaxActive: 1, DecodeBudget: 0, PrefillBudget: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	runCtx, runCancel := context.WithCancel(context.Background())
	waitCtx, waitCancel := context.WithCancel(context.Background())
	wr := newTestWork("running", 4, 0)
	ww := newTestWork("waiting", 1, 0)
	if err := s.Add(runCtx, testRequest{id: "running", seq: 1}, wr); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(waitCtx, testRequest{id: "waiting", seq: 2}, ww); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Step(context.Background()); err != nil {
		t.Fatal(err)
	}

	waitCancel()
	report, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	res := findResult(report.Results, "waiting")
	if !res.Canceled || res.Err != nil {
		t.Fatalf("waiting cancel result=%+v", res)
	}
	if got := ww.snapshot(); got.cancelCalls != 1 || got.closeCalls != 1 {
		t.Fatalf("waiting cancel/close=%+v", got)
	}

	runCancel()
	report, err = s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	res = findResult(report.Results, "running")
	if !res.Canceled || res.Err != nil {
		t.Fatalf("running cancel result=%+v", res)
	}
	if got := wr.snapshot(); got.cancelCalls != 1 || got.closeCalls != 1 {
		t.Fatalf("running cancel/close=%+v", got)
	}
}

func TestErrorsAndValidation(t *testing.T) {
	s, err := New(Config{MaxActive: 2, DecodeBudget: 2, PrefillBudget: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	prefillErr := errors.New("prefill boom")
	decodeErr := errors.New("decode boom")
	wp := newTestWork("prefill", 1, 0)
	wp.prefillErr = prefillErr
	wd := newTestWork("decode", 0, 1)
	wd.decodeErr = decodeErr
	if err := s.Add(context.Background(), testRequest{id: "prefill", seq: 1}, wp); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), testRequest{id: "decode", seq: 2}, wd); err != nil {
		t.Fatal(err)
	}

	report, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report.Stats.Errors != 2 || report.Stats.Running != 0 {
		t.Fatalf("unexpected stats: %+v", report.Stats)
	}
	if !errors.Is(findResult(report.Results, "prefill").Err, prefillErr) {
		t.Fatalf("prefill error=%v", findResult(report.Results, "prefill").Err)
	}
	if !errors.Is(findResult(report.Results, "decode").Err, decodeErr) {
		t.Fatalf("decode error=%v", findResult(report.Results, "decode").Err)
	}
	if got := wp.snapshot(); got.closeCalls != 1 || got.cancelCalls != 0 {
		t.Fatalf("prefill close/cancel=%+v", got)
	}
	if got := wd.snapshot(); got.closeCalls != 1 || got.cancelCalls != 0 {
		t.Fatalf("decode close/cancel=%+v", got)
	}
}

func TestMaxActiveAdmission(t *testing.T) {
	s, err := New(Config{MaxActive: 2, DecodeBudget: 1, PrefillBudget: 0})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	w1 := newTestWork("r1", 0, 1)
	w2 := newTestWork("r2", 0, 1)
	w3 := newTestWork("r3", 0, 1)
	for i, w := range []*testWork{w1, w2, w3} {
		if err := s.Add(context.Background(), testRequest{id: fmt.Sprintf("r%d", i+1), seq: uint64(i + 1)}, w); err != nil {
			t.Fatal(err)
		}
	}

	report1, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report1.Stats.Admitted != 2 || report1.Stats.Running != 1 || report1.Stats.Waiting != 1 {
		t.Fatalf("step1 stats=%+v", report1.Stats)
	}
	if !findResult(report1.Results, "r1").Finished {
		t.Fatalf("r1 should finish in step1: %+v", findResult(report1.Results, "r1"))
	}

	report2, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report2.Stats.Admitted != 1 || report2.Stats.Waiting != 0 {
		t.Fatalf("step2 stats=%+v", report2.Stats)
	}
	if !findResult(report2.Results, "r2").Finished {
		t.Fatalf("r2 should finish in step2: %+v", findResult(report2.Results, "r2"))
	}
}

func TestPrefillFairnessAcrossSteps(t *testing.T) {
	s, err := New(Config{MaxActive: 3, DecodeBudget: 0, PrefillBudget: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	log := &eventLog{}
	works := []*testWork{
		newTestWork("a", 2, 0),
		newTestWork("b", 2, 0),
		newTestWork("c", 2, 0),
	}
	for _, w := range works {
		w.prefillQuantum = 1
		w.log = log
	}
	for i, w := range works {
		if err := s.Add(context.Background(), testRequest{id: w.id, seq: uint64(i + 1)}, w); err != nil {
			t.Fatal(err)
		}
	}

	for i := 0; i < 4; i++ {
		if _, err := s.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if got, want := log.snapshot(), []string{"prefill:a", "prefill:b", "prefill:c", "prefill:a"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("events=%v want %v", got, want)
	}
}

func TestZeroBudgetsStillAdmit(t *testing.T) {
	s, err := New(Config{MaxActive: 1, DecodeBudget: 0, PrefillBudget: 0})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	w := newTestWork("a", 1, 1)
	if err := s.Add(context.Background(), testRequest{id: "a", seq: 1}, w); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := w.snapshot()
	if got.prefillCalls != 0 || got.decodeCalls != 0 {
		t.Fatalf("work progressed with zero budgets: %+v", got)
	}
}

func TestConcurrentAddAndStep(t *testing.T) {
	s, err := New(Config{MaxActive: 8, DecodeBudget: 4, PrefillBudget: 4})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	var addWG sync.WaitGroup
	for g := 0; g < 4; g++ {
		g := g
		addWG.Add(1)
		go func() {
			defer addWG.Done()
			for i := 0; i < 25; i++ {
				id := fmt.Sprintf("g%d-%d", g, i)
				w := newTestWork(id, 1, 1)
				if err := s.Add(context.Background(), testRequest{id: id, seq: uint64(g*100 + i)}, w); err != nil && !errors.Is(err, ErrClosed) {
					t.Errorf("add %s: %v", id, err)
				}
			}
		}()
	}

	var stepWG sync.WaitGroup
	stepWG.Add(1)
	go func() {
		defer stepWG.Done()
		for i := 0; i < 80; i++ {
			_, err := s.Step(context.Background())
			if err != nil && !errors.Is(err, ErrClosed) {
				t.Errorf("step: %v", err)
				return
			}
		}
	}()

	addWG.Wait()
	stepWG.Wait()
	for i := 0; i < 80; i++ {
		if _, err := s.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCloseIsIdempotentAndClosesOnce(t *testing.T) {
	s, err := New(Config{MaxActive: 1, DecodeBudget: 0, PrefillBudget: 0})
	if err != nil {
		t.Fatal(err)
	}

	w1 := newTestWork("a", 1, 0)
	w2 := newTestWork("b", 1, 0)
	if err := s.Add(context.Background(), testRequest{id: "a", seq: 1}, w1); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), testRequest{id: "b", seq: 2}, w2); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Step(context.Background()); err != nil {
		t.Fatal(err)
	}

	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if got := w1.snapshot(); got.cancelCalls != 1 || got.closeCalls != 1 {
		t.Fatalf("running close counts=%+v", got)
	}
	if got := w2.snapshot(); got.cancelCalls != 1 || got.closeCalls != 1 {
		t.Fatalf("waiting close counts=%+v", got)
	}
}

func TestInvalidProgressRemovesRequest(t *testing.T) {
	s, err := New(Config{MaxActive: 1, DecodeBudget: 0, PrefillBudget: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	w := &badWork{testWork: *newTestWork("bad", 1, 0), prefillReturn: 2}
	if err := s.Add(context.Background(), testRequest{id: "bad", seq: 1}, w); err != nil {
		t.Fatal(err)
	}
	report, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	res := findResult(report.Results, "bad")
	if !errors.Is(res.Err, ErrInvalidProgress) {
		t.Fatalf("error=%v", res.Err)
	}
	if report.Stats.Errors != 1 || report.Stats.Running != 0 {
		t.Fatalf("stats=%+v", report.Stats)
	}
}

type badWork struct {
	testWork
	prefillReturn int
}

func (w *badWork) Prefill(int) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.prefillCalls++
	return w.prefillReturn, nil
}

func findResult(results []StepResult, id string) StepResult {
	for _, res := range results {
		if res.RequestID == id {
			return res
		}
	}
	return StepResult{}
}

func resultIDs(results []StepResult) []string {
	out := make([]string, len(results))
	for i, res := range results {
		out[i] = res.RequestID
	}
	return out
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
