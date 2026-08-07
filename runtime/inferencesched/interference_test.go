package inferencesched

import (
	"context"
	"testing"
	"time"
)

type timedRequest struct {
	id  string
	seq uint64
}

func (r timedRequest) ID() string         { return r.id }
func (r timedRequest) ArrivalSeq() uint64 { return r.seq }

type timedWork struct {
	prefill, decode           int
	prefillDelay, decodeDelay time.Duration
	decodeTimes               []time.Time
	closed                    bool
}

func (w *timedWork) RemainingPrefill() int { return w.prefill }
func (w *timedWork) Prefill(n int) (int, error) {
	if n > w.prefill {
		n = w.prefill
	}
	if n > 0 {
		time.Sleep(time.Duration(n) * w.prefillDelay)
		w.prefill -= n
	}
	return n, nil
}
func (w *timedWork) Decode(n int) (int, error) {
	if n > w.decode {
		n = w.decode
	}
	for i := 0; i < n; i++ {
		time.Sleep(w.decodeDelay)
		w.decodeTimes = append(w.decodeTimes, time.Now())
	}
	w.decode -= n
	return n, nil
}
func (w *timedWork) Finished() bool { return w.prefill == 0 && w.decode == 0 }
func (w *timedWork) Cancel()        {}
func (w *timedWork) Close() error   { w.closed = true; return nil }

type interferenceResult struct {
	activeFirst, shortFirst, total time.Duration
	activeITL                      time.Duration
}

func runInterference(t *testing.T, prefillBudget int) interferenceResult {
	t.Helper()
	s, err := New(Config{MaxActive: 3, DecodeBudget: 1, PrefillBudget: prefillBudget})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	active := &timedWork{decode: 4, decodeDelay: time.Millisecond}
	long := &timedWork{prefill: 8, decode: 1, prefillDelay: 2 * time.Millisecond, decodeDelay: time.Millisecond}
	short := &timedWork{prefill: 1, decode: 1, prefillDelay: 2 * time.Millisecond, decodeDelay: time.Millisecond}
	start := time.Now()
	for i, x := range []struct {
		id string
		w  *timedWork
	}{{"active", active}, {"long", long}, {"short", short}} {
		if err := s.Add(context.Background(), timedRequest{x.id, uint64(i)}, x.w); err != nil {
			t.Fatal(err)
		}
	}
	for step := 0; step < 32; step++ {
		r, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if r.Stats.Running == 0 && r.Stats.Waiting == 0 {
			break
		}
	}
	res := interferenceResult{total: time.Since(start)}
	if len(active.decodeTimes) > 0 {
		res.activeFirst = active.decodeTimes[0].Sub(start)
	}
	if len(active.decodeTimes) > 1 {
		res.activeITL = active.decodeTimes[1].Sub(active.decodeTimes[0])
	}
	if len(short.decodeTimes) > 0 {
		res.shortFirst = short.decodeTimes[0].Sub(start)
	}
	return res
}

func TestDecodeFirstChunkedPrefillReducesInterference(t *testing.T) {
	monolithic := runInterference(t, 8)
	chunked := runInterference(t, 1)
	if chunked.activeFirst >= monolithic.activeFirst {
		t.Fatalf("active first chunked=%v monolithic=%v", chunked.activeFirst, monolithic.activeFirst)
	}
	if chunked.activeITL >= monolithic.activeITL {
		t.Fatalf("active ITL chunked=%v monolithic=%v", chunked.activeITL, monolithic.activeITL)
	}
	if chunked.shortFirst >= monolithic.shortFirst {
		t.Fatalf("short first chunked=%v monolithic=%v", chunked.shortFirst, monolithic.shortFirst)
	}
	if chunked.total > monolithic.total*2 {
		t.Fatalf("chunked total=%v excessive vs monolithic=%v", chunked.total, monolithic.total)
	}
}
