package expertstream

import (
	"bytes"
	"errors"
	"io"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestLoad_EmptyKeysReturnsNil(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	defer r.Close()
	out, err := r.Load(nil)
	if err != nil || out != nil {
		t.Fatalf("Load(nil) = %v, %v; want nil, nil", out, err)
	}
}

func TestLoad_DuplicateKeysPreserveOrderAndShareSlot(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 3})
	defer r.Close()

	out, err := r.Load([]uint64{1, 2, 1, 3, 2})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("len(out) = %d, want 5", len(out))
	}
	wantKeys := []uint64{1, 2, 1, 3, 2}
	for i, want := range wantKeys {
		if out[i].Key != want {
			t.Fatalf("out[%d].Key = %d, want %d", i, out[i].Key, want)
		}
	}
	// Duplicate requests for the same key must resolve to the same slot and bytes.
	if out[0].Slot.Index != out[2].Slot.Index {
		t.Fatalf("key=1 slot indices differ: %d vs %d", out[0].Slot.Index, out[2].Slot.Index)
	}
	if !bytes.Equal(out[0].Gate.Bytes, out[2].Gate.Bytes) {
		t.Fatalf("key=1 gate bytes differ across duplicate requests")
	}
	if out[1].Slot.Index != out[4].Slot.Index {
		t.Fatalf("key=2 slot indices differ: %d vs %d", out[1].Slot.Index, out[4].Slot.Index)
	}
}

func TestLoad_RequestOrderDoesNotChangeSlotAssignment(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 3})
	defer r.Close()

	first, err := r.Load([]uint64{1, 2})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	slotOf := map[uint64]int{first[0].Key: first[0].Slot.Index, first[1].Key: first[1].Slot.Index}

	second, err := r.Load([]uint64{2, 1})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, le := range second {
		if le.Slot.Index != slotOf[le.Key] {
			t.Fatalf("key=%d slot changed from %d to %d on reorder", le.Key, slotOf[le.Key], le.Slot.Index)
		}
	}
}

func TestLoad_DeterministicLRUSlotReuse(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	defer r.Close()

	// Step 1: cold start fills unloaded slots ascending -> key1 gets slot 0.
	out1, err := r.Load([]uint64{1})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if out1[0].Slot.Index != 0 {
		t.Fatalf("key=1 slot = %d, want 0", out1[0].Slot.Index)
	}

	// Step 2: key2 takes the remaining unloaded slot 1.
	out2, err := r.Load([]uint64{2})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if out2[0].Slot.Index != 1 {
		t.Fatalf("key=2 slot = %d, want 1", out2[0].Slot.Index)
	}

	// Step 3: both slots loaded; key1 (touched earlier -> smaller lastUse) is
	// the LRU victim, so key3 must evict slot 0.
	out3, err := r.Load([]uint64{3})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if out3[0].Slot.Index != 0 {
		t.Fatalf("key=3 slot = %d, want 0 (evicting LRU key=1)", out3[0].Slot.Index)
	}

	// Step 4: key1 is no longer resident; requesting it again must evict the
	// now-LRU key2 from slot 1, deterministically.
	out4, err := r.Load([]uint64{1})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if out4[0].Slot.Index != 1 {
		t.Fatalf("key=1 reload slot = %d, want 1 (evicting LRU key=2)", out4[0].Slot.Index)
	}

	// key3 was touched most recently among {2,3} at step 3/4, so it should
	// still be resident at slot 0.
	out5, err := r.Load([]uint64{3})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if out5[0].Slot.Index != 0 {
		t.Fatalf("key=3 should still be resident at slot 0, got %d", out5[0].Slot.Index)
	}
}

func TestLoad_TouchingResidentKeysProtectsFromEviction(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	defer r.Close()

	if _, err := r.Load([]uint64{1, 2}); err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Re-touch key1 so it becomes more-recently-used than key2.
	if _, err := r.Load([]uint64{1}); err != nil {
		t.Fatalf("Load: %v", err)
	}
	// key2 is now LRU; loading key3 must evict key2's slot, not key1's.
	out, err := r.Load([]uint64{1, 3})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, le := range out {
		if le.Key == 1 && le.Slot.Index != 0 {
			t.Fatalf("key=1 slot changed unexpectedly to %d", le.Slot.Index)
		}
	}
}

func TestLoad_UnknownExpertKey(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	defer r.Close()
	_, err := r.Load([]uint64{9999})
	if !errors.Is(err, ErrUnknownExpert) {
		t.Fatalf("err = %v, want ErrUnknownExpert", err)
	}
}

func TestLoad_UnknownKeyAmongKnownKeysStillErrors(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 3})
	defer r.Close()
	_, err := r.Load([]uint64{1, 2, 9999})
	if !errors.Is(err, ErrUnknownExpert) {
		t.Fatalf("err = %v, want ErrUnknownExpert", err)
	}
}

func TestLoad_SlotCapacityExceeded(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	defer r.Close()
	_, err := r.Load([]uint64{1, 2, 3})
	if !errors.Is(err, ErrSlotCapacity) {
		t.Fatalf("err = %v, want ErrSlotCapacity", err)
	}
}

func TestLoad_SlotCapacityExceededAcrossCalls(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 1})
	defer r.Close()
	if _, err := r.Load([]uint64{1}); err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Two unique keys but only 1 slot total.
	_, err := r.Load([]uint64{2, 3})
	if !errors.Is(err, ErrSlotCapacity) {
		t.Fatalf("err = %v, want ErrSlotCapacity", err)
	}
}

func TestLoad_OnClosedReader(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, err := r.Load([]uint64{1})
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("err = %v, want ErrClosed", err)
	}
}

func TestClose_Idempotent(t *testing.T) {
	r, _, _ := mustOpenFixture(t, Options{Slots: 2})
	if err := r.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestClose_NilReceiver(t *testing.T) {
	var r *Reader
	if err := r.Close(); err != nil {
		t.Fatalf("Close on nil = %v, want nil", err)
	}
}

func TestLoad_TruncatedFileYieldsShortRead(t *testing.T) {
	dir := t.TempDir()
	defs := []expertDef{
		{key: 1, gateN: 4, upN: 4, downN: 4},
		{key: 2, gateN: 64, upN: 64, downN: 64}, // large enough to survive truncation of key1 region
	}
	manifest, _, dataPath := buildFixture(t, dir, 64, defs)
	path := writeManifest(t, dir, manifest)

	r, err := Open(path, Options{Slots: 2})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	// Load key2 first so it is resident and unaffected by later truncation.
	if _, err := r.Load([]uint64{2}); err != nil {
		t.Fatalf("Load key2: %v", err)
	}

	// Truncate the backing file out from under the already-open *os.File so
	// that reading key1's span now hits EOF partway through.
	if err := os.Truncate(dataPath, 4); err != nil {
		t.Fatalf("truncate: %v", err)
	}

	_, err = r.Load([]uint64{1})
	if !errors.Is(err, ErrShortRead) {
		t.Fatalf("err = %v, want ErrShortRead", err)
	}
}

func TestLoad_DiscardsPartialAssignmentsOnError(t *testing.T) {
	dir := t.TempDir()
	defs := []expertDef{
		{key: 1, gateN: 4, upN: 4, downN: 4},
		{key: 2, gateN: 4, upN: 4, downN: 4},
	}
	manifest, _, dataPath := buildFixture(t, dir, 64, defs)
	path := writeManifest(t, dir, manifest)

	r, err := Open(path, Options{Slots: 2, Workers: 1})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	if err := os.Truncate(dataPath, 4); err != nil {
		t.Fatalf("truncate: %v", err)
	}

	_, err = r.Load([]uint64{1, 2})
	if !errors.Is(err, ErrShortRead) {
		t.Fatalf("err = %v, want ErrShortRead", err)
	}
	// After a failed Load, no slot should be left marked as loaded for keys
	// that failed to populate (discardAssignmentsLocked must have run).
	r.mu.Lock()
	for _, sl := range r.slots {
		if sl.loaded {
			t.Errorf("slot %d unexpectedly left loaded after failed Load", sl.index)
		}
	}
	r.mu.Unlock()
	if len(r.keyToSlot) != 0 {
		t.Errorf("keyToSlot not empty after failed Load: %v", r.keyToSlot)
	}
}

// --- Bounded workers -------------------------------------------------------

func TestLoad_BoundedWorkers_NoGoroutineLeakAcrossConfigs(t *testing.T) {
	for _, workers := range []int{0, 1, 2, 8, 100} {
		workers := workers
		t.Run("", func(t *testing.T) {
			dir := t.TempDir()
			defs := make([]expertDef, 0, 6)
			for i := uint64(1); i <= 6; i++ {
				defs = append(defs, expertDef{key: i, gateN: 32, upN: 32, downN: 32})
			}
			manifest, _, _ := buildFixture(t, dir, 64, defs)
			path := writeManifest(t, dir, manifest)

			r, err := Open(path, Options{Slots: 6, Workers: workers})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer r.Close()

			runtime.GC()
			before := runtime.NumGoroutine()

			keys := make([]uint64, 0, 6)
			for i := uint64(1); i <= 6; i++ {
				keys = append(keys, i)
			}
			if _, err := r.Load(keys); err != nil {
				t.Fatalf("Load: %v", err)
			}

			// Worker goroutines are joined via wg.Wait() before Load returns,
			// so goroutine count must settle back near baseline.
			deadline := time.Now().Add(time.Second)
			var after int
			for {
				runtime.GC()
				after = runtime.NumGoroutine()
				if after <= before+1 || time.Now().After(deadline) {
					break
				}
				time.Sleep(5 * time.Millisecond)
			}
			if after > before+1 {
				t.Errorf("workers=%d: goroutine leak: before=%d after=%d", workers, before, after)
			}
		})
	}
}

func TestLoad_BoundedWorkers_PeakGoroutineDeltaIsBounded(t *testing.T) {
	const workers = 2
	dir := t.TempDir()
	defs := make([]expertDef, 0, 8)
	for i := uint64(1); i <= 8; i++ {
		// Reasonably sized payloads so the read isn't instantaneous.
		defs = append(defs, expertDef{key: i, gateN: 1 << 14, upN: 1 << 14, downN: 1 << 14})
	}
	manifest, _, _ := buildFixture(t, dir, 64, defs)
	path := writeManifest(t, dir, manifest)

	r, err := Open(path, Options{Slots: 8, Workers: workers})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()

	runtime.GC()
	baseline := runtime.NumGoroutine()

	stop := make(chan struct{})
	peak := make(chan int, 1)
	go func() {
		max := 0
		ticker := time.NewTicker(50 * time.Microsecond)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				peak <- max
				return
			case <-ticker.C:
				if d := runtime.NumGoroutine() - baseline; d > max {
					max = d
				}
			}
		}
	}()

	keys := make([]uint64, 0, 8)
	for i := uint64(1); i <= 8; i++ {
		keys = append(keys, i)
	}
	if _, err := r.Load(keys); err != nil {
		t.Fatalf("Load: %v", err)
	}
	close(stop)
	maxDelta := <-peak

	// The pool must never spawn more than `workers` extra goroutines
	// (generous tolerance for scheduler/test noise).
	if maxDelta > workers+4 {
		t.Errorf("observed goroutine delta %d exceeds bound for workers=%d", maxDelta, workers)
	}
}

// --- readFullAt (direct, fake ReadAt) --------------------------------------

type step struct {
	n   int
	err error
}

type fakeReaderAt struct {
	steps []step
	calls int
}

func (f *fakeReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if f.calls >= len(f.steps) {
		panic("fakeReaderAt: no more steps configured")
	}
	s := f.steps[f.calls]
	f.calls++
	n := s.n
	if n > len(p) {
		n = len(p)
	}
	for i := 0; i < n; i++ {
		p[i] = byte(off) + byte(i)
	}
	return n, s.err
}

func TestReadFullAt_EmptyDestinationSkipsRead(t *testing.T) {
	f := &fakeReaderAt{} // no steps configured; ReadAt must not be called
	if err := readFullAt(f, "path", nil, 0); err != nil {
		t.Fatalf("readFullAt(empty) = %v, want nil", err)
	}
}

func TestReadFullAt_MultiChunkSuccess(t *testing.T) {
	f := &fakeReaderAt{steps: []step{{n: 3, err: nil}, {n: 2, err: nil}, {n: 1, err: nil}}}
	dst := make([]byte, 6)
	if err := readFullAt(f, "path", dst, 100); err != nil {
		t.Fatalf("readFullAt = %v, want nil", err)
	}
	if f.calls != 3 {
		t.Fatalf("calls = %d, want 3", f.calls)
	}
}

func TestReadFullAt_EOFWithFullDataIsOK(t *testing.T) {
	// io.EOF returned alongside a fully-satisfied read must not be an error.
	f := &fakeReaderAt{steps: []step{{n: 4, err: io.EOF}}}
	dst := make([]byte, 4)
	if err := readFullAt(f, "path", dst, 0); err != nil {
		t.Fatalf("readFullAt = %v, want nil", err)
	}
}

func TestReadFullAt_EarlyEOFIsShortRead(t *testing.T) {
	f := &fakeReaderAt{steps: []step{{n: 2, err: io.EOF}}}
	dst := make([]byte, 5)
	err := readFullAt(f, "somefile", dst, 10)
	if !errors.Is(err, ErrShortRead) {
		t.Fatalf("err = %v, want ErrShortRead", err)
	}
}

func TestReadFullAt_ZeroProgressIsShortRead(t *testing.T) {
	f := &fakeReaderAt{steps: []step{{n: 0, err: nil}}}
	dst := make([]byte, 3)
	err := readFullAt(f, "somefile", dst, 0)
	if !errors.Is(err, ErrShortRead) {
		t.Fatalf("err = %v, want ErrShortRead", err)
	}
}

func TestReadFullAt_GenericErrorPropagates(t *testing.T) {
	boom := errors.New("boom")
	f := &fakeReaderAt{steps: []step{{n: 0, err: boom}}}
	dst := make([]byte, 3)
	err := readFullAt(f, "somefile", dst, 0)
	if err == nil || !errors.Is(err, boom) {
		t.Fatalf("err = %v, want wrapped boom", err)
	}
	if errors.Is(err, ErrShortRead) {
		t.Fatalf("generic error should not be classified as ErrShortRead")
	}
}

func TestReadFullAt_PartialThenGenericError(t *testing.T) {
	boom := errors.New("boom")
	f := &fakeReaderAt{steps: []step{{n: 2, err: nil}, {n: 0, err: boom}}}
	dst := make([]byte, 5)
	err := readFullAt(f, "somefile", dst, 0)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want wrapped boom", err)
	}
}
