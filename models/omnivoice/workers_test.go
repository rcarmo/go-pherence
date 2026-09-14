package omnivoice

import (
	"context"
	"errors"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func makeWorkerFixture(t *testing.T, tokens int) (*Backbone, residentPrepackedFixture, []int, []bool) {
	t.Helper()
	seed, f := loadResidentPrepackedFixture(t)
	b, err := NewBackbone(seed.weights, tokens)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Close() })
	ids := make([]int, f.config.NumAudioCodebook*tokens)
	audio := make([]bool, tokens)
	for t := 0; t < tokens; t++ {
		isAudio := t%4 != 0
		audio[t] = isAudio
		for book := 0; book < f.config.NumAudioCodebook; book++ {
			if isAudio {
				ids[book*tokens+t] = (t + book*3 + 1) % f.config.AudioVocabSize
				continue
			}
			if book == 0 {
				ids[t] = (t*5 + 1) % f.config.LLMConfig.VocabSize
			}
		}
	}
	return b, f, ids, audio
}

func workerLogitsLen(f residentPrepackedFixture, tokens int) int {
	return f.config.NumAudioCodebook * tokens * f.config.AudioVocabSize
}

func forwardWorkerFixture(t *testing.T, b *Backbone, f residentPrepackedFixture, ids []int, audio []bool) []float32 {
	t.Helper()
	out := make([]float32, workerLogitsLen(f, len(audio)))
	if err := b.ForwardInto(context.Background(), out, ids, audio, nil, nil); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestBackboneWorkersStreamingSingleWorkerExactZeroAllocsAndClose(t *testing.T) {
	b, f, ids, audio := makeWorkerFixture(t, 13)
	want := forwardWorkerFixture(t, b, f, ids, audio)
	if err := b.EnableWorkers(1); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableWorkers(1); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableWorkers(2); err == nil {
		t.Fatal("worker pool reconfiguration accepted")
	}
	if err := b.EnableWorkers(0); err == nil {
		t.Fatal("zero workers accepted")
	}
	if b.pool == nil || !b.ownsPool || b.poolWorkers != 1 {
		t.Fatal("single-worker pool not attached as owned direct path")
	}
	got := make([]float32, len(want))
	if err := b.ForwardInto(context.Background(), got, ids, audio, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
	var err error
	if n := testing.AllocsPerRun(10, func() {
		err = b.ForwardInto(context.Background(), got, ids, audio, nil, nil)
	}); err != nil {
		t.Fatal(err)
	} else if n != 0 {
		t.Fatalf("streaming worker allocs %g", n)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if b.pool != nil || b.ownsPool || b.poolWorkers != 0 {
		t.Fatal("close left worker pool attached")
	}
	if err := b.ForwardInto(context.Background(), got, ids, audio, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
}

func TestBackboneWorkersResidentExactZeroAllocs(t *testing.T) {
	b, f, ids, audio := makeWorkerFixture(t, 13)
	want := forwardWorkerFixture(t, b, f, ids, audio)
	if err := b.EnableResident(context.Background(), b.ResidentRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableWorkers(3); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(want))
	if err := b.ForwardInto(context.Background(), got, ids, audio, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
	var err error
	if n := testing.AllocsPerRun(10, func() {
		err = b.ForwardInto(context.Background(), got, ids, audio, nil, nil)
	}); err != nil {
		t.Fatal(err)
	} else if n != 0 {
		t.Fatalf("resident worker allocs %g", n)
	}
}

func TestBackboneWorkersResidentPrepackedExactZeroAllocs(t *testing.T) {
	b, f, ids, audio := makeWorkerFixture(t, 13)
	want := forwardWorkerFixture(t, b, f, ids, audio)
	if err := b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableWorkers(3); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(want))
	if err := b.ForwardInto(context.Background(), got, ids, audio, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
	var err error
	if n := testing.AllocsPerRun(10, func() {
		err = b.ForwardInto(context.Background(), got, ids, audio, nil, nil)
	}); err != nil {
		t.Fatal(err)
	} else if n != 0 {
		t.Fatalf("prepacked worker allocs %g", n)
	}
}

func TestBackboneWorkersSiblingInheritanceBorrowAndCloseOwnership(t *testing.T) {
	parent, f, ids, audio := makeWorkerFixture(t, 13)
	older, err := NewBackboneSibling(parent, 13)
	if err != nil {
		t.Fatal(err)
	}
	if err = parent.EnableWorkers(3); err != nil {
		t.Fatal(err)
	}
	newer, err := NewBackboneSibling(parent, 13)
	if err != nil {
		t.Fatal(err)
	}
	if older.pool != nil || older.poolWorkers != 0 || older.ownsPool {
		t.Fatal("existing sibling inherited workers retroactively")
	}
	if newer.pool != parent.pool || newer.poolWorkers != parent.poolWorkers || newer.ownsPool {
		t.Fatal("new sibling did not borrow parent worker pool")
	}
	want := forwardWorkerFixture(t, parent, f, ids, audio)
	got := forwardWorkerFixture(t, newer, f, ids, audio)
	assertFloat32Exact(t, got, want)
	if err := newer.Close(); err != nil {
		t.Fatal(err)
	}
	if parent.pool == nil || !parent.ownsPool || parent.poolWorkers != 3 {
		t.Fatal("borrower close affected parent-owned pool")
	}
	got = forwardWorkerFixture(t, parent, f, ids, audio)
	assertFloat32Exact(t, got, want)
}

func TestBackboneWorkersCancelRecovery(t *testing.T) {
	b, f, ids, audio := makeWorkerFixture(t, 13)
	if err := b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableWorkers(3); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, workerLogitsLen(f, len(audio)))
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 5}
	if err := b.ForwardInto(ctx, out, ids, audio, nil, nil); err != context.Canceled {
		t.Fatalf("expected cancellation, got %v", err)
	}
	want := forwardWorkerFixture(t, b, f, ids, audio)
	if err := b.ForwardInto(context.Background(), out, ids, audio, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, out, want)
}

func TestBackboneBorrowerRejectsClosedOwnerPool(t *testing.T) {
	ctx := context.Background()
	parent, f, ids, audio := makeWorkerFixture(t, 13)
	if err := parent.EnableWorkers(2); err != nil {
		t.Fatal(err)
	}
	borrower, err := NewBackboneSibling(parent, 13)
	if err != nil {
		t.Fatal(err)
	}
	defer borrower.Close()
	if err := parent.Close(); err != nil {
		t.Fatal(err)
	}
	out := make([]float32, workerLogitsLen(f, 13))
	if err := borrower.ForwardInto(ctx, out, ids, audio, nil, nil); !errors.Is(err, simd.ErrGEMMPoolClosed) {
		t.Fatalf("closed owner: %v", err)
	}
}
