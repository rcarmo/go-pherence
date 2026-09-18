package omnivoice

import (
	"context"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"reflect"
	"testing"
)

func assertFloat32Exact(t testing.TB, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d want %d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("index %d got %08x want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestBackboneResidentParityAndIdempotence(t *testing.T) {
	b, f := loadBackboneFixture(t)
	want := make([]float32, len(f.Logits))
	if err := b.ForwardInto(context.Background(), want, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	required := b.ResidentRequiredBytes()
	if required <= 0 {
		t.Fatal("resident bytes not reported")
	}
	if b.ResidentBytes() != 0 {
		t.Fatal("resident cache unexpectedly enabled")
	}
	if err := b.EnableResident(context.Background(), required-1); err == nil {
		t.Fatal("resident cache accepted undersized budget")
	}
	if b.ResidentBytes() != 0 || b.resident != nil {
		t.Fatal("budget failure left partial resident state")
	}
	if err := b.EnableResident(context.Background(), required); err != nil {
		t.Fatal(err)
	}
	if b.resident == nil || b.ResidentBytes() != required {
		t.Fatalf("resident bytes %d want %d", b.ResidentBytes(), required)
	}
	got := make([]float32, len(want))
	if err := b.ForwardInto(context.Background(), got, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, want)
	if err := b.EnableResident(context.Background(), 1); err != nil {
		t.Fatal(err)
	}
	if b.ResidentBytes() != required {
		t.Fatalf("resident bytes changed to %d", b.ResidentBytes())
	}
}

func TestBackboneResidentSiblingInheritanceAndCFGParity(t *testing.T) {
	w, err := loader.OpenWeights("../../testdata/omnivoice/backbone")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	cond, err := NewBackbone(w, 4)
	if err != nil {
		t.Fatal(err)
	}
	preexisting, err := NewBackboneSibling(cond, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := cond.EnableResident(context.Background(), cond.ResidentRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	uncond, err := NewBackboneSibling(cond, 2)
	if err != nil {
		t.Fatal(err)
	}
	if cond.resident == nil || uncond.resident != cond.resident {
		t.Fatal("new sibling did not inherit resident cache")
	}
	if preexisting.resident != nil {
		t.Fatal("existing sibling inherited resident cache retroactively")
	}
	if cond.layer != uncond.layer {
		t.Fatal("sibling lost streamed arena sharing")
	}
	streamCond, err := NewBackbone(w, 4)
	if err != nil {
		t.Fatal(err)
	}
	streamUncond, err := NewBackboneSibling(streamCond, 2)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultGenerationConfig()
	cfg.Steps = 4
	shared, err := NewGeneration(cond, uncond, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	streamed, err := NewGeneration(streamCond, streamUncond, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 3, 4, 4, 2, 3, 4, 4}
	audio := []bool{false, false, true, true}
	uncondIDs := []int{4, 4, 4, 4}
	uncondAudio := []bool{true, true}
	got, want := make([]int, 4), make([]int, 4)
	if err = streamed.GenerateInto(context.Background(), want, append([]int(nil), ids...), audio, append([]int(nil), uncondIDs...), uncondAudio); err != nil {
		t.Fatal(err)
	}
	if err = shared.GenerateInto(context.Background(), got, append([]int(nil), ids...), audio, append([]int(nil), uncondIDs...), uncondAudio); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("shared resident cfg got %v want %v", got, want)
	}
}

func TestBackboneResidentReconfigureParityAndZeroAllocs(t *testing.T) {
	b, f := loadBackboneFixture(t)
	wantFull := make([]float32, len(f.Logits))
	if err := b.ForwardInto(context.Background(), wantFull, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	books := len(f.IDs) / f.Tokens
	vocab := b.weights.Config.AudioVocabSize
	streamSmall, err := NewBackbone(b.weights, 2)
	if err != nil {
		t.Fatal(err)
	}
	ids2 := sliceBookMajor(f.IDs, books, f.Tokens, 2)
	audio2 := append([]bool(nil), f.AudioMask[:2]...)
	want2 := make([]float32, books*2*vocab)
	if err = streamSmall.ForwardInto(context.Background(), want2, ids2, audio2, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableResident(context.Background(), b.ResidentRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		tokens int
		ids    []int
		audio  []bool
		want   []float32
	}{
		{tokens: 2, ids: ids2, audio: audio2, want: want2},
		{tokens: f.Tokens, ids: f.IDs, audio: f.AudioMask, want: wantFull},
		{tokens: 2, ids: ids2, audio: audio2, want: want2},
	}
	for i, tc := range cases {
		if err := b.Reconfigure(tc.tokens); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(tc.want))
		if err := b.ForwardInto(context.Background(), got, tc.ids, tc.audio, nil, nil); err != nil {
			t.Fatal(err)
		}
		assertFloat32Exact(t, got, tc.want)
		_ = i
	}
	out2 := make([]float32, len(want2))
	outFull := make([]float32, len(wantFull))
	var allocErr error
	n := testing.AllocsPerRun(10, func() {
		if allocErr = b.Reconfigure(2); allocErr != nil {
			return
		}
		allocErr = b.ForwardInto(context.Background(), out2, ids2, audio2, nil, nil)
		if allocErr != nil {
			return
		}
		if allocErr = b.Reconfigure(f.Tokens); allocErr != nil {
			return
		}
		allocErr = b.ForwardInto(context.Background(), outFull, f.IDs, f.AudioMask, nil, nil)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	if n != 0 {
		t.Fatalf("resident reuse allocations %g", n)
	}
}

func TestBackboneResidentCancelAndBudgetTransactional(t *testing.T) {
	b, f := loadBackboneFixture(t)
	required := b.ResidentRequiredBytes()
	if err := b.EnableResident(context.Background(), required-1); err == nil {
		t.Fatal("resident cache accepted undersized budget")
	}
	if b.resident != nil || b.ResidentBytes() != 0 {
		t.Fatal("budget failure left resident cache attached")
	}
	ctx := &cancelAfterChecks{Context: context.Background(), remaining: 3}
	if err := b.EnableResident(ctx, required); err != context.Canceled {
		t.Fatalf("expected cancellation, got %v", err)
	}
	if b.resident != nil || b.ResidentBytes() != 0 {
		t.Fatal("cancellation left partial resident cache")
	}
	out := make([]float32, len(f.Logits))
	if err := b.ForwardInto(context.Background(), out, f.IDs, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableResident(context.Background(), required); err != nil {
		t.Fatal(err)
	}
	if b.resident == nil || b.ResidentBytes() != required {
		t.Fatal("resident cache did not enable after retry")
	}
}

func TestResidentByteAccountingMatchesLayerArena(t *testing.T) {
	b, _ := loadBackboneFixture(t)
	want := int64(b.layer.Bytes()) * int64(b.weights.Config.LLMConfig.NumHiddenLayers)
	if b.ResidentRequiredBytes() != want {
		t.Fatalf("required%d actualarenas%d", b.ResidentRequiredBytes(), want)
	}
}
