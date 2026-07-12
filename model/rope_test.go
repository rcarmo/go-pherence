package model

import (
	"math"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/model/common"
)

func TestBuildRoPEFreqsRejectsOverflowAndBadDims(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if got := buildRoPEFreqs(maxInt, 2, 4, 10000); got != nil {
		t.Fatalf("overflow maxSeq returned len=%d, want nil", len(got))
	}
	if got := buildRoPEFreqs(4, 2, 0, 10000); got != nil {
		t.Fatalf("zero headDim returned len=%d, want nil", len(got))
	}
	if got := buildRoPEFreqs(0, 2, 4, 10000); got != nil {
		t.Fatalf("zero maxSeq returned len=%d, want nil", len(got))
	}
}

func TestBuildRoPEFreqsNegativeThetaFallsBack(t *testing.T) {
	freqs := buildRoPEFreqs(2, 2, 4, -1)
	if len(freqs) != 8 {
		t.Fatalf("len=%d, want 8", len(freqs))
	}
	for i, v := range freqs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("freqs[%d]=%v, want finite", i, v)
		}
	}
}

func TestGemma4RoPEUsesLocalGGMLThetaProgression(t *testing.T) {
	maxSeq, headDim := 1024, 512
	factors := synthesizedGemma4FullAttentionRoPEFactors(headDim)
	got := buildGemma4GGMLRoPEFreqsWithFactors(maxSeq, headDim/2, headDim, 1000000, factors)
	generic := simd.BuildRoPEFreqsWithFactors(maxSeq, headDim/2, headDim, 1000000, factors)
	if len(got) != len(generic) || len(got) == 0 {
		t.Fatalf("bad RoPE table lens got=%d generic=%d", len(got), len(generic))
	}
	// High positions make ggml's float32 iterative theta progression diverge from
	// the shared generic pow-based table. Keep this local to Gemma4 so other model
	// families retain the shared runtime builder they were validated against.
	idx := ((maxSeq-1)*(headDim/2) + 63) * 2
	if got[idx] == generic[idx] && got[idx+1] == generic[idx+1] {
		t.Fatalf("Gemma4 local RoPE unexpectedly matches shared generic table at idx=%d", idx)
	}
}

func TestGemma4RealGGUFFullAttentionRoPEFactors(t *testing.T) {
	g := openLocalGemma4GGUFForTest(t)
	defer g.Close()
	tensor, ok := g.TensorByName("rope_freqs.weight")
	if !ok {
		t.Fatal("missing rope_freqs.weight")
	}
	factors, err := g.DequantF32(tensor)
	if err != nil {
		t.Fatal(err)
	}
	if len(factors) != 256 {
		t.Fatalf("rope_freqs.weight len=%d want 256", len(factors))
	}
	for i := 0; i < 64; i++ {
		if factors[i] != 1 {
			t.Fatalf("rope_freqs[%d]=%g, want active factor 1", i, factors[i])
		}
	}
	for i := 64; i < len(factors); i++ {
		if factors[i] < 1e20 {
			t.Fatalf("rope_freqs[%d]=%g, want disabled large factor", i, factors[i])
		}
	}
}

func TestRoPEGrowsPastInitialCache(t *testing.T) {
	m := &LlamaModel{Config: common.Config{MaxSeqLen: 8192, HeadDim: 8, NumHeads: 1, RopeTheta: 10000}}
	m.precomputeRoPE()
	if got := ropePositions(m.RopeFreqs, 4); got != initialRoPESeqCap {
		t.Fatalf("initial positions=%d, want %d", got, initialRoPESeqCap)
	}
	want := buildRoPEFreqs(4096, 4, 8, 10000)
	for _, pos := range []int{2047, 2048, 4095} {
		freqs := m.ensureRoPE(pos)
		if got := ropePositions(freqs, 4); got <= pos {
			t.Fatalf("position %d: grown positions=%d, want at least %d", pos, got, pos+1)
		}
		off := (pos*4 + 3) * 2
		if freqs[off] != want[off] || freqs[off+1] != want[off+1] {
			t.Fatalf("position %d pair mismatch: got=(%g,%g) want=(%g,%g)", pos, freqs[off], freqs[off+1], want[off], want[off+1])
		}
	}
}

func TestGemma4LayerRoPESelectionPreservesLegacySemantics(t *testing.T) {
	types := []string{"sliding_attention", "full_attention", "unknown"}
	if !gemma4LayerUsesSWA(types, 0) {
		t.Fatal("sliding_attention must use SWA RoPE")
	}
	if gemma4LayerUsesSWA(types, 1) {
		t.Fatal("full_attention must use full RoPE")
	}
	if gemma4LayerUsesSWA(types, 2) {
		t.Fatal("unknown explicit layer type must preserve legacy full-RoPE fallback")
	}
	if !gemma4LayerUsesSWA(types, 3) {
		t.Fatal("missing layer type must preserve legacy SWA default")
	}
}

func TestGemma4RoPEGrowthPreservesLoadedFullFactors(t *testing.T) {
	cfg := common.Config{ModelType: "gemma4_text", MaxSeqLen: 8192, HeadDim: 8, GlobalHeadDim: 16, LayerTypes: []string{"full_attention"}}
	factors := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	m := &LlamaModel{Config: cfg}
	m.precomputeGemma4RoPEWithFullFactors(factors)
	factors[0] = 99 // stored factors must not alias loader scratch storage
	freqs, half := m.ensureGemma4RoPE(0, 4095)
	if half != 8 || ropePositions(freqs, half) < 4096 {
		t.Fatalf("grown full table half/positions=%d/%d, want 8/4096", half, ropePositions(freqs, half))
	}
	wantFactors := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	want := buildGemma4GGMLRoPEFreqsWithFactors(4096, 8, 16, 1000000, wantFactors)
	off := (4095*8 + 0) * 2
	if freqs[off] != want[off] || freqs[off+1] != want[off+1] {
		t.Fatalf("position 4095 loaded-factor pair mismatch: got=(%g,%g) want=(%g,%g)", freqs[off], freqs[off+1], want[off], want[off+1])
	}
}

func TestGemma4MTPDrafterRoPEGrowthPreservesLoadedFullFactors(t *testing.T) {
	cfg := common.Config{ModelType: "gemma4_text", MaxSeqLen: 8192, HeadDim: 8, GlobalHeadDim: 16, LayerTypes: []string{"full_attention"}}
	factors := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	d := &Gemma4MTPDrafter{Config: cfg}
	d.precomputeGemma4RoPEWithFullFactors(factors)
	factors[0] = 99
	freqs, half := d.ensureGemma4RoPE(0, 4095)
	want := buildGemma4GGMLRoPEFreqsWithFactors(4096, 8, 16, 1000000, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	off := (4095*8 + 0) * 2
	if half != 8 || freqs[off] != want[off] || freqs[off+1] != want[off+1] {
		t.Fatalf("drafter position 4095 mismatch half=%d got=(%g,%g) want=(%g,%g)", half, freqs[off], freqs[off+1], want[off], want[off+1])
	}
}

func TestConcurrentRoPEGrowthCoversLargestPosition(t *testing.T) {
	m := &LlamaModel{Config: common.Config{MaxSeqLen: 8192, HeadDim: 8, NumHeads: 1, RopeTheta: 10000}}
	// Deliberately skip precomputeRoPE so the first-use state initialization is
	// exercised concurrently as well as the table growth.
	done := make(chan struct{}, 4)
	for _, pos := range []int{2047, 2048, 3072, 4095} {
		go func(pos int) {
			_ = m.ensureRoPE(pos)
			done <- struct{}{}
		}(pos)
	}
	for range 4 {
		<-done
	}
	state := m.ensureRoPEState()
	state.mu.RLock()
	covered := ropePositions(m.RopeFreqs, 4)
	state.mu.RUnlock()
	if covered < 4096 {
		t.Fatalf("concurrent growth covers %d positions, want at least 4096", covered)
	}
}

func TestGemma4FullAttentionRoPEUsesFullHeadFactors(t *testing.T) {
	m := &LlamaModel{Config: common.Config{ModelType: "gemma4_text", MaxSeqLen: 8, HeadDim: 256, GlobalHeadDim: 512}}
	m.precomputeGemma4RoPE()
	if m.RopeHalfSWA != 128 {
		t.Fatalf("RopeHalfSWA=%d, want 128", m.RopeHalfSWA)
	}
	if m.RopeHalfFull != 256 {
		t.Fatalf("RopeHalfFull=%d, want full-head half 256", m.RopeHalfFull)
	}
	if got, want := len(m.RopeFreqsFull), 8*256*2; got != want {
		t.Fatalf("full RoPE table len=%d, want %d", got, want)
	}
	factors := synthesizedGemma4FullAttentionRoPEFactors(512)
	if len(factors) != 256 {
		t.Fatalf("factors len=%d, want 256", len(factors))
	}
	for i := 0; i < 64; i++ {
		if factors[i] != 1 {
			t.Fatalf("factor[%d]=%g, want 1", i, factors[i])
		}
	}
	for i := 64; i < len(factors); i++ {
		if factors[i] < 1e20 {
			t.Fatalf("factor[%d]=%g, want disabled large factor", i, factors[i])
		}
	}
}
