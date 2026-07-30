package kv

import "testing"

func TestF32KVStoreImplementations(t *testing.T) {
	var _ F32KVStore = (*LinearF32KV)(nil)
	var _ F32KVStore = (*RingF32KV)(nil)
}

func TestLinearF32KVBasic(t *testing.T) {
	s := NewLinearF32KV(2)
	if got := s.Dim(); got != 2 {
		t.Fatalf("Dim=%d want 2", got)
	}
	if got := s.Capacity(); got != 0 {
		t.Fatalf("Capacity=%d want 0", got)
	}
	if got := s.Tokens(); got != 0 {
		t.Fatalf("Tokens=%d want 0", got)
	}
	if got := s.Bytes(); got != 0 {
		t.Fatalf("Bytes=%d want 0", got)
	}

	if err := s.Append([]float32{1, 2}, []float32{10, 20}); err != nil {
		t.Fatalf("Append #1: %v", err)
	}
	if err := s.Append([]float32{3, 4}, []float32{30, 40}); err != nil {
		t.Fatalf("Append #2: %v", err)
	}

	view := s.View()
	if view.StartToken != 0 {
		t.Fatalf("View.StartToken=%d want 0", view.StartToken)
	}
	if !sameFloat32s(view.FirstK, []float32{1, 2, 3, 4}) {
		t.Fatalf("View.FirstK=%v", view.FirstK)
	}
	if !sameFloat32s(view.FirstV, []float32{10, 20, 30, 40}) {
		t.Fatalf("View.FirstV=%v", view.FirstV)
	}
	if len(view.SecondK) != 0 || len(view.SecondV) != 0 {
		t.Fatalf("linear view unexpectedly wrapped: secondK=%v secondV=%v", view.SecondK, view.SecondV)
	}

	k, v, startToken := s.Materialize()
	if startToken != 0 {
		t.Fatalf("Materialize.StartToken=%d want 0", startToken)
	}
	if !sameFloat32s(k, []float32{1, 2, 3, 4}) || !sameFloat32s(v, []float32{10, 20, 30, 40}) {
		t.Fatalf("Materialize mismatch K=%v V=%v", k, v)
	}
	k[0] = 99
	if got := s.View().FirstK[0]; got != 1 {
		t.Fatalf("Materialize returned aliased K, store[0]=%v", got)
	}

	if got, want := s.Bytes(), int64(2*2*2*4); got != want {
		t.Fatalf("Bytes=%d want %d", got, want)
	}

	cp := s.Checkpoint()
	if err := s.Append([]float32{5, 6}, []float32{50, 60}); err != nil {
		t.Fatalf("Append #3: %v", err)
	}
	if err := s.Restore(cp); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	k, v, startToken = s.Materialize()
	if startToken != 0 || !sameFloat32s(k, []float32{1, 2, 3, 4}) || !sameFloat32s(v, []float32{10, 20, 30, 40}) {
		t.Fatalf("restored mismatch start=%d K=%v V=%v", startToken, k, v)
	}

	s.Reset()
	if got := s.Tokens(); got != 0 {
		t.Fatalf("Tokens after Reset=%d want 0", got)
	}
	if got := s.View().StartToken; got != 0 {
		t.Fatalf("StartToken after Reset=%d want 0", got)
	}
}

func TestRingF32KVWrapAndRestore(t *testing.T) {
	s := NewRingF32KV(2, 3)
	for _, row := range []struct {
		k []float32
		v []float32
	}{
		{k: []float32{1, 2}, v: []float32{10, 20}},
		{k: []float32{3, 4}, v: []float32{30, 40}},
		{k: []float32{5, 6}, v: []float32{50, 60}},
		{k: []float32{7, 8}, v: []float32{70, 80}},
	} {
		if err := s.Append(row.k, row.v); err != nil {
			t.Fatalf("Append(%v,%v): %v", row.k, row.v, err)
		}
	}

	if got := s.Tokens(); got != 3 {
		t.Fatalf("Tokens=%d want 3", got)
	}
	if got := s.Capacity(); got != 3 {
		t.Fatalf("Capacity=%d want 3", got)
	}
	if got, want := s.Bytes(), int64(3*2*2*4); got != want {
		t.Fatalf("Bytes=%d want %d", got, want)
	}

	view := s.View()
	if view.StartToken != 1 {
		t.Fatalf("View.StartToken=%d want 1", view.StartToken)
	}
	if !sameFloat32s(view.FirstK, []float32{3, 4, 5, 6}) {
		t.Fatalf("View.FirstK=%v", view.FirstK)
	}
	if !sameFloat32s(view.FirstV, []float32{30, 40, 50, 60}) {
		t.Fatalf("View.FirstV=%v", view.FirstV)
	}
	if !sameFloat32s(view.SecondK, []float32{7, 8}) {
		t.Fatalf("View.SecondK=%v", view.SecondK)
	}
	if !sameFloat32s(view.SecondV, []float32{70, 80}) {
		t.Fatalf("View.SecondV=%v", view.SecondV)
	}

	k, v, startToken := s.Materialize()
	if startToken != 1 {
		t.Fatalf("Materialize.StartToken=%d want 1", startToken)
	}
	if !sameFloat32s(k, []float32{3, 4, 5, 6, 7, 8}) || !sameFloat32s(v, []float32{30, 40, 50, 60, 70, 80}) {
		t.Fatalf("Materialize mismatch start=%d K=%v V=%v", startToken, k, v)
	}

	cp := s.Checkpoint()
	if err := s.Append([]float32{9, 10}, []float32{90, 100}); err != nil {
		t.Fatalf("Append #5: %v", err)
	}
	k, v, startToken = s.Materialize()
	if startToken != 2 || !sameFloat32s(k, []float32{5, 6, 7, 8, 9, 10}) || !sameFloat32s(v, []float32{50, 60, 70, 80, 90, 100}) {
		t.Fatalf("post-wrap mismatch start=%d K=%v V=%v", startToken, k, v)
	}
	if err := s.Restore(cp); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	k, v, startToken = s.Materialize()
	if startToken != 1 || !sameFloat32s(k, []float32{3, 4, 5, 6, 7, 8}) || !sameFloat32s(v, []float32{30, 40, 50, 60, 70, 80}) {
		t.Fatalf("restored mismatch start=%d K=%v V=%v", startToken, k, v)
	}
}

func TestF32KVMalformedAndOverflowGuards(t *testing.T) {
	linear := NewLinearF32KV(2)
	if err := linear.Append([]float32{1}, []float32{2, 3}); err == nil {
		t.Fatal("linear Append accepted malformed row")
	}

	ring := NewRingF32KV(2, 0)
	if err := ring.Append([]float32{1, 2}, []float32{3, 4}); err == nil {
		t.Fatal("ring Append accepted zero-capacity store")
	}

	cp := linear.Checkpoint()
	cp.valid = false
	if err := linear.Restore(cp); err == nil {
		t.Fatal("Restore accepted invalid checkpoint")
	}

	cp = linear.Checkpoint()
	cp.tokens = 1
	if err := linear.Restore(cp); err == nil {
		t.Fatal("Restore accepted checkpoint with mismatched lengths")
	}

	cp = linear.Checkpoint()
	cp.tokens = int(^uint(0) >> 1)
	cp.dim = 2
	if err := linear.Restore(cp); err == nil {
		t.Fatal("Restore accepted overflowing checkpoint")
	}

	ring = NewRingF32KV(2, 1)
	if err := ring.Append([]float32{1, 2}, []float32{3, 4}); err != nil {
		t.Fatalf("Append to full ring setup: %v", err)
	}
	ring.startToken = int(^uint(0) >> 1)
	if err := ring.Append([]float32{5, 6}, []float32{7, 8}); err == nil {
		t.Fatal("ring Append accepted startToken overflow")
	}

	ring = NewRingF32KV(2, 1)
	cp = F32KVCheckpoint{valid: true, dim: 2, tokens: 2, k: make([]float32, 4), v: make([]float32, 4)}
	if err := ring.Restore(cp); err == nil {
		t.Fatal("ring Restore accepted checkpoint beyond capacity")
	}
}

func TestF32KVCrossRestorePreservesStartToken(t *testing.T) {
	ring := NewRingF32KV(2, 3)
	for _, row := range []struct {
		k []float32
		v []float32
	}{
		{k: []float32{1, 2}, v: []float32{10, 20}},
		{k: []float32{3, 4}, v: []float32{30, 40}},
		{k: []float32{5, 6}, v: []float32{50, 60}},
		{k: []float32{7, 8}, v: []float32{70, 80}},
	} {
		if err := ring.Append(row.k, row.v); err != nil {
			t.Fatalf("ring Append(%v,%v): %v", row.k, row.v, err)
		}
	}

	linear := NewLinearF32KV(2)
	if err := linear.Restore(ring.Checkpoint()); err != nil {
		t.Fatalf("cross Restore: %v", err)
	}
	k, v, startToken := linear.Materialize()
	if startToken != 1 {
		t.Fatalf("Materialize.StartToken=%d want 1", startToken)
	}
	if !sameFloat32s(k, []float32{3, 4, 5, 6, 7, 8}) || !sameFloat32s(v, []float32{30, 40, 50, 60, 70, 80}) {
		t.Fatalf("cross-restore mismatch start=%d K=%v V=%v", startToken, k, v)
	}
}
