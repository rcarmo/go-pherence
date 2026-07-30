package kv

import (
	"fmt"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestRingFP16KVWrapViewResetAndAccounting(t *testing.T) {
	s := NewRingFP16KV(2, 4)
	if got := s.Dim(); got != 2 {
		t.Fatalf("Dim=%d want 2", got)
	}
	if got := s.Capacity(); got != 4 {
		t.Fatalf("Capacity=%d want 4", got)
	}
	if got := s.Tokens(); got != 0 {
		t.Fatalf("Tokens=%d want 0", got)
	}
	if got, want := s.Bytes(), int64(0); got != want {
		t.Fatalf("Bytes=%d want %d", got, want)
	}
	if got, want := s.LogicalBytes(), int64(0); got != want {
		t.Fatalf("LogicalBytes=%d want %d", got, want)
	}
	if got, want := s.PhysicalBytes(), int64(4*2*2*2); got != want {
		t.Fatalf("PhysicalBytes=%d want %d", got, want)
	}

	rows := []struct {
		k []float32
		v []float32
	}{
		{k: []float32{1, 10}, v: []float32{101, 110}},
		{k: []float32{2, 20}, v: []float32{102, 120}},
		{k: []float32{3, 30}, v: []float32{103, 130}},
		{k: []float32{4, 40}, v: []float32{104, 140}},
		{k: []float32{5, 50}, v: []float32{105, 150}},
	}
	for i, row := range rows {
		if err := s.Append(row.k, row.v); err != nil {
			t.Fatalf("Append #%d: %v", i+1, err)
		}
	}

	if got := s.Tokens(); got != 4 {
		t.Fatalf("Tokens=%d want 4", got)
	}
	if got, want := s.Bytes(), int64(4*2*2*2); got != want {
		t.Fatalf("Bytes=%d want %d", got, want)
	}
	if got, want := s.LogicalBytes(), int64(4*2*2*2); got != want {
		t.Fatalf("LogicalBytes=%d want %d", got, want)
	}
	if got, want := s.PhysicalBytes(), int64(4*2*2*2); got != want {
		t.Fatalf("PhysicalBytes=%d want %d", got, want)
	}

	view := s.View()
	if view.StartToken != 1 {
		t.Fatalf("View.StartToken=%d want 1", view.StartToken)
	}
	if got, want := view.FirstK, encodeF16([]float32{2, 20, 3, 30, 4, 40}); !sameUint16s(got, want) {
		t.Fatalf("View.FirstK=%v want %v", got, want)
	}
	if got, want := view.FirstV, encodeF16([]float32{102, 120, 103, 130, 104, 140}); !sameUint16s(got, want) {
		t.Fatalf("View.FirstV=%v want %v", got, want)
	}
	if got, want := view.SecondK, encodeF16([]float32{5, 50}); !sameUint16s(got, want) {
		t.Fatalf("View.SecondK=%v want %v", got, want)
	}
	if got, want := view.SecondV, encodeF16([]float32{105, 150}); !sameUint16s(got, want) {
		t.Fatalf("View.SecondV=%v want %v", got, want)
	}

	k, v, startToken := s.MaterializeF32()
	if startToken != 1 {
		t.Fatalf("MaterializeF32.StartToken=%d want 1", startToken)
	}
	if got, want := k, []float32{2, 20, 3, 30, 4, 40, 5, 50}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeF32.K=%v want %v", got, want)
	}
	if got, want := v, []float32{102, 120, 103, 130, 104, 140, 105, 150}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeF32.V=%v want %v", got, want)
	}
	k[0] = 999
	if got, want := half.F16ToF32(s.View().FirstK[0]), float32(2); got != want {
		t.Fatalf("MaterializeF32 returned aliased storage, first K=%v want %v", got, want)
	}

	s.Reset()
	if got := s.Tokens(); got != 0 {
		t.Fatalf("Tokens after Reset=%d want 0", got)
	}
	if got, want := s.Bytes(), int64(0); got != want {
		t.Fatalf("Bytes after Reset=%d want %d", got, want)
	}
	if got, want := s.LogicalBytes(), int64(0); got != want {
		t.Fatalf("LogicalBytes after Reset=%d want %d", got, want)
	}
	if got, want := s.PhysicalBytes(), int64(4*2*2*2); got != want {
		t.Fatalf("PhysicalBytes after Reset=%d want %d", got, want)
	}
	view = s.View()
	if view.StartToken != 0 || len(view.FirstK) != 0 || len(view.FirstV) != 0 || len(view.SecondK) != 0 || len(view.SecondV) != 0 {
		t.Fatalf("View after Reset=%+v", view)
	}
}

func TestRingFP16KVTailsAcrossWraps(t *testing.T) {
	cases := []struct {
		name        string
		appends     int
		wantStart   int
		wantFirstK  []float32
		wantSecondK []float32
		wantFirstV  []float32
		wantSecondV []float32
	}{
		{
			name:        "partial tail without wrap",
			appends:     3,
			wantStart:   0,
			wantFirstK:  []float32{1, 2, 3},
			wantSecondK: nil,
			wantFirstV:  []float32{101, 102, 103},
			wantSecondV: nil,
		},
		{
			name:        "exact full without wrap",
			appends:     5,
			wantStart:   0,
			wantFirstK:  []float32{1, 2, 3, 4, 5},
			wantSecondK: nil,
			wantFirstV:  []float32{101, 102, 103, 104, 105},
			wantSecondV: nil,
		},
		{
			name:        "one row wrapped tail",
			appends:     6,
			wantStart:   1,
			wantFirstK:  []float32{2, 3, 4, 5},
			wantSecondK: []float32{6},
			wantFirstV:  []float32{102, 103, 104, 105},
			wantSecondV: []float32{106},
		},
		{
			name:        "two row wrapped tail",
			appends:     7,
			wantStart:   2,
			wantFirstK:  []float32{3, 4, 5},
			wantSecondK: []float32{6, 7},
			wantFirstV:  []float32{103, 104, 105},
			wantSecondV: []float32{106, 107},
		},
		{
			name:        "long tail after multiple wraps",
			appends:     9,
			wantStart:   4,
			wantFirstK:  []float32{5},
			wantSecondK: []float32{6, 7, 8, 9},
			wantFirstV:  []float32{105},
			wantSecondV: []float32{106, 107, 108, 109},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := NewRingFP16KV(1, 5)
			for i := 1; i <= tc.appends; i++ {
				if err := s.Append([]float32{float32(i)}, []float32{float32(i + 100)}); err != nil {
					t.Fatalf("Append #%d: %v", i, err)
				}
			}

			view := s.View()
			if view.StartToken != tc.wantStart {
				t.Fatalf("View.StartToken=%d want %d", view.StartToken, tc.wantStart)
			}
			if got, want := view.FirstK, encodeF16(tc.wantFirstK); !sameUint16s(got, want) {
				t.Fatalf("View.FirstK=%v want %v", got, want)
			}
			if got, want := view.SecondK, encodeF16(tc.wantSecondK); !sameUint16s(got, want) {
				t.Fatalf("View.SecondK=%v want %v", got, want)
			}
			if got, want := view.FirstV, encodeF16(tc.wantFirstV); !sameUint16s(got, want) {
				t.Fatalf("View.FirstV=%v want %v", got, want)
			}
			if got, want := view.SecondV, encodeF16(tc.wantSecondV); !sameUint16s(got, want) {
				t.Fatalf("View.SecondV=%v want %v", got, want)
			}

			k, v, startToken := s.MaterializeF32()
			if startToken != tc.wantStart {
				t.Fatalf("MaterializeF32.StartToken=%d want %d", startToken, tc.wantStart)
			}
			wantK := append(append([]float32(nil), tc.wantFirstK...), tc.wantSecondK...)
			wantV := append(append([]float32(nil), tc.wantFirstV...), tc.wantSecondV...)
			if !sameFloat32s(k, wantK) {
				t.Fatalf("MaterializeF32.K=%v want %v", k, wantK)
			}
			if !sameFloat32s(v, wantV) {
				t.Fatalf("MaterializeF32.V=%v want %v", v, wantV)
			}
		})
	}
}

func TestRingFP16KVAttentionLikeErrorMetrics(t *testing.T) {
	const dim = 128
	const capacity = 32
	const totalTokens = 48

	s := NewRingFP16KV(dim, capacity)
	wantK := make([]float32, capacity*dim)
	wantV := make([]float32, capacity*dim)
	query := make([]float32, dim)
	fillAttentionLikeQuery(query)

	for token := 0; token < totalTokens; token++ {
		kRow := make([]float32, dim)
		vRow := make([]float32, dim)
		fillAttentionLikeKVRow(token, kRow, vRow)
		if err := s.Append(kRow, vRow); err != nil {
			t.Fatalf("Append token %d: %v", token, err)
		}
		if token >= totalTokens-capacity {
			row := token - (totalTokens - capacity)
			copy(wantK[row*dim:(row+1)*dim], kRow)
			copy(wantV[row*dim:(row+1)*dim], vRow)
		}
	}

	gotK, gotV, startToken := s.MaterializeF32()
	if startToken != totalTokens-capacity {
		t.Fatalf("MaterializeF32.StartToken=%d want %d", startToken, totalTokens-capacity)
	}
	if len(gotK) != len(wantK) || len(gotV) != len(wantV) {
		t.Fatalf("MaterializeF32 lens K=%d/%d V=%d/%d", len(gotK), len(wantK), len(gotV), len(wantV))
	}

	metrics := attentionLikeErrorMetrics(query, wantK, wantV, gotK, gotV, dim)
	t.Logf("attention-like FP16 errors: k_max=%.6g k_rmse=%.6g v_max=%.6g v_rmse=%.6g logits_max=%.6g logits_rmse=%.6g ctx_max=%.6g ctx_rmse=%.6g", metrics.kMaxAbs, metrics.kRMSE, metrics.vMaxAbs, metrics.vRMSE, metrics.logitMaxAbs, metrics.logitRMSE, metrics.ctxMaxAbs, metrics.ctxRMSE)

	if metrics.kMaxAbs > 3e-4 {
		t.Fatalf("K max abs error=%.6g too large", metrics.kMaxAbs)
	}
	if metrics.vMaxAbs > 3e-4 {
		t.Fatalf("V max abs error=%.6g too large", metrics.vMaxAbs)
	}
	if metrics.logitMaxAbs > 2e-3 {
		t.Fatalf("logit max abs error=%.6g too large", metrics.logitMaxAbs)
	}
	if metrics.ctxMaxAbs > 3e-4 {
		t.Fatalf("context max abs error=%.6g too large", metrics.ctxMaxAbs)
	}
}

func TestRingFP16KVGuards(t *testing.T) {
	var nilStore *RingFP16KV
	if err := nilStore.Append([]float32{1}, []float32{2}); err == nil {
		t.Fatal("nil Append succeeded")
	}

	s := NewRingFP16KV(2, 0)
	if err := s.Append([]float32{1, 2}, []float32{3, 4}); err == nil {
		t.Fatal("Append succeeded on zero-capacity ring")
	}

	s = NewRingFP16KV(2, 1)
	if err := s.Append([]float32{1}, []float32{2, 3}); err == nil {
		t.Fatal("Append accepted malformed row")
	}
	if err := s.Append([]float32{1, 2}, []float32{3, 4}); err != nil {
		t.Fatalf("Append setup: %v", err)
	}
	s.startToken = int(^uint(0) >> 1)
	if err := s.Append([]float32{5, 6}, []float32{7, 8}); err == nil {
		t.Fatal("Append accepted startToken overflow")
	}
}

func BenchmarkRingKVAppend(b *testing.B) {
	for _, dim := range []int{128, 512} {
		for _, capacity := range []int{1152, 4224} {
			kRows, vRows := benchmarkRows(64, dim)
			bytesPerAppend := int64(dim * 2 * 4)

			b.Run(fmt.Sprintf("f32/dim=%d/cap=%d", dim, capacity), func(b *testing.B) {
				s := NewRingF32KV(dim, capacity)
				b.ReportAllocs()
				b.SetBytes(bytesPerAppend)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := s.Append(kRows[i%len(kRows)], vRows[i%len(vRows)]); err != nil {
						b.Fatalf("Append #%d: %v", i, err)
					}
				}
			})

			b.Run(fmt.Sprintf("fp16/dim=%d/cap=%d", dim, capacity), func(b *testing.B) {
				s := NewRingFP16KV(dim, capacity)
				b.ReportAllocs()
				b.SetBytes(bytesPerAppend)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := s.Append(kRows[i%len(kRows)], vRows[i%len(vRows)]); err != nil {
						b.Fatalf("Append #%d: %v", i, err)
					}
				}
			})
		}
	}
}

func BenchmarkRingKVMaterialize(b *testing.B) {
	for _, dim := range []int{128, 512} {
		for _, capacity := range []int{1152, 4224} {
			kRows, vRows := benchmarkRows(capacity, dim)
			logicalBytes := int64(dim * capacity * 2 * 4)

			f32 := NewRingF32KV(dim, capacity)
			for i := 0; i < capacity; i++ {
				if err := f32.Append(kRows[i], vRows[i]); err != nil {
					b.Fatalf("prefill f32 #%d: %v", i, err)
				}
			}
			b.Run(fmt.Sprintf("f32/dim=%d/cap=%d", dim, capacity), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(logicalBytes)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					k, v, startToken := f32.Materialize()
					if len(k) != dim*capacity || len(v) != dim*capacity || startToken != 0 {
						b.Fatalf("Materialize mismatch len=%d/%d start=%d", len(k), len(v), startToken)
					}
				}
			})

			fp16 := NewRingFP16KV(dim, capacity)
			for i := 0; i < capacity; i++ {
				if err := fp16.Append(kRows[i], vRows[i]); err != nil {
					b.Fatalf("prefill fp16 #%d: %v", i, err)
				}
			}
			b.Run(fmt.Sprintf("fp16/dim=%d/cap=%d", dim, capacity), func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(logicalBytes)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					k, v, startToken := fp16.MaterializeF32()
					if len(k) != dim*capacity || len(v) != dim*capacity || startToken != 0 {
						b.Fatalf("MaterializeF32 mismatch len=%d/%d start=%d", len(k), len(v), startToken)
					}
				}
			})
		}
	}
}

type attentionMetrics struct {
	kMaxAbs     float64
	kRMSE       float64
	vMaxAbs     float64
	vRMSE       float64
	logitMaxAbs float64
	logitRMSE   float64
	ctxMaxAbs   float64
	ctxRMSE     float64
}

func attentionLikeErrorMetrics(query, wantK, wantV, gotK, gotV []float32, dim int) attentionMetrics {
	m := attentionMetrics{}
	m.kMaxAbs, m.kRMSE = sliceErrorMetrics(wantK, gotK)
	m.vMaxAbs, m.vRMSE = sliceErrorMetrics(wantV, gotV)
	wantLogits := attentionLogits(query, wantK, dim)
	gotLogits := attentionLogits(query, gotK, dim)
	m.logitMaxAbs, m.logitRMSE = sliceErrorMetrics(wantLogits, gotLogits)
	wantCtx := attentionContext(softmax(wantLogits), wantV, dim)
	gotCtx := attentionContext(softmax(gotLogits), gotV, dim)
	m.ctxMaxAbs, m.ctxRMSE = sliceErrorMetrics(wantCtx, gotCtx)
	return m
}

func sliceErrorMetrics(want, got []float32) (maxAbs, rmse float64) {
	if len(want) != len(got) {
		return math.Inf(1), math.Inf(1)
	}
	var sumSq float64
	for i := range want {
		d := math.Abs(float64(want[i] - got[i]))
		if d > maxAbs {
			maxAbs = d
		}
		sumSq += d * d
	}
	if len(want) == 0 {
		return 0, 0
	}
	return maxAbs, math.Sqrt(sumSq / float64(len(want)))
}

func attentionLogits(query, keys []float32, dim int) []float32 {
	rows := len(keys) / dim
	logits := make([]float32, rows)
	scale := float32(1 / math.Sqrt(float64(dim)))
	for row := 0; row < rows; row++ {
		base := row * dim
		var sum float32
		for i := 0; i < dim; i++ {
			sum += query[i] * keys[base+i]
		}
		logits[row] = sum * scale
	}
	return logits
}

func softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	maxLogit := logits[0]
	for _, x := range logits[1:] {
		if x > maxLogit {
			maxLogit = x
		}
	}
	weights := make([]float32, len(logits))
	var sum float64
	for i, x := range logits {
		w := math.Exp(float64(x - maxLogit))
		weights[i] = float32(w)
		sum += w
	}
	inv := float32(1 / sum)
	for i := range weights {
		weights[i] *= inv
	}
	return weights
}

func attentionContext(weights, values []float32, dim int) []float32 {
	ctx := make([]float32, dim)
	for row, w := range weights {
		base := row * dim
		for i := 0; i < dim; i++ {
			ctx[i] += w * values[base+i]
		}
	}
	return ctx
}

func fillAttentionLikeQuery(dst []float32) {
	for i := range dst {
		dst[i] = float32(0.35*math.Sin(0.07*float64(i+1)) + 0.18*math.Cos(0.11*float64((i+1)*(i+3))))
	}
}

func fillAttentionLikeKVRow(token int, k, v []float32) {
	for i := range k {
		x := float64(token + 1)
		y := float64(i + 1)
		k[i] = float32(0.42*math.Sin(0.09*x+0.05*y) + 0.13*math.Cos(0.04*x*y+0.17*y))
		v[i] = float32(0.37*math.Cos(0.06*x-0.08*y) + 0.16*math.Sin(0.03*x*y+0.19*x))
	}
}

func benchmarkRows(rows, dim int) (kRows, vRows [][]float32) {
	kRows = make([][]float32, rows)
	vRows = make([][]float32, rows)
	for row := 0; row < rows; row++ {
		kRows[row] = make([]float32, dim)
		vRows[row] = make([]float32, dim)
		for i := 0; i < dim; i++ {
			kRows[row][i] = float32(((row*17+i*13)%257)-128) / 128
			vRows[row][i] = float32(((row*19+i*11+7)%257)-128) / 128
		}
	}
	return kRows, vRows
}

func encodeF16(src []float32) []uint16 {
	if len(src) == 0 {
		return nil
	}
	dst := make([]uint16, len(src))
	for i, x := range src {
		dst[i] = half.F32ToF16(x)
	}
	return dst
}

func sameUint16s(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
