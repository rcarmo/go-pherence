package omnivoice

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"testing"
)

func packedAttentionReference(s *Workspace, attended, q, k, v []float32, tokens, d, nh, nkv int, mask, qhead, khead, vhead []float32) error {
	scores, headout := s.scores, s.headout
	for head := 0; head < nh; head++ {
		kh := head / (nh / nkv)
		for t := 0; t < tokens; t++ {
			copy(qhead[t*d:(t+1)*d], q[(t*nh+head)*d:(t*nh+head+1)*d])
			copy(khead[t*d:(t+1)*d], k[(t*nkv+kh)*d:(t*nkv+kh+1)*d])
			copy(vhead[t*d:(t+1)*d], v[(t*nkv+kh)*d:(t*nkv+kh+1)*d])
		}
		clear(scores)
		simd.SgemmNTTo(scores, qhead, khead, tokens, tokens, d, float32(1/math.Sqrt(float64(d))), d, d, tokens)
		if mask != nil {
			simd.VecAdd(scores, scores, mask)
		}
		for t := 0; t < tokens; t++ {
			if !simd.SoftmaxSIMDInPlace(scores[t*tokens : (t+1)*tokens]) {
				return fmt.Errorf("omnivoice: attention softmax failed")
			}
		}
		clear(headout)
		simd.SgemmNNTo(headout, scores, vhead, tokens, d, tokens, 1, tokens, d, d)
		for t := 0; t < tokens; t++ {
			copy(attended[(t*nh+head)*d:(t*nh+head+1)*d], headout[t*d:(t+1)*d])
		}
	}
	return nil
}

func attentionInputs(tokens, d, nh, nkv int) (*Workspace, []float32, []float32, []float32, []float32) {
	s := &Workspace{scores: make([]float32, tokens*tokens), headout: make([]float32, tokens*d), qhead: make([]float32, tokens*d), khead: make([]float32, tokens*d), vhead: make([]float32, tokens*d)}
	q, k, v := make([]float32, tokens*d*nh), make([]float32, tokens*d*nkv), make([]float32, tokens*d*nkv)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*.71)) * .3
	}
	for i := range k {
		k[i] = float32(math.Cos(float64(i)*.13)) * .2
		v[i] = float32(math.Sin(float64(i) * .23))
	}
	return s, make([]float32, len(q)), q, k, v
}
func TestAttentionStridedExact(t *testing.T) {
	for _, shape := range [][4]int{{1, 8, 2, 1}, {9, 16, 4, 4}, {11, 16, 8, 1}, {7, 10, 6, 2}, {19, 32, 8, 2}, {75, 128, 16, 8}, {210, 128, 16, 8}} {
		tokens, d, nh, nkv := shape[0], shape[1], shape[2], shape[3]
		s, got, q, k, v := attentionInputs(tokens, d, nh, nkv)
		want := make([]float32, len(got))
		qh, kh, vh := make([]float32, tokens*d), make([]float32, tokens*d), make([]float32, tokens*d)
		mask := make([]float32, tokens*tokens)
		for i := range mask {
			if i%tokens != i/tokens && i%3 == 0 {
				mask[i] = float32(math.Inf(-1))
			}
		}
		for _, mask := range [][]float32{nil, mask} {
			if err := packedAttentionReference(s, want, q, k, v, tokens, d, nh, nkv, mask, qh, kh, vh); err != nil {
				t.Fatal(err)
			}
			if err := stridedAttentionCandidate(s, got, q, k, v, tokens, d, nh, nkv, mask); err != nil {
				t.Fatal(err)
			}
			assertFloat32Exact(t, got, want)
			if err := attentionInto(s, got, q, k, v, tokens, d, nh, nkv, mask); err != nil {
				t.Fatal(err)
			}
			assertFloat32Exact(t, got, want)
			if n := testing.AllocsPerRun(3, func() {
				if err := stridedAttentionCandidate(s, got, q, k, v, tokens, d, nh, nkv, mask); err != nil {
					t.Fatal(err)
				}
			}); n != 0 {
				t.Fatalf("allocs %g", n)
			}
		}
	}
}
func BenchmarkAttentionStrided(b *testing.B) {
	for _, tokens := range []int{75, 210} {
		for _, strided := range []bool{false, true} {
			b.Run(fmt.Sprintf("tokens%d/strided%v", tokens, strided), func(b *testing.B) {
				const d, nh, nkv = 128, 16, 8
				s, out, q, k, v := attentionInputs(tokens, d, nh, nkv)
				qh, kh, vh := make([]float32, tokens*d), make([]float32, tokens*d), make([]float32, tokens*d)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var err error
					if strided {
						err = stridedAttentionCandidate(s, out, q, k, v, tokens, d, nh, nkv, nil)
					} else {
						err = packedAttentionReference(s, out, q, k, v, tokens, d, nh, nkv, nil, qh, kh, vh)
					}
					if err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// attentionInto reads strided head views directly. Only scores and one output
// head are materialised; the SIMD kernels preserve the packed path arithmetic.
func stridedAttentionCandidate(s *Workspace, attended, q, k, v []float32, tokens, d, nh, nkv int, mask []float32) error {
	scores, headout := s.scores, s.headout
	for head := 0; head < nh; head++ {
		kh := head / (nh / nkv)
		clear(scores)
		simd.SgemmNTTo(scores, q[head*d:], k[kh*d:], tokens, tokens, d, float32(1/math.Sqrt(float64(d))), nh*d, nkv*d, tokens)
		if mask != nil {
			simd.VecAdd(scores, scores, mask)
		}
		for t := 0; t < tokens; t++ {
			if !simd.SoftmaxSIMDInPlace(scores[t*tokens : (t+1)*tokens]) {
				return fmt.Errorf("omnivoice: attention softmax failed")
			}
		}
		clear(headout)
		simd.SgemmNNTo(headout, scores, v[kh*d:], tokens, d, tokens, 1, tokens, nkv*d, d)
		for t := 0; t < tokens; t++ {
			copy(attended[(t*nh+head)*d:(t*nh+head+1)*d], headout[t*d:(t+1)*d])
		}
	}
	return nil
}

func BenchmarkAttentionKVReuse(b *testing.B) {
	for _, tokens := range []int{75, 210} {
		b.Run(fmt.Sprint(tokens), func(b *testing.B) {
			s, out, q, k, v := attentionInputs(tokens, 128, 16, 8)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := attentionInto(s, out, q, k, v, tokens, 128, 16, 8, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
