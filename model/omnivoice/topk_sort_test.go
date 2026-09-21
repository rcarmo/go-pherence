package omnivoice

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

func insertionTopKReference(vals []float32, idx []int, scores []float32, k int, tokens []int, maskID int) int {
	if k <= 0 || len(scores) == 0 {
		return 0
	}
	if k > len(scores) {
		k = len(scores)
	}
	n := 0
	for i, v := range scores {
		if tokens != nil && tokens[i] != maskID {
			continue
		}
		if n < k {
			vals[n] = v
			idx[n] = i
			siftUpWorst(vals, idx, n)
			n++
			continue
		}
		if betterScore(v, i, vals[0], idx[0]) {
			vals[0] = v
			idx[0] = i
			siftDownWorst(vals[:n], idx[:n], 0)
		}
	}
	for i := 1; i < n; i++ {
		vv, ii := vals[i], idx[i]
		j := i
		for j > 0 && betterScore(vv, ii, vals[j-1], idx[j-1]) {
			vals[j], idx[j] = vals[j-1], idx[j-1]
			j--
		}
		vals[j], idx[j] = vv, ii
	}
	return n
}

func TestTopKHeapSortExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(12, 34))
	for _, size := range []int{1, 7, 32, 33, 103, 600, 2000} {
		for _, kind := range []string{"random", "ties", "infinities", "nan"} {
			scores := make([]float32, size)
			tokens := make([]int, size)
			for i := range scores {
				scores[i] = rng.Float32()*20 - 10
				tokens[i] = i % 2
				if kind == "ties" {
					scores[i] = float32(i % 3)
				}
				if kind == "infinities" {
					if i%2 == 0 {
						scores[i] = float32(math.Inf(-1))
					} else {
						scores[i] = float32(math.Inf(1))
					}
				}
				if kind == "nan" && i%5 == 0 {
					scores[i] = float32(math.NaN())
				}
			}
			for _, mask := range [][]int{nil, tokens} {
				for _, k := range []int{0, 1, 31, 32, 33, 103, size, size + 1} {
					want, got := make([]float32, size), make([]float32, size)
					wi, gi := make([]int, size), make([]int, size)
					n := insertionTopKReference(want, wi, scores, k, mask, 1)
					m := selectTopKStableMasked(got, gi, scores, k, mask, 1)
					if n != m {
						t.Fatal("count mismatch")
					}
					for i := 0; i < n; i++ {
						if wi[i] != gi[i] || math.Float32bits(want[i]) != math.Float32bits(got[i]) {
							t.Fatalf("size=%d kind=%s k=%d row=%d", size, kind, k, i)
						}
					}
					if a := testing.AllocsPerRun(2, func() { selectTopKStableMasked(got, gi, scores, k, mask, 1) }); a != 0 {
						t.Fatalf("allocs %v", a)
					}
				}
			}
		}
	}
}
func BenchmarkTopKFinalSort(b *testing.B) {
	for _, k := range []int{16, 32, 64, 103, 300, 600, 2000} {
		for _, heap := range []bool{false, true} {
			b.Run(fmt.Sprintf("k%d/heap%v", k, heap), func(b *testing.B) {
				size := max(1025, k)
				scores := make([]float32, size)
				vals := make([]float32, size)
				idx := make([]int, size)
				rng := rand.New(rand.NewPCG(12, 34))
				for i := range scores {
					scores[i] = rng.Float32()
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if heap {
						selectTopKStableMasked(vals, idx, scores, k, nil, 0)
					} else {
						insertionTopKReference(vals, idx, scores, k, nil, 0)
					}
				}
			})
		}
	}
}

func TestTopKHeapSortMixedExceptionalValues(t *testing.T) {
	rng := rand.New(rand.NewPCG(98, 76))
	for trial := 0; trial < 1000; trial++ {
		const size = 257
		scores := make([]float32, size)
		for i := range scores {
			scores[i] = rng.Float32()
			switch rng.IntN(32) {
			case 0:
				scores[i] = float32(math.NaN())
			case 1:
				scores[i] = float32(math.Inf(-1))
			case 2:
				scores[i] = float32(math.Inf(1))
			case 3:
				scores[i] = float32(math.Copysign(0, -1))
			case 4:
				scores[i] = 0
			}
		}
		k := 33 + rng.IntN(size-32)
		a, b := make([]float32, size), make([]float32, size)
		ai, bi := make([]int, size), make([]int, size)
		n := insertionTopKReference(a, ai, scores, k, nil, 0)
		selectTopKStableMasked(b, bi, scores, k, nil, 0)
		for i := 0; i < n; i++ {
			if ai[i] != bi[i] || math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("trial %d k%d entry%d", trial, k, i)
			}
		}
	}
}
