package vulkan

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// Scalar materialised tile model; shuffled ownership and independent F64 oracle.
func attentionKey32Model(t *testing.T, q, k, v []float32, m, n, h, d int, seed int64) []float32 {
	t.Helper()
	w := h * d
	out := make([]float32, m*w+4)
	for i := m * w; i < len(out); i++ {
		out[i] = 123.25
	}
	writers := make([]int, m*w)
	rng := rand.New(rand.NewSource(seed))
	lanes := rng.Perm(256)
	tiles := (m + 15) / 16
	scale := float32(1 / math.Sqrt(float64(d)))
	for _, g := range rng.Perm(tiles * h) {
		head, queryBase := g/tiles, (g%tiles)*16
		var qt, acc [1024]float32
		var kv [2048]float32
		var p [512]float32
		var rowMax, rowSum, alpha [16]float32
		load := func(dst []float32, src []float32, base, rows int) {
			owners := make([]int, len(dst))
			for _, lane := range lanes {
				for i := lane; i < len(dst); i += 256 {
					r, c := i/64, i%64
					dst[i] = 0
					if base+r < rows && c < d {
						dst[i] = src[(base+r)*w+head*d+c]
					}
					owners[i]++
				}
			}
			for _, x := range owners {
				if x != 1 {
					t.Fatal("shared tile writer")
				}
			}
		}
		load(qt[:], q, queryBase, m)
		for base := 0; base < n; base += 32 {
			load(kv[:], k, base, n)
			for _, lane := range lanes {
				x, y := lane%16, lane/16
				for key := x; key < 32; key += 16 {
					s := float32(0)
					for c := 0; c < d; c++ {
						s = simd.FMA32Scalar(qt[y*64+c], kv[key*64+c], s)
					}
					p[y*32+key] = s * scale
				}
			}
			for _, y := range rng.Perm(16) {
				mx := p[y*32]
				for j := 1; j < 32 && base+j < n; j++ {
					if mx < p[y*32+j] {
						mx = p[y*32+j]
					}
				}
				a := float32(0)
				if base > 0 {
					if mx < rowMax[y] {
						mx = rowMax[y]
					}
					a = float32(math.Exp(float64(float32(rowMax[y] - mx))))
				}
				rowMax[y] = mx
				alpha[y] = a
			}
			// Parallel exponent writers run only after every row maximum is
			// visible. Every probability has exactly one writer per tile.
			var owners [512]int
			for _, lane := range lanes {
				x, y := lane%16, lane/16
				for j := x; j < 32; j += 16 {
					prob := float32(0)
					if base+j < n {
						prob = float32(math.Exp(float64(float32(p[y*32+j] - rowMax[y]))))
					}
					p[y*32+j] = prob
					owners[y*32+j]++
				}
			}
			for _, count := range owners {
				if count != 1 {
					t.Fatal("probability writer", count)
				}
			}
			// Sum retains increasing-key order after the probability barrier.
			for _, y := range rng.Perm(16) {
				sum := float32(0)
				for j := 0; j < 32; j++ {
					sum += p[y*32+j]
				}
				rowSum[y] = float32(float32(rowSum[y]*alpha[y]) + sum)
			}
			load(kv[:], v, base, n)
			for _, lane := range lanes {
				x, y := lane%16, lane/16
				for i := 0; i < 4; i++ {
					c := x + i*16
					idx := y*64 + c
					acc[idx] *= alpha[y]
					if c < d {
						for j := 0; j < 32; j++ {
							acc[idx] = simd.FMA32Scalar(p[y*32+j], kv[j*64+c], acc[idx])
						}
					}
				}
			}
		}
		for _, lane := range lanes {
			x, y := lane%16, lane/16
			if queryBase+y < m {
				if rowSum[y] <= 0 || math.IsNaN(float64(rowSum[y])) {
					t.Fatal("invalid denominator")
				}
				for i := 0; i < 4; i++ {
					c := x + i*16
					if c < d {
						idx := (queryBase+y)*w + head*d + c
						out[idx] = acc[y*64+c] / rowSum[y]
						writers[idx]++
					}
				}
			}
		}
	}
	for _, n := range writers {
		if n != 1 {
			t.Fatal("output ownership", n)
		}
	}
	for _, x := range out[m*w:] {
		if x != 123.25 {
			t.Fatal("tail canary")
		}
	}
	return out[:m*w]
}

func TestVulkanOfflineAttentionKey32Numerics(t *testing.T) {
	for _, dims := range [][4]int{{1, 1, 1, 1}, {2, 3, 2, 3}, {15, 17, 3, 7}, {16, 32, 2, 16}, {17, 31, 2, 17}, {31, 33, 3, 31}, {33, 65, 2, 32}, {3, 129, 2, 63}, {2, 257, 3, 64}, {1, 4096, 1, 1}, {4096, 1, 1, 1}} {
		m, n, h, d := dims[0], dims[1], dims[2], dims[3]
		q, k, v := nativeData(m*h*d, 11, 2), nativeData(n*h*d, 12, 2), nativeData(n*h*d, 13, 2)
		ref := attentionReference(q, k, v, m, n, h, d)
		var first []float32
		for seed := int64(0); seed < 4; seed++ {
			out := attentionKey32Model(t, q, k, v, m, n, h, d, seed)
			if first == nil {
				first = out
			} else if !reflect.DeepEqual(first, out) {
				t.Fatal("scheduling drift", dims)
			}
			for i, x := range out {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x)-ref[i]) > 2e-5+2e-5*math.Abs(ref[i]) {
					t.Fatal("oracle", dims, i, x, ref[i])
				}
			}
		}
	}
}

func TestVulkanOfflineAttentionKey32SoftmaxStress(t *testing.T) {
	for _, sign := range []float32{-1, 0, 1} {
		const m, n, h, d = 17, 97, 2, 3
		const width = h * d
		q, k, v := make([]float32, m*width), make([]float32, n*width), make([]float32, n*width)
		for i := range q {
			q[i] = sign
		}
		for j := 0; j < n; j++ {
			for c := 0; c < width; c++ {
				k[j*width+c] = []float32{-800, 900, 900, -1000}[j/32] + float32(j%3)
				v[j*width+c] = float32(j%7-3) * float32(c+1)
			}
		}
		ref := attentionReference(q, k, v, m, n, h, d)
		for seed := int64(0); seed < 4; seed++ {
			out := attentionKey32Model(t, q, k, v, m, n, h, d, seed)
			for i, x := range out {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.Abs(float64(x)-ref[i]) > 2e-4+2e-5*math.Abs(ref[i]) {
					t.Fatal("rescale", sign, seed, i, x, ref[i])
				}
			}
		}
	}
}
