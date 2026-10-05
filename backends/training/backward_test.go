package training

import (
	"math"
	"testing"
)

func TestTrainingBackwardShapeAdmission(t *testing.T) {
	for _, mode := range []uint32{0, 1, 2} {
		if err := LayerNormShape(1024, 16384, mode); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range [][3]int{{0, 1, 0}, {1, 0, 0}, {1025, 1, 0}, {1, 16385, 0}, {1, 1, 3}} {
		if LayerNormShape(s[0], s[1], uint32(s[2])) == nil {
			t.Fatal("norm admitted", s)
		}
	}
	for _, s := range [][4]int{{1, 1, 1, 0}, {1024, 32, 256, 1024}} {
		for mode := uint32(0); mode < 5; mode++ {
			if err := AttentionShape(s[0], s[1], s[2], s[3], mode, .5); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, s := range [][4]int{{0, 1, 1, 0}, {1025, 1, 1, 0}, {1, 0, 1, 0}, {1, 33, 1, 0}, {1, 1, 0, 0}, {1, 1, 257, 0}, {2, 1, 1, -1}, {2, 1, 1, 3}} {
		if AttentionShape(s[0], s[1], s[2], s[3], 0, 1) == nil {
			t.Fatal("attention admitted", s)
		}
	}
	for _, scale := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if AttentionShape(1, 1, 1, 0, 0, scale) == nil {
			t.Fatal("scale admitted")
		}
	}
	if AttentionShape(1, 1, 1, 0, 5, 1) == nil {
		t.Fatal("mode admitted")
	}
}

// Independent F64 forward with stable softmax. The gradient schedule below
// follows GPU one-owner stages in F32; no SPIR-V/PTX execution occurs here.
func attentionObjective(q, k, v, dy []float64, s, h, d, context int) (float64, []float32) {
	width := h * d
	scale := 1 / math.Sqrt(float64(d))
	prob := make([]float32, h*s*s)
	loss := float64(0)
	for head := 0; head < h; head++ {
		for query := 0; query < s; query++ {
			start := 0
			if context > 0 && query+1 > context {
				start = query - context + 1
			}
			scores := make([]float64, query-start+1)
			max := math.Inf(-1)
			for key := start; key <= query; key++ {
				sum := float64(0)
				for c := 0; c < d; c++ {
					sum += q[query*width+head*d+c] * k[key*width+head*d+c]
				}
				scores[key-start] = sum * scale
				max = math.Max(max, sum*scale)
			}
			denominator := float64(0)
			for i, x := range scores {
				scores[i] = math.Exp(x - max)
				denominator += scores[i]
			}
			for key := start; key <= query; key++ {
				p := scores[key-start] / denominator
				prob[(head*s+query)*s+key] = float32(p)
				for c := 0; c < d; c++ {
					loss += p * v[key*width+head*d+c] * dy[query*width+head*d+c]
				}
			}
		}
	}
	return loss, prob
}
func f32Values(x []float64) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return out
}
func attentionBackwardSchedule(q, k, v, dy []float32, p []float32, s, h, d, context int) ([]float32, []float32, []float32, []float32, []float32) {
	width := h * d
	scale := float32(1 / math.Sqrt(float64(d)))
	dp, ds := make([]float32, h*s*s), make([]float32, h*s*s)
	allowed := func(query, key int) bool { return key <= query && (context == 0 || query-key < context) }
	for i := range dp {
		key, query, head := i%s, (i/s)%s, i/(s*s)
		if allowed(query, key) {
			for c := 0; c < d; c++ {
				dp[i] += dy[query*width+head*d+c] * v[key*width+head*d+c]
			}
		}
	}
	for row := 0; row < h*s; row++ {
		query, base := row%s, row*s
		weighted := float32(0)
		for key := 0; key <= query; key++ {
			if allowed(query, key) {
				weighted += p[base+key] * dp[base+key]
			}
		}
		for key := 0; key < s; key++ {
			if allowed(query, key) {
				ds[base+key] = (p[base+key] * (dp[base+key] - weighted)) * scale
			}
		}
	}
	dq, dk, dv := make([]float32, len(q)), make([]float32, len(k)), make([]float32, len(v))
	for i := range dq {
		row, head, c := i/width, (i%width)/d, i%d
		for key := 0; key <= row; key++ {
			if allowed(row, key) {
				dq[i] += ds[(head*s+row)*s+key] * k[key*width+head*d+c]
			}
		}
		for query := row; query < s; query++ {
			if allowed(query, row) {
				idx := (head*s+query)*s + row
				dk[i] += ds[idx] * q[query*width+head*d+c]
				dv[i] += p[idx] * dy[query*width+head*d+c]
			}
		}
	}
	return dq, dk, dv, dp, ds
}
func TestTrainingAttentionFiniteDifference(t *testing.T) {
	var square, max float64
	count := 0
	for _, shape := range [][3]int{{1, 1, 1}, {3, 2, 2}, {5, 2, 3}} {
		s, h, d := shape[0], shape[1], shape[2]
		n := s * h * d
		q, k, v, dy := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
		for i := range q {
			q[i] = float64(float32((i%7 - 3)) * .13)
			k[i] = float64(float32((i*3%11 - 5)) * .09)
			v[i] = float64(float32((i*5%13 - 6)) * .11)
			dy[i] = float64(float32((i%5 - 2)) * .2)
		}
		for _, context := range []int{0, 1, min(2, s), s} {
			_, p := attentionObjective(q, k, v, dy, s, h, d, context)
			// Invalid probabilities deliberately contain garbage; all masks must be
			// applied to reductions as well as stores, never trusted as zero input.
			for head := 0; head < h; head++ {
				for query := 0; query < s; query++ {
					for key := 0; key < s; key++ {
						if key > query || (context > 0 && query-key >= context) {
							p[(head*s+query)*s+key] = 123
						}
					}
				}
			}
			dq, dk, dv, dp, ds := attentionBackwardSchedule(f32Values(q), f32Values(k), f32Values(v), f32Values(dy), p, s, h, d, context)
			for i := range dp {
				key, query := i%s, (i/s)%s
				if key > query || (context > 0 && query-key >= context) {
					if dp[i] != 0 || ds[i] != 0 {
						t.Fatal("masked derivative not zero")
					}
				}
			}
			for j, values := range [][]float64{q, k, v} {
				got := [][]float32{dq, dk, dv}[j]
				for i := range values {
					original := values[i]
					eps := 1e-5
					values[i] = original + eps
					plus, _ := attentionObjective(q, k, v, dy, s, h, d, context)
					values[i] = original - eps
					minus, _ := attentionObjective(q, k, v, dy, s, h, d, context)
					values[i] = original
					want := (plus - minus) / (2 * eps)
					err := math.Abs(float64(got[i]) - want)
					max = math.Max(max, err)
					square += err * err
					count++
					if err > 2e-6 {
						t.Fatal("F32 owner schedule vs independent F64 finite difference", shape, context, j, i, got[i], want, err)
					}
				}
			}
		}
	}
	t.Logf("attention host schedule: values=%d max_abs=%g RMS=%g (no device execution)", count, max, math.Sqrt(square/float64(count)))
}
func TestTrainingLayerNormFiniteDifference(t *testing.T) {
	rows, width := 3, 5
	eps := 1e-5
	x, weight, bias, dy := make([]float64, rows*width), make([]float64, width), make([]float64, width), make([]float64, rows*width)
	for i := range x {
		x[i] = float64(float32((i%7 - 3)) * .21)
		dy[i] = float64(float32((i%5 - 2)) * .15)
	}
	for i := range weight {
		weight[i] = float64(float32(i-2) * .3)
		bias[i] = float64(i) * .1
	}
	forward := func() float64 {
		loss := float64(0)
		for r := 0; r < rows; r++ {
			mean := float64(0)
			for c := 0; c < width; c++ {
				mean += x[r*width+c]
			}
			mean /= float64(width)
			variance := float64(0)
			for c := 0; c < width; c++ {
				delta := x[r*width+c] - mean
				variance += delta * delta
			}
			inv := 1 / math.Sqrt(variance/float64(width)+eps)
			for c := 0; c < width; c++ {
				loss += ((x[r*width+c]-mean)*inv*weight[c] + bias[c]) * dy[r*width+c]
			}
		}
		return loss
	}
	dx, dw, db := make([]float32, len(x)), make([]float32, width), make([]float32, width)
	for r := 0; r < rows; r++ {
		mean := float64(0)
		for c := 0; c < width; c++ {
			mean += x[r*width+c]
		}
		mean /= float64(width)
		variance := float64(0)
		for c := 0; c < width; c++ {
			variance += (x[r*width+c] - mean) * (x[r*width+c] - mean)
		}
		inv := float32(1 / math.Sqrt(variance/float64(width)+eps))
		base := make([]float32, width)
		sum, dot := float32(0), float32(0)
		for c := 0; c < width; c++ {
			base[c] = float32((x[r*width+c] - mean) * float64(inv))
			g := float32(dy[r*width+c]) * float32(weight[c])
			sum += g
			dot += g * base[c]
			dw[c] += float32(dy[r*width+c]) * base[c]
			db[c] += float32(dy[r*width+c])
		}
		for c := 0; c < width; c++ {
			g := float32(dy[r*width+c]) * float32(weight[c])
			dx[r*width+c] = inv * (g - sum/float32(width) - base[c]*dot/float32(width))
		}
	}
	var max, square float64
	count := 0
	for j, values := range [][]float64{x, weight, bias} {
		got := [][]float32{dx, dw, db}[j]
		for i := range values {
			original := values[i]
			h := 1e-5
			values[i] = original + h
			plus := forward()
			values[i] = original - h
			minus := forward()
			values[i] = original
			want := (plus - minus) / (2 * h)
			err := math.Abs(float64(got[i]) - want)
			max = math.Max(max, err)
			square += err * err
			count++
			if err > 2e-6 {
				t.Fatal("LN host schedule vs independent F64 finite difference", j, i, got[i], want)
			}
		}
	}
	t.Logf("layernorm host schedule: values=%d max_abs=%g RMS=%g", count, max, math.Sqrt(square/float64(count)))
}
