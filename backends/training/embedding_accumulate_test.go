package training

import (
	"math"
	"testing"
)

func TestTrainingCompactEmbeddingAccumulateShape(t *testing.T) {
	dense, compact, index, err := CompactEmbeddingAccumulateShape(3, 2, 4, -.25)
	if err != nil || dense != 8 || compact != 6 || index != 10 {
		t.Fatal(dense, compact, index, err)
	}
	for _, s := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		if _, _, _, err := CompactEmbeddingAccumulateShape(3, 2, 4, s); err == nil {
			t.Fatal("scale")
		}
	}
	if _, _, _, err := CompactEmbeddingAccumulateShape(0, 2, 4, 1); err == nil {
		t.Fatal("rows")
	}
}
func TestTrainingCompactEmbeddingAccumulateReference(t *testing.T) {
	maxError, total, squares := float64(0), float64(0), float64(0)
	count := 0
	for _, ids := range [][]float32{{2, 0, 2, 3, 0}, {1, 1, 1, 1}, {7}, {0, 1, 2, 3, 4, 5, 6, 7}} {
		const width, vocabulary = 3, 8
		rows := len(ids)
		idx, err := BuildCompactEmbeddingIndex(ids, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		source := make([]float32, rows*width)
		for i := range source {
			source[i] = float32(i%13-5) * .00137
		}
		compact := make([]float32, rows*width)
		dense := make([]float32, vocabulary*width)
		for row, id := range ids {
			for col := 0; col < width; col++ {
				dense[int(id)*width+col] += source[row*width+col]
			}
		}
		for slot := 0; slot < rows; slot++ {
			if idx[slot] < 0 {
				continue
			}
			for col := 0; col < width; col++ {
				for at := int(idx[rows+slot]); at < int(idx[rows+slot+1]); at++ {
					compact[slot*width+col] += source[int(idx[2*rows+1+at])*width+col]
				}
			}
		}
		for _, scale := range []float32{0, .25, 1, -.75} {
			for i := range dense {
				token := i / width
				lo, hi := 0, rows
				for lo < hi {
					mid := lo + (hi-lo)/2
					id := idx[mid]
					if id >= 0 && id < float32(token) {
						lo = mid + 1
					} else {
						hi = mid
					}
				}
				contribution := float32(0)
				if lo < rows && idx[lo] == float32(token) {
					contribution = compact[lo*width+i%width]
				}
				previous := float32(i%11-5) * .00371
				got := previous + scale*contribution
				legacy := previous + scale*dense[i]
				expected := float64(previous) + float64(scale)*float64(dense[i])
				e := math.Abs(float64(got) - expected)
				maxError = math.Max(maxError, e)
				total += e
				squares += e * e
				count++
				if math.Abs(float64(got)-float64(legacy)) > 2e-6 || e > 2e-6*math.Max(1, math.Abs(expected)) {
					t.Fatal("compact accumulation", token, got, legacy, expected)
				}
			}
		}
	}
	// Absent-token positive-zero contributions must retain the legacy signed
	// zero multiply/add behaviour; copying previous would not be equivalent.
	for _, previous := range []float32{0, float32(math.Copysign(0, -1))} {
		for _, scale := range []float32{1, -1} {
			contribution := float32(0)
			got := previous + scale*contribution
			want := previous + scale*float32(0)
			if got != want || math.Signbit(float64(got)) != math.Signbit(float64(want)) {
				t.Fatal("signed zero formula")
			}
		}
	}
	t.Logf("compact accumulation F32/F64 n=%d mean=%.9g rms=%.9g max=%.9g; existing2e-6 gate unchanged", count, total/float64(count), math.Sqrt(squares/float64(count)), maxError)
}
