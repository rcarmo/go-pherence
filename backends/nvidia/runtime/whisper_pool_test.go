package nvidia

import (
	"math"
	"testing"
)

func TestWhisperAttentivePoolParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA SGEMM not available")
	}
	channels, length, attnDim := 4, 6, 3
	h := make([]float32, channels*length)
	attnW := make([]float32, attnDim*channels)
	attnB := make([]float32, attnDim)
	v := make([]float32, attnDim)
	for i := range h {
		h[i] = float32((i%13)-6) * 0.047
	}
	for i := range attnW {
		attnW[i] = float32((i%11)-5) * 0.039
	}
	for i := range attnB {
		attnB[i] = float32((i%7)-3) * 0.021
	}
	for i := range v {
		v[i] = float32((i%5)-2) * 0.033
	}
	vBias := float32(0.017)
	want := whisperAttentivePoolCPU(h, attnW, attnB, v, vBias, channels, length, attnDim)

	outBuf, err := Malloc(2 * channels)
	if err != nil {
		t.Fatalf("out malloc: %v", err)
	}
	defer outBuf.Free()
	hBuf, err := Malloc(len(h))
	if err != nil {
		t.Fatalf("h malloc: %v", err)
	}
	defer hBuf.Free()
	wBuf, err := Malloc(len(attnW))
	if err != nil {
		t.Fatalf("attnW malloc: %v", err)
	}
	defer wBuf.Free()
	bBuf, err := Malloc(len(attnB))
	if err != nil {
		t.Fatalf("attnB malloc: %v", err)
	}
	defer bBuf.Free()
	vBuf, err := Malloc(len(v))
	if err != nil {
		t.Fatalf("v malloc: %v", err)
	}
	defer vBuf.Free()
	if err := hBuf.Upload(h); err != nil {
		t.Fatalf("h upload: %v", err)
	}
	if err := wBuf.Upload(attnW); err != nil {
		t.Fatalf("attnW upload: %v", err)
	}
	if err := bBuf.Upload(attnB); err != nil {
		t.Fatalf("attnB upload: %v", err)
	}
	if err := vBuf.Upload(v); err != nil {
		t.Fatalf("v upload: %v", err)
	}
	if err := WhisperAttentivePoolBuffer(outBuf, hBuf, wBuf, bBuf, vBuf, channels, length, attnDim, vBias); err != nil {
		t.Fatalf("WhisperAttentivePoolBuffer: %v", err)
	}
	got := make([]float32, 2*channels)
	if err := outBuf.Download(got); err != nil {
		t.Fatalf("download: %v", err)
	}
	assertRuntimeClose(t, got, want, 5e-4)
}

func whisperAttentivePoolCPU(h, attnW, attnB, v []float32, vBias float32, channels, length, attnDim int) []float32 {
	out := make([]float32, 2*channels)
	for c := 0; c < channels; c++ {
		scores := make([]float64, length)
		maxScore := math.Inf(-1)
		for t := 0; t < length; t++ {
			score := float64(vBias)
			for a := 0; a < attnDim; a++ {
				inner := float64(attnB[a])
				for ch := 0; ch < channels; ch++ {
					inner += float64(attnW[a*channels+ch]) * float64(h[ch*length+t])
				}
				score += float64(v[a]) * math.Tanh(inner)
			}
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}
		var sumW, mean, second float64
		for t := 0; t < length; t++ {
			w := math.Exp(scores[t] - maxScore)
			hv := float64(h[c*length+t])
			sumW += w
			mean += w * hv
			second += w * hv * hv
		}
		mean /= sumW
		second /= sumW
		variance := second - mean*mean
		if variance < 0 {
			variance = 0
		}
		out[c] = float32(mean)
		out[channels+c] = float32(math.Sqrt(variance))
	}
	return out
}

func assertRuntimeClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got=%d want=%d", len(got), len(want))
	}
	maxDiff := 0.0
	maxIdx := -1
	for i := range got {
		d := math.Abs(float64(got[i] - want[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	if maxDiff > tol {
		t.Fatalf("max diff %.6g at %d exceeds %.6g: got=%g want=%g", maxDiff, maxIdx, tol, got[maxIdx], want[maxIdx])
	}
}
