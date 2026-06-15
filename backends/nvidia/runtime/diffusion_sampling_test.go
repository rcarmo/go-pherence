package nvidia

import (
	"math"
	"testing"
)

func cpuDenseSample(logits []float32, uniforms []float32, positions, vocab int, invTemp float32) ([]int, []float64, []int) {
	arg := make([]int, positions)
	ent := make([]float64, positions)
	sam := make([]int, positions)
	for p := 0; p < positions; p++ {
		row := logits[p*vocab : (p+1)*vocab]
		m := float32(math.Inf(-1))
		amax := 0
		for i, v := range row {
			z := v * invTemp
			if z > m {
				m, amax = z, i
			}
		}
		var zsum, tsum float64
		for _, v := range row {
			d := float64(v*invTemp - m)
			e := math.Exp(d)
			zsum += e
			tsum += d * e
		}
		arg[p] = amax
		ent[p] = math.Log(zsum) - tsum/zsum
		target := float64(uniforms[p]) * zsum
		cum := 0.0
		sam[p] = vocab - 1
		for i, v := range row {
			cum += math.Exp(float64(v*invTemp - m))
			if cum >= target {
				sam[p] = i
				break
			}
		}
	}
	return arg, ent, sam
}

func TestDiffusionSoftmaxRowsSynthetic(t *testing.T) {
	if !SgemmReady() {
		t.Skip("GPU unavailable")
	}
	positions, vocab := 2, 19
	logits := make([]float32, positions*vocab)
	for i := range logits {
		logits[i] = float32((i*17)%23-11) * 0.07
	}
	inBuf, err := Malloc(len(logits))
	if err != nil {
		t.Fatal(err)
	}
	defer inBuf.Free()
	outBuf, err := Malloc(len(logits))
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Free()
	if err := inBuf.Upload(logits); err != nil {
		t.Fatal(err)
	}
	if err := DiffusionSoftmaxRows(inBuf, outBuf, positions, vocab, 1.3); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(logits))
	if err := outBuf.Download(got); err != nil {
		t.Fatal(err)
	}
	for p := 0; p < positions; p++ {
		row := logits[p*vocab : (p+1)*vocab]
		m := float32(math.Inf(-1))
		for _, v := range row {
			if z := v * 1.3; z > m {
				m = z
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v*1.3 - m))
		}
		var gotSum float64
		for i, v := range row {
			want := math.Exp(float64(v*1.3-m)) / sum
			g := float64(got[p*vocab+i])
			gotSum += g
			if math.Abs(g-want) > 2e-3 {
				t.Fatalf("softmax row=%d col=%d got=%g want=%g", p, i, g, want)
			}
		}
		if math.Abs(gotSum-1) > 2e-3 {
			t.Fatalf("softmax row=%d sum=%g", p, gotSum)
		}
	}
}

func TestDiffusionDenseSampleSynthetic(t *testing.T) {
	if !SgemmReady() {
		t.Skip("GPU unavailable")
	}
	positions, vocab := 3, 17
	logits := make([]float32, positions*vocab)
	for i := range logits {
		logits[i] = float32((i*37)%19-9) * 0.13
	}
	uniforms := []float32{0.1, 0.5, 0.9}
	buf, err := Malloc(len(logits))
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Free()
	if err := buf.Upload(logits); err != nil {
		t.Fatal(err)
	}
	gotArg, gotEnt, gotSam, err := DiffusionDenseSample(buf, uniforms, positions, vocab, 1.25)
	if err != nil {
		t.Fatal(err)
	}
	wantArg, wantEnt, wantSam := cpuDenseSample(logits, uniforms, positions, vocab, 1.25)
	for i := 0; i < positions; i++ {
		if gotArg[i] != wantArg[i] {
			t.Fatalf("arg[%d]=%d want %d", i, gotArg[i], wantArg[i])
		}
		if gotSam[i] != wantSam[i] {
			t.Fatalf("sample[%d]=%d want %d (entropy got=%v want=%v)", i, gotSam[i], wantSam[i], gotEnt, wantEnt)
		}
		if math.Abs(gotEnt[i]-wantEnt[i]) > 5e-3 {
			t.Fatalf("entropy[%d]=%g want %g", i, gotEnt[i], wantEnt[i])
		}
	}
}
