package audio

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The input is the repository-owned JFK WAV; the fixture was independently
// produced by the pinned Transformers/PyTorch feature extractor, not by Go.
func nemotronJFK(t testing.TB) []float32 {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "jfk.wav")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e" {
		t.Fatal("JFK input fixture provenance changed")
	}
	pcm, rate, err := WAV(path)
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK WAV: rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	return pcm
}

func TestNemotronLogMelReferenceAndDispatch(t *testing.T) {
	input := nemotronJFK(t)
	file, err := os.Open("testdata/nemotron_jfk_transformers_5_18_features.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	reference, err := io.ReadAll(io.LimitReader(gz, 1101*128*4+1))
	if err != nil || len(reference) != 1101*128*4 {
		t.Fatalf("reference length=%d err=%v", len(reference), err)
	}
	var scalar []float32
	for _, impl := range []struct {
		name string
		call func([]float32) ([]float32, int, error)
	}{{"scalar", NemotronLogMelScalar}, {"simd dispatch", NemotronLogMel}} {
		got, frames, err := impl.call(input)
		if err != nil || frames != 1101 || len(got) != 1101*128 {
			t.Fatalf("%s shape: frames=%d len=%d err=%v", impl.name, frames, len(got), err)
		}
		if impl.name == "scalar" {
			scalar = got
		}
		var maxAbs, sumAbs float64
		var over, drift int
		for i, value := range got {
			want := math.Float32frombits(binary.LittleEndian.Uint32(reference[i*4:]))
			delta := math.Abs(float64(value) - float64(want))
			if delta > maxAbs {
				maxAbs = delta
			}
			sumAbs += delta
			if delta > 5e-4+1e-5*math.Abs(float64(want)) {
				over++
			}
			if scalar != nil && impl.name != "scalar" && math.Abs(float64(value-scalar[i])) > 5e-4 {
				drift++
			}
		}
		t.Logf("%s: max_abs=%g mean_abs=%g outside_tolerance=%d scalar_drift=%d", impl.name, maxAbs, sumAbs/float64(len(got)), over, drift)
		if over != 0 || drift != 0 || sumAbs/float64(len(got)) > 1e-5 {
			t.Fatalf("%s differs from independent reference distribution", impl.name)
		}
		for _, v := range got[(frames-1)*128:] {
			if v != 0 {
				t.Fatal("masked final feature frame must be zero")
			}
		}
	}
}

func TestNemotronLogMelBoundsAndOwnership(t *testing.T) {
	for _, bad := range [][]float32{nil, make([]float32, 480001), {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, _, err := NemotronLogMel(bad); err == nil {
			t.Fatalf("accepted invalid input length %d", len(bad))
		}
	}
	for _, n := range []int{1, 159, 160, 161, 511, 512, 513, 1600} {
		input := make([]float32, n)
		for i := range input {
			input[i] = float32((i%11)-5) / 100
		}
		original := append([]float32(nil), input...)
		got, frames, err := NemotronLogMel(input)
		if err != nil || frames != n/160+1 || len(got) != frames*128 {
			t.Fatalf("length %d: frames=%d result=%d err=%v", n, frames, len(got), err)
		}
		for i := range input {
			if input[i] != original[i] {
				t.Fatalf("input mutated at %d", i)
			}
		}
		if &got[0] == &input[0] {
			t.Fatal("output aliases caller")
		}
		copyBefore := append([]float32(nil), got...)
		input[0]++
		if _, _, err = NemotronLogMel(input); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != copyBefore[i] {
				t.Fatalf("returned output changed at %d", i)
			}
		}
	}
}

func BenchmarkNemotronJFKLogMelScalar(b *testing.B) {
	input := nemotronJFK(b)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := NemotronLogMelScalar(input); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNemotronJFKLogMelSIMD(b *testing.B) {
	input := nemotronJFK(b)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := NemotronLogMel(input); err != nil {
			b.Fatal(err)
		}
	}
}
