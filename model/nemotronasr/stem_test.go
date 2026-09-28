package nemotronasr

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func readStemFixture(t testing.TB, name string, n int) []float32 {
	t.Helper()
	file, err := os.Open("testdata/" + name + ".f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	b, err := io.ReadAll(io.LimitReader(gz, int64(n*4+1)))
	if err != nil || len(b) != n*4 {
		t.Fatalf("%s bytes=%d err=%v", name, len(b), err)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

func TestStemConv2DPyTorchBoundary(t *testing.T) {
	features := readStemFixture(t, "features", 32*128)
	stem := &StemConv2D{Weight: readStemFixture(t, "weight", 256*9), Bias: readStemFixture(t, "bias", 256)}
	want := readStemFixture(t, "output", 256*17*65)
	for _, tc := range []struct {
		name string
		fn   func([]float32, int) ([]float32, error)
	}{{"scalar", stem.ForwardOfflineScalar}, {"SIMD", stem.ForwardOffline}} {
		got, err := tc.fn(features, 32)
		if err != nil || len(got) != len(want) {
			t.Fatalf("%s shape=%d err=%v", tc.name, len(got), err)
		}
		var maxAbs, sumAbs float64
		var outside int
		for i, v := range got {
			d := math.Abs(float64(v - want[i]))
			if d > maxAbs {
				maxAbs = d
			}
			sumAbs += d
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-5+2e-5*math.Abs(float64(want[i])) {
				outside++
			}
		}
		mean := sumAbs / float64(len(got))
		t.Logf("%s stem max_abs=%g mean_abs=%g outside=%d", tc.name, maxAbs, mean, outside)
		if outside != 0 || mean > 3e-6 {
			t.Fatalf("%s outside independent PyTorch numerical distribution", tc.name)
		}
	}
}

func TestReleasedStemConv2DLoad(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	stat, err := f.Stat()
	if err != nil {
		f.Close()
		t.Fatal(err)
	}
	if stat.Size() != 2552062944 {
		f.Close()
		t.Fatal("ASR checkpoint length mismatch")
	}
	sha := sha256.New()
	if _, err := io.Copy(sha, f); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(sha.Sum(nil)) != "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174" {
		t.Fatal("ASR checkpoint integrity mismatch")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	stem, loadErr := LoadStemConv2D(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	weight := readStemFixture(t, "weight", 256*9)
	bias := readStemFixture(t, "bias", 256)
	for i, v := range weight {
		if stem.Weight[i] != v {
			t.Fatalf("stem weight %d changed", i)
		}
	}
	for i, v := range bias {
		if stem.Bias[i] != v {
			t.Fatalf("stem bias %d changed", i)
		}
	}
	features := readStemFixture(t, "features", 32*128)
	got, err := stem.ForwardOffline(features, 32)
	if err != nil || len(got) != 256*17*65 {
		t.Fatalf("loaded stem result=%d err=%v", len(got), err)
	}
}

func TestStemConv2DTailsOwnershipAndMalformed(t *testing.T) {
	stem := &StemConv2D{Weight: make([]float32, stemChannels*9), Bias: make([]float32, stemChannels)}
	for i := range stem.Weight {
		stem.Weight[i] = float32(i%13-6) / 17
	}
	for _, frames := range []int{1, 2, 3, 5, 8, 17, 31, 32} {
		features := make([]float32, frames*128)
		for i := range features {
			features[i] = float32(i%17-8) / 19
		}
		original := append([]float32(nil), features...)
		got, err := stem.ForwardOffline(features, frames)
		if err != nil {
			t.Fatal(err)
		}
		scalar, err := stem.ForwardOfflineScalar(features, frames)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != stemChannels*(frames/2+1)*65 {
			t.Fatalf("frames=%d output=%d", frames, len(got))
		}
		for i := range got {
			if math.Abs(float64(got[i]-scalar[i])) > 3e-5 {
				t.Fatalf("frames=%d row=%d scalar/SIMD drift=%g", frames, i, got[i]-scalar[i])
			}
		}
		for i := range features {
			if features[i] != original[i] {
				t.Fatal("mutated caller")
			}
		}
		stable := append([]float32(nil), got...)
		features[0]++
		if _, err := stem.ForwardOffline(features, frames); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != stable[i] {
				t.Fatal("output retained scratch alias")
			}
		}
	}
	for _, bad := range []struct {
		features []float32
		frames   int
	}{{nil, 0}, {make([]float32, 127), 1}, {make([]float32, 3002*128), 3002}, {[]float32{float32(math.NaN())}, 1}} {
		if _, err := stem.ForwardOffline(bad.features, bad.frames); err == nil {
			t.Fatalf("accepted invalid frames=%d len=%d", bad.frames, len(bad.features))
		}
	}
}

func TestStemScratchReuseAndOwnedOutput(t *testing.T) {
	features := readStemFixture(t, "features", 32*128)
	stem := &StemConv2D{Weight: readStemFixture(t, "weight", 256*9), Bias: readStemFixture(t, "bias", 256)}
	var scratch StemScratch
	first, err := stem.ForwardOfflineScratch(features, 32, &scratch)
	if err != nil {
		t.Fatal(err)
	}
	want, err := stem.ForwardOffline(features, 32)
	if err != nil {
		t.Fatal(err)
	}
	for i := range first {
		if math.Abs(float64(first[i]-want[i])) > 1e-6 {
			t.Fatalf("scratch first mismatch %d", i)
		}
	}
	original := append([]float32(nil), first...)
	for i := range scratch.patches {
		scratch.patches[i] = 42
	}
	second, err := stem.ForwardOfflineScratch(features, 32, &scratch)
	if err != nil {
		t.Fatal(err)
	}
	for i := range second {
		if math.Abs(float64(second[i]-want[i])) > 1e-6 {
			t.Fatalf("scratch reuse mismatch %d", i)
		}
		if first[i] != original[i] {
			t.Fatalf("owned output changed %d", i)
		}
	}
	if len(scratch.patches) != 17*65*9 {
		t.Fatal("unexpected scratch geometry")
	}
}

func BenchmarkNemotronStemScalar(b *testing.B) { benchmarkStem(b, false) }
func BenchmarkNemotronStemSIMD(b *testing.B)   { benchmarkStem(b, true) }
func BenchmarkNemotronStemSIMDReuse(b *testing.B) {
	features := readStemFixture(b, "features", 32*128)
	stem := &StemConv2D{Weight: readStemFixture(b, "weight", 256*9), Bias: readStemFixture(b, "bias", 256)}
	var scratch StemScratch
	if _, err := stem.ForwardOfflineScratch(features, 32, &scratch); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := stem.ForwardOfflineScratch(features, 32, &scratch); err != nil {
			b.Fatal(err)
		}
	}
}
func benchmarkStem(b *testing.B, vector bool) {
	features := readStemFixture(b, "features", 32*128)
	stem := &StemConv2D{Weight: readStemFixture(b, "weight", 256*9), Bias: readStemFixture(b, "bias", 256)}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var err error
		if vector {
			_, err = stem.ForwardOffline(features, 32)
		} else {
			_, err = stem.ForwardOfflineScalar(features, 32)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}
