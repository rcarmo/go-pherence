package nemotrondiarization

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

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedStacking(t testing.TB) *StackingProjection {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	fileForHash, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	stat, err := fileForHash.Stat()
	if err != nil {
		fileForHash.Close()
		t.Fatal(err)
	}
	if stat.Size() != 396954592 {
		fileForHash.Close()
		t.Fatal("Nemotron diarization checkpoint length mismatch")
	}
	sha := sha256.New()
	if _, err = io.Copy(sha, fileForHash); err != nil {
		fileForHash.Close()
		t.Fatal(err)
	}
	if err = fileForHash.Close(); err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(sha.Sum(nil)) != "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6" {
		t.Fatal("Nemotron diarization checkpoint integrity mismatch")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, loadErr := LoadStackingProjection(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	return model
}

func TestStackingProjectionTinyAndOwnership(t *testing.T) {
	weight := make([]float32, projectedWidth*stackWidth)
	for channel := 0; channel < projectedWidth; channel++ {
		weight[channel*stackWidth+(channel%stackWidth)] = float32(channel+1) / 512
	}
	model := &StackingProjection{weight: weight}
	for _, frames := range []int{1, 7, 8, 9, 15, 16, 17} {
		input := make([]float32, frames*melBins)
		for i := range input {
			input[i] = float32(i%17-8) / 17
		}
		original := append([]float32(nil), input...)
		got, err := model.Project(input, frames)
		if err != nil {
			t.Fatal(err)
		}
		scalar, err := model.ProjectScalar(input, frames)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != ((frames+7)/8)*projectedWidth {
			t.Fatal("incorrect stack output shape")
		}
		for row := range got {
			if math.Abs(float64(got[row]-scalar[row])) > 2e-5 {
				t.Fatalf("frames=%d row=%d SIMD/scalar drift", frames, row)
			}
		}
		for row := 0; row < (frames+7)/8; row++ {
			for ch := 0; ch < projectedWidth; ch++ {
				index := row*stackWidth + ch%stackWidth
				var want float32
				if index < len(input) {
					want = input[index] * weight[ch*stackWidth+ch%stackWidth]
				}
				if math.Abs(float64(got[row*projectedWidth+ch]-want)) > 2e-5 {
					t.Fatalf("frames=%d row=%d channel=%d got=%g want=%g", frames, row, ch, got[row*projectedWidth+ch], want)
				}
			}
		}
		for i := range input {
			if input[i] != original[i] {
				t.Fatal("mutated caller input")
			}
		}
		if &got[0] == &input[0] {
			t.Fatal("output aliases caller input")
		}
		stable := append([]float32(nil), got...)
		input[0]++
		if _, err := model.Project(input, frames); err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if got[i] != stable[i] {
				t.Fatal("returned output aliases scratch")
			}
		}
	}
	for _, n := range []int{0, 3002} {
		if _, err := model.Project(make([]float32, n*melBins), n); err == nil {
			t.Fatalf("accepted invalid rows %d", n)
		}
	}
	if _, err := model.Project([]float32{float32(math.NaN())}, 1); err == nil {
		t.Fatal("accepted malformed input")
	}
	if _, err := model.Project(make([]float32, melBins-1), 1); err == nil {
		t.Fatal("accepted short input")
	}
	// A distinct typed view may overlap the owned weight storage. Treat it as
	// ordinary caller input; the projection must not mutate either slice.
	alias := model.weight[:melBins]
	original := append([]float32(nil), alias...)
	if _, err := model.Project(alias, 1); err != nil {
		t.Fatal(err)
	}
	for i := range alias {
		if alias[i] != original[i] {
			t.Fatal("mutated aliased weight/input")
		}
	}
}

func TestReleasedStackingPyTorchParity(t *testing.T) {
	model := releasedStacking(t)
	path := filepath.Join("..", "..", "testdata", "jfk.wav")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e" {
		t.Fatal("JFK input provenance changed")
	}
	pcm, rate, err := audio.WAV(path)
	if err != nil || rate != 16000 {
		t.Fatalf("WAV rate=%d err=%v", rate, err)
	}
	features, frames, err := audio.NemotronLogMel(pcm)
	if err != nil || frames != 1101 {
		t.Fatalf("frontend frames=%d err=%v", frames, err)
	}
	// Also qualify the projection independently of Go frontend rounding.
	inputRef := readStackingFixture(t, "testdata/jfk_features_transformers_5_18.f32.gz", 1101*128)
	compareStacking(t, model, inputRef, frames, "independent feature input")
	compareStacking(t, model, features, frames, "composed Go frontend")
}

func readStackingFixture(t testing.TB, path string, elements int) []float32 {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	gz, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	ref, err := io.ReadAll(io.LimitReader(gz, int64(elements*4+1)))
	if err != nil || len(ref) != elements*4 {
		t.Fatalf("reference len=%d err=%v", len(ref), err)
	}
	out := make([]float32, elements)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(ref[i*4:]))
	}
	return out
}

func compareStacking(t *testing.T, model *StackingProjection, features []float32, frames int, label string) {
	t.Helper()
	reference := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*512)
	for _, tc := range []struct {
		name string
		f    func([]float32, int) ([]float32, error)
	}{{"scalar", model.ProjectScalar}, {"SIMD", model.Project}} {
		got, err := tc.f(features, frames)
		if err != nil || len(got) != 138*512 {
			t.Fatalf("%s len=%d err=%v", tc.name, len(got), err)
		}
		var maxAbs, sumAbs float64
		var outside int
		for i, v := range got {
			want := reference[i]
			d := math.Abs(float64(v - want))
			if d > maxAbs {
				maxAbs = d
			}
			sumAbs += d
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 2e-3+2e-4*math.Abs(float64(want)) {
				outside++
			}
		}
		mean := sumAbs / float64(len(got))
		t.Logf("%s %s projected: max_abs=%g mean_abs=%g outside=%d", label, tc.name, maxAbs, mean, outside)
		if outside != 0 || mean > 1e-4 {
			t.Fatalf("%s outside numerical distribution", tc.name)
		}
	}
}

func BenchmarkReleasedStackingScalar(b *testing.B) { benchmarkReleasedStacking(b, false) }
func BenchmarkReleasedStackingSIMD(b *testing.B)   { benchmarkReleasedStacking(b, true) }
func benchmarkReleasedStacking(b *testing.B, vector bool) {
	model := releasedStacking(b)
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		b.Fatal(err)
	}
	features, frames, err := audio.NemotronLogMel(pcm)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if vector {
			_, err = model.Project(features, frames)
		} else {
			_, err = model.ProjectScalar(features, frames)
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}
