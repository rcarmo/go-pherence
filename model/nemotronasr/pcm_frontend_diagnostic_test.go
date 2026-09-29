package nemotronasr

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Diagnostic only: compare the same pinned podcast mel and native PCM mel
// through independent streaming subsampling instances. No acceptance gate is
// changed by this test; the composed PCM path retains its existing failure.
func TestPodcastFrontendPropagationDiagnostic(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_PODCAST_FRONTEND_DIAGNOSTIC") != "1" {
		t.Skip("set GO_PHERENCE_TEST_PODCAST_FRONTEND_DIAGNOSTIC=1")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL")
	}
	reference := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_GENERATION_LONG_REF")
	if reference == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_GENERATION_LONG_REF to pinned podcast20 JSON")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, loadErr := LoadSubsampling(file)
	closeErr := file.Close()
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if !strings.HasSuffix(reference, "podcast20.json") {
		t.Fatal("diagnostic requires the independent podcast20 reference")
	}
	refPath := strings.TrimSuffix(reference, ".json") + ".mel.f32.gz"
	f, err := os.Open(refPath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	data, err := io.ReadAll(gz)
	if err != nil || len(data) != 2000*128*4 {
		t.Fatalf("mel bytes=%d err=%v", len(data), err)
	}
	ref := make([]float32, 2000*128)
	for i := range ref {
		ref[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
	}
	projectedFile, err := os.Open(strings.TrimSuffix(reference, ".json") + ".input.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer projectedFile.Close()
	projectedGzip, err := gzip.NewReader(projectedFile)
	if err != nil {
		t.Fatal(err)
	}
	defer projectedGzip.Close()
	projectedData, err := io.ReadAll(projectedGzip)
	if err != nil || len(projectedData) != 252*1024*4 {
		t.Fatalf("projected reference bytes=%d err=%v", len(projectedData), err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "podcast.wav"))
	if err != nil || rate != 16000 || len(pcm) < 320*rate {
		t.Fatalf("wav rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	pcm = pcm[300*rate : 320*rate]
	var mel audio.NemotronMelStream
	native := make([]float32, 0, len(ref))
	for offset := 0; offset < len(pcm); offset += 80000 {
		end := min(offset+80000, len(pcm))
		part, e := mel.AppendPCM(pcm[offset:end])
		if e != nil {
			t.Fatal(e)
		}
		native = append(native, part...)
	}
	part, err := mel.Finish()
	if err != nil {
		t.Fatal(err)
	}
	native = append(native, part...)
	if len(native) != 2001*128 {
		t.Fatalf("native mel rows=%d", len(native)/128)
	}
	var refStream, nativeStream SubsamplingStream
	refStream.Model = model
	nativeStream.Model = model
	var referenceOutside, composedOutside int
	for chunk, start := 0, 0; start < 2000; chunk++ {
		rows := 32
		if chunk == 0 {
			rows = 25
		}
		end := min(start+rows, 2000)
		refInput := make([]float32, rows*128)
		nativeInput := make([]float32, rows*128)
		copy(refInput, ref[start*128:end*128])
		copy(nativeInput, native[start*128:end*128])
		refOutput, e := refStream.ForwardUnmaskedChunk(refInput, rows)
		if e != nil {
			t.Fatal(e)
		}
		nativeOutput, e := nativeStream.ForwardUnmaskedChunk(nativeInput, rows)
		if e != nil {
			t.Fatal(e)
		}
		for i, value := range refOutput {
			pinned := math.Float32frombits(binary.LittleEndian.Uint32(projectedData[(chunk*4*1024+i)*4:]))
			limit := 3e-3 + 4e-5*math.Abs(float64(pinned))
			if math.IsNaN(float64(value)) || math.Abs(float64(value-pinned)) > limit {
				referenceOutside++
			}
			if math.IsNaN(float64(nativeOutput[i])) || math.Abs(float64(nativeOutput[i]-pinned)) > limit {
				composedOutside++
				if composedOutside <= 4 {
					t.Logf("composed outlier chunk=%d row=%d col=%d native=%g ref-mel=%g pinned=%g", chunk, i/1024, i%1024, nativeOutput[i], value, pinned)
				}
			}
		}
		if chunk == 36 {
			for _, col := range []int{466, 639} {
				i := 1024 + col
				delta := float64(nativeOutput[i] - refOutput[i])
				t.Logf("chunk=%d row=1 col=%d native=%g reference-mel=%g frontend-contribution=%g", chunk, col, nativeOutput[i], refOutput[i], delta)
			}
			var maxMel, meanMel float64
			var maxRow, maxCol int
			for i, v := range nativeInput {
				d := math.Abs(float64(v - refInput[i]))
				meanMel += d
				if d > maxMel {
					maxMel = d
					maxRow = i / 128
					maxCol = i % 128
				}
			}
			t.Logf("chunk36 mel max=%g mean=%g row=%d col=%d", maxMel, meanMel/float64(len(nativeInput)), maxRow, maxCol)
		}
		start = end
	}
	t.Logf("reference-mel outliers=%d composed-PCM outliers=%d (composed result is diagnostic, not qualified)", referenceOutside, composedOutside)
	if referenceOutside != 0 {
		t.Fatal("independently pinned reference-mel subsampling differs")
	}
}
