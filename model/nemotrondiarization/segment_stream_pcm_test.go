package nemotrondiarization

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedPCMStreamingSegmentsPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	segmentsPath := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_STREAM_SEGMENTS_REF")
	if path == "" || segmentsPath == "" {
		t.Skip("set pinned model and GO_PHERENCE_NEMOTRON_DIARIZATION_STREAM_SEGMENTS_REF")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	request, loadErr := LoadPCMStreamingRequest(file)
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if loadErr != nil {
		t.Fatal(loadErr)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK input rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	expectedRows := 1099
	if duration := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_PCM_STREAM_SECONDS"); duration != "" {
		if duration != "100" {
			t.Fatal("segment reference supports only 11 or 100 seconds")
		}
		original := pcm
		pcm = make([]float32, 16000*100)
		for i := range pcm {
			pcm[i] = original[i%len(original)]
		}
		expectedRows = 9999
	}
	logitsPath := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_PCM_STREAM_REF")
	if logitsPath == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_PCM_STREAM_REF to matching streaming logits fixture")
	}
	f, err := os.Open(logitsPath)
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
	if err != nil || len(data) != expectedRows*diarizationSpeakers*4 {
		t.Fatalf("reference logits bytes=%d err=%v", len(data), err)
	}
	var spans SegmentStream
	var got []Segment
	var compared, outside int
	var maxAbs, sumAbs float64
	consume := func(logits []float32) {
		t.Helper()
		if compared+len(logits) > len(data)/4 {
			t.Fatal("more logits than pinned reference")
		}
		for i, value := range logits {
			want := math.Float32frombits(binary.LittleEndian.Uint32(data[4*(compared+i):]))
			delta := math.Abs(float64(value - want))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		compared += len(logits)
		part, err := spans.Append(logits)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, part...)
	}
	chunkSamples := 7979
	if expectedRows == 9999 {
		chunkSamples = 16000 * 5
	}
	for offset := 0; offset < len(pcm); {
		end := offset + chunkSamples
		if end > len(pcm) {
			end = len(pcm)
		}
		logits, err := request.AppendPCM(pcm[offset:end])
		if err != nil {
			t.Fatal(err)
		}
		if len(logits) > 0 {
			consume(logits)
		}
		offset = end
	}
	logits, err := request.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if len(logits) > 0 {
		consume(logits)
	}
	part, err := spans.Finish()
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, part...)
	sortSegments(got) // offline JSON is sorted globally by start time
	segmentsData, err := os.ReadFile(segmentsPath)
	if err != nil {
		t.Fatal(err)
	}
	var want []Segment
	if err := json.Unmarshal(segmentsData, &want); err != nil {
		t.Fatal(err)
	}
	t.Logf("streamed rows=%d logits max_abs=%g mean_abs=%g outside=%d segments=%v", spans.frames, maxAbs, sumAbs/float64(compared), outside, got)
	if spans.frames != int64(expectedRows) || compared != len(data)/4 || outside != 0 || sumAbs/float64(compared) > 1e-5 || !reflect.DeepEqual(got, want) {
		t.Fatalf("streamed rows=%d segments=%v want=%v", spans.frames, got, want)
	}
}
