package nemotrondiarization

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestPCMStreamingRequestRejectsMalformed(t *testing.T) {
	if _, err := LoadPCMStreamingRequest(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*PCMStreamingRequest)(nil).AppendPCM([]float32{1}); err == nil {
		t.Fatal("accepted nil request")
	}
	if _, err := (*PCMStreamingRequest)(nil).Finish(); err == nil {
		t.Fatal("accepted nil finish")
	}
	s := &PCMStreamingRequest{frontend: &PCMStackingStream{}, window: &StreamingWindow{}}
	for _, input := range [][]float32{nil, make([]float32, 80001), {1, float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := s.AppendPCM(input); err == nil || s.samples != 0 || s.closed {
			t.Fatalf("accepted invalid chunk of %d samples or mutated state", len(input))
		}
	}
	if _, err := s.Finish(); err == nil || s.closed {
		t.Fatal("accepted empty request or mutated state")
	}
}

func TestReleasedPCMStreamingRequestPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	fixture := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_PCM_STREAM_REF")
	if path == "" || fixture == "" {
		t.Skip("set model and GO_PHERENCE_NEMOTRON_DIARIZATION_PCM_STREAM_REF to independently generated reference")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	s, err := LoadPCMStreamingRequest(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK PCM rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	switch duration := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_PCM_STREAM_SECONDS"); duration {
	case "100":
		original := pcm
		pcm = make([]float32, 16000*100)
		for i := range pcm {
			pcm[i] = original[i%len(original)]
		}
	case "podcast20":
		pcm, rate, err = audio.WAV(filepath.Join("..", "..", "testdata", "podcast.wav"))
		if err != nil || rate != 16000 || len(pcm) < 320*rate {
			t.Fatalf("podcast PCM rate=%d samples=%d err=%v", rate, len(pcm), err)
		}
		pcm = pcm[300*rate : 320*rate]
	case "160", "200", "11520", "16639", "16640", "16680", "17040", "28000":
		n, err := strconv.Atoi(duration)
		if err != nil {
			t.Fatal(err)
		}
		pcm = pcm[:n]
	}
	f, err := os.Open(fixture)
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
	if err != nil || len(data)%4 != 0 {
		t.Fatalf("reference read bytes=%d err=%v", len(data), err)
	}
	var count, outside int
	var maxAbs, sumAbs float64

	compare := func(got []float32) {
		for i, value := range got {
			if (count+i)*4+4 > len(data) {
				break
			}
			want := math.Float32frombits(binary.LittleEndian.Uint32(data[(count+i)*4:]))
			delta := math.Abs(float64(value - want))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		count += len(got)
	}
	for offset := 0; offset < len(pcm); {
		end := min(offset+16000*5, len(pcm))
		got, err := s.AppendPCM(pcm[offset:end])
		if err != nil {
			t.Fatal(err)
		}
		if len(s.pcmTail) > pcmTailReserve || len(s.pending)/projectedWidth >= lowLatencyFrames+lowLatencyLookahead {
			t.Fatalf("unbounded streaming state tail=%d pending=%d", len(s.pcmTail), len(s.pending)/projectedWidth)
		}
		compare(got)
		offset = end
	}
	got, err := s.Finish()
	if err != nil {
		t.Fatal(err)
	}
	compare(got)
	t.Logf("rows=%d ref_rows=%d max_abs=%g mean_abs=%g outside=%d", count/diarizationSpeakers, len(data)/4/diarizationSpeakers, maxAbs, sumAbs/float64(count), outside)
	if count*4 != len(data) || outside != 0 || sumAbs/float64(count) > 1e-5 {
		t.Fatal("PCM streaming logits differ from PyTorch")
	}
	if _, err := s.AppendPCM([]float32{0}); err == nil {
		t.Fatal("accepted append after finish")
	}
	if _, err := s.Finish(); err == nil {
		t.Fatal("accepted second finish")
	}
}
