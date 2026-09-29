package nemotrondiarization

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedPCMStackingStreamHundredSecondPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := LoadPCMStackingStream(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	refFile, err := os.Open("testdata/jfk_loop100_stacking_transformers_5_18.f32.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer refFile.Close()
	gz, err := gzip.NewReader(refFile)
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	var maxAbs, sumAbs float64
	var outside, total int
	var first []float32
	var firstValue float32
	compare := func(got []float32) {
		if len(got) == 0 {
			return
		}
		if first == nil {
			first, firstValue = got, got[0]
		}
		bytes := make([]byte, len(got)*4)
		if _, err := io.ReadFull(gz, bytes); err != nil {
			t.Fatal(err)
		}
		for i, value := range got {
			want := math.Float32frombits(binary.LittleEndian.Uint32(bytes[i*4:]))
			delta := math.Abs(float64(value - want))
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		total += len(got)
	}
	for part := 0; part < 20; part++ {
		chunk := make([]float32, 80000)
		for j := range chunk {
			chunk[j] = pcm[(part*len(chunk)+j)%len(pcm)]
		}
		out, err := stream.AppendPCM(chunk)
		if err != nil {
			t.Fatal(err)
		}
		compare(out)
	}
	last, err := stream.Finish()
	if err != nil {
		t.Fatal(err)
	}
	compare(last)
	var extra [1]byte
	if n, err := gz.Read(extra[:]); err != io.EOF || n != 0 {
		t.Fatalf("reference trailing data n=%d err=%v", n, err)
	}
	mean := sumAbs / float64(total)
	t.Logf("100s rows=%d max_abs=%g mean_abs=%g outside=%d", total/projectedWidth, maxAbs, mean, outside)
	if total != 1251*projectedWidth || first[0] != firstValue || outside != 0 || mean > 2e-5 {
		t.Fatal("PCM streaming stack differs from PyTorch")
	}
	if _, err := stream.AppendPCM([]float32{0}); err == nil {
		t.Fatal("accepted append after finish")
	}
}

func TestPCMStackingStreamRejectsMalformed(t *testing.T) {
	if _, err := LoadPCMStackingStream(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*PCMStackingStream)(nil).AppendPCM([]float32{0}); err == nil {
		t.Fatal("accepted nil stream")
	}
	if _, err := (*PCMStackingStream)(nil).Finish(); err == nil {
		t.Fatal("accepted nil finish")
	}
	bad := &PCMStackingStream{stack: StackingStream{Projection: &StackingProjection{weight: make([]float32, projectedWidth*stackWidth)}}}
	if _, err := bad.AppendPCM([]float32{1, 2, float32(math.NaN())}); err == nil {
		t.Fatal("accepted nonfinite PCM")
	}
	if _, err := bad.mel.Finish(); err == nil {
		t.Fatal("rejected PCM changed mel stream")
	}
	if bad.stack.frames != 0 {
		t.Fatal("rejected PCM changed stacking state")
	}
	if _, err := bad.AppendPCM(make([]float32, 16000*5+1)); err == nil {
		t.Fatal("accepted oversized PCM")
	}
	if _, err := bad.mel.Finish(); err == nil {
		t.Fatal("oversized PCM changed mel stream")
	}
}
