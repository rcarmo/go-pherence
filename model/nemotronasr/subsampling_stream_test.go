package nemotronasr

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestSubsamplingStreamRejectsMalformed(t *testing.T) {
	if _, err := (*SubsamplingStream)(nil).ForwardChunk(make([]float32, 32*128), 32); err == nil {
		t.Fatal("accepted nil stream")
	}
	s := &SubsamplingStream{}
	if _, err := s.ForwardChunk(make([]float32, 32*128), 32); err == nil {
		t.Fatal("accepted nil model")
	}
	model := releasedSubsampling(t)
	s.Model = model
	for _, input := range []struct {
		features []float32
		frames   int
	}{
		{nil, 0}, {make([]float32, 31*128), 31}, {make([]float32, 129*128), 129},
		{append([]float32{float32(math.NaN())}, make([]float32, 32*128-1)...), 32},
	} {
		if _, err := s.ForwardChunk(input.features, input.frames); err == nil || s.started || s.last[0] != nil {
			t.Fatalf("invalid chunk accepted or mutated cache: rows=%d", input.frames)
		}
	}
}

func TestReleasedSubsamplingStreamPyTorchParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_NEMOTRON_ASR_STREAM_SUBSAMPLING_REF") == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_STREAM_SUBSAMPLING_REF to independent PyTorch fixture")
	}
	model := releasedSubsampling(t)
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_STREAM_SUBSAMPLING_REF")
	if !strings.HasSuffix(path, ".f32.gz") {
		t.Fatal("reference path must end in .f32.gz")
	}
	read := func(name string, expected int) []float32 {
		t.Helper()
		file, err := os.Open(name)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		gz, err := gzip.NewReader(file)
		if err != nil {
			t.Fatal(err)
		}
		defer gz.Close()
		data, err := io.ReadAll(gz)
		if err != nil || len(data) != expected*4 {
			t.Fatalf("reference %s bytes=%d err=%v", name, len(data), err)
		}
		out := make([]float32, expected)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return out
	}
	features := read(path[:len(path)-len(".f32.gz")]+".features.f32.gz", 128*128)
	reference := read(path, 16*1024)
	chunk := 32
	if setting := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_STREAM_CHUNK"); setting != "" {
		var err error
		chunk, err = strconv.Atoi(setting)
		if err != nil || chunk < 8 || chunk > 128 || chunk%8 != 0 || 128%chunk != 0 {
			t.Fatal("invalid reference chunk size")
		}
	}
	stream := &SubsamplingStream{Model: model}
	var maxAbs, sumAbs float64
	var outside int
	for step := 0; step < 128/chunk; step++ {
		got, err := stream.ForwardChunk(features[step*chunk*128:(step+1)*chunk*128], chunk)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != chunk/8*1024 {
			t.Fatalf("step %d rows=%d", step, len(got)/1024)
		}
		var chunkMax, chunkMean float64
		var chunkOutside int
		for i, v := range got {
			want := reference[step*len(got)+i]
			diff := math.Abs(float64(v - want))
			maxAbs = math.Max(maxAbs, diff)
			sumAbs += diff
			chunkMax = math.Max(chunkMax, diff)
			chunkMean += diff
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || diff > 3e-4+2e-5*math.Abs(float64(want)) {
				outside++
				chunkOutside++
			}
		}
		chunkMean /= float64(len(got))
		t.Logf("step=%d max_abs=%g mean_abs=%g outside=%d", step, chunkMax, chunkMean, chunkOutside)
		if chunkOutside != 0 || chunkMean > 1e-4 {
			t.Fatalf("stream subsampling step %d differs from PyTorch", step)
		}
	}
	t.Logf("128 mel rows max_abs=%g mean_abs=%g outside=%d", maxAbs, sumAbs/float64(len(reference)), outside)
	if !stream.started || outside != 0 {
		t.Fatal("invalid streaming subsampling state")
	}
}
