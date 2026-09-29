package nemotronasr

import (
	"compress/gzip"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
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

func TestReleasedSubsamplingMaskedStreamPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MASKED_SUBSAMPLING_REF")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MASKED_SUBSAMPLING_REF to pinned PyTorch fixture")
	}
	if !strings.HasSuffix(path, ".f32.gz") {
		t.Fatal("reference path must end in .f32.gz")
	}
	read := func(name string, n int) []float32 {
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
		if err != nil || len(data) != n*4 {
			t.Fatalf("reference %s bytes=%d err=%v", name, len(data), err)
		}
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return out
	}
	features := read(strings.TrimSuffix(path, ".f32.gz")+".features.f32.gz", 90*128)
	reference := read(path, 12*1024)
	stream := &SubsamplingStream{Model: releasedSubsampling(t)}
	for _, invalid := range []struct {
		frames, valid int
		features      []float32
	}{
		{26, 27, make([]float32, 26*128)},
		{26, -1, make([]float32, 26*128)},
		{129, 0, make([]float32, 129*128)},
		{26, 25, make([]float32, 25*128)},
		{26, 25, append([]float32{float32(math.NaN())}, make([]float32, 26*128-1)...)},
	} {
		if _, _, err := stream.ForwardMaskedChunk(invalid.features, invalid.frames, invalid.valid); err == nil || stream.started || stream.last[0] != nil {
			t.Fatalf("invalid chunk accepted or mutated state: frames=%d valid=%d", invalid.frames, invalid.valid)
		}
	}
	var maxAbs, sumAbs float64
	var outside int
	start := 0
	for step, chunk := range []struct{ frames, valid, outputValid int }{{26, 25, 3}, {32, 32, 4}, {32, 24, 3}} {
		input := append([]float32(nil), features[start*128:(start+chunk.frames)*128]...)
		// Deliberately poison the finite masked tail: the reference sees
		// zeros, and the native caller must not mutate or cache the poison.
		for i := chunk.valid * 128; i < len(input); i++ {
			input[i] = 17
		}
		copyInput := append([]float32(nil), input...)
		got, valid, err := stream.ForwardMaskedChunk(input, chunk.frames, chunk.valid)
		if err != nil || len(got) != 4*1024 || valid != chunk.outputValid {
			t.Fatalf("chunk %d: rows=%d valid=%d err=%v", step, len(got)/1024, valid, err)
		}
		for i, v := range got {
			want := reference[step*len(got)+i]
			diff := math.Abs(float64(v - want))
			maxAbs = math.Max(maxAbs, diff)
			sumAbs += diff
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || diff > 3e-4+2e-5*math.Abs(float64(want)) {
				outside++
			}
		}
		for i, v := range input {
			if v != copyInput[i] {
				t.Fatalf("chunk %d changed caller input at %d", step, i)
			}
		}
		for row := 0; row < 4; row++ {
			var rowMax float64
			for j := 0; j < 1024; j++ {
				rowMax = math.Max(rowMax, math.Abs(float64(got[row*1024+j]-reference[step*len(got)+row*1024+j])))
			}
			t.Logf("chunk=%d row=%d max_abs=%g", step, row, rowMax)
		}
		start += chunk.frames
		if _, _, err := stream.ForwardMaskedChunk(make([]float32, 31*128), 31, 32); err == nil {
			t.Fatalf("chunk %d accepted invalid valid count", step)
		}
	}
	t.Logf("masked chunks: max_abs=%g mean_abs=%g outside=%d", maxAbs, sumAbs/float64(len(reference)), outside)
	if outside != 0 || sumAbs/float64(len(reference)) > 1e-4 {
		t.Fatal("masked stream subsampling differs from pinned PyTorch")
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		t.Fatalf("JFK WAV rate=%d err=%v", rate, err)
	}
	var scheduler ASRMelChunkStream
	var scheduled []ASRMelChunk
	for pos := 0; pos < 13000; {
		n := 397
		if n > 13000-pos {
			n = 13000 - pos
		}
		part, err := scheduler.AppendPCM(pcm[pos : pos+n])
		if err != nil {
			t.Fatal(err)
		}
		scheduled = append(scheduled, part...)
		pos += n
	}
	terminal, err := scheduler.Finish()
	if err != nil {
		t.Fatal(err)
	}
	scheduled = append(scheduled, terminal...)
	if len(scheduled) != 3 {
		t.Fatalf("scheduled chunks=%d", len(scheduled))
	}
	nativePath := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MASKED_NATIVE_INPUT_REF")
	if nativePath == "" {
		t.Log("PCM-to-subsampling comparison needs GO_PHERENCE_NEMOTRON_ASR_MASKED_NATIVE_INPUT_REF; masked operator parity above is independent")
		return
	}
	var composed SubsamplingStream
	composed.Model = stream.Model
	nativeReference := read(nativePath, 12*1024)
	var featureMax, outputMax, featureMean, outputMean, operatorMax, operatorMean float64
	var featureOutside, outputOutside, operatorOutside int
	start = 0
	for step, chunk := range scheduled {
		for i, value := range chunk.Features {
			want := features[start*128+i]
			delta := math.Abs(float64(value - want))
			featureMax = math.Max(featureMax, delta)
			featureMean += delta
			if delta > 5e-4+1e-5*math.Abs(float64(want)) {
				featureOutside++
			}
		}
		got, valid, err := composed.ForwardMaskedChunk(chunk.Features, chunk.Frames, chunk.Valid)
		if err != nil || len(got) != 4*1024 || valid != []int{3, 4, 3}[step] {
			t.Fatalf("scheduled chunk %d valid=%d output=%d err=%v", step, valid, len(got), err)
		}
		for i, value := range got {
			want := reference[step*len(got)+i]
			delta := math.Abs(float64(value - want))
			outputMax = math.Max(outputMax, delta)
			outputMean += delta
			if delta > 8e-4+3e-5*math.Abs(float64(want)) {
				outputOutside++
			}
			operatorWant := nativeReference[step*len(got)+i]
			operatorDiff := math.Abs(float64(value - operatorWant))
			operatorMax = math.Max(operatorMax, operatorDiff)
			operatorMean += operatorDiff
			if operatorDiff > 3e-4+2e-5*math.Abs(float64(operatorWant)) {
				operatorOutside++
			}
		}
		start += chunk.Frames
	}
	t.Logf("PCM scheduled features max=%g mean=%g outside=%d; end-to-end subsampling max=%g mean=%g outside=%d; operator on native features max=%g mean=%g outside=%d", featureMax, featureMean/float64(len(features)), featureOutside, outputMax, outputMean/float64(len(reference)), outputOutside, operatorMax, operatorMean/float64(len(reference)), operatorOutside)
	if featureOutside != 0 || operatorOutside != 0 {
		t.Fatal("PCM feature or native-input subsampling operator differs from pinned PyTorch")
	}
	if outputOutside != 0 {
		t.Logf("PCM-to-subsampling gate NOT qualified: %d/12288 outside 8e-4+3e-5*abs(reference); frontend rounding amplified; do not infer end-to-end parity", outputOutside)
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
