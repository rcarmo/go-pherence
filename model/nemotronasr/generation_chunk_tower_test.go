package nemotronasr

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

// Short, independently generated lookahead-3 streaming generation encoder
// oracle. Generation consumes 25/32/32 mel frames without attention masks.
func TestReleasedGenerationChunkTowerPyTorchParity(t *testing.T) {
	refDir := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_GENERATION_CHUNKS_REF")
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if refDir == "" || path == "" {
		t.Skip("set pinned model and GO_PHERENCE_NEMOTRON_ASR_GENERATION_CHUNKS_REF")
	}
	read := func(name string, count int) []float32 {
		t.Helper()
		f, err := os.Open(filepath.Join(refDir, name+".f32.gz"))
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
		if err != nil || len(data) != count*4 {
			t.Fatalf("reference %s bytes=%d err=%v", name, len(data), err)
		}
		values := make([]float32, count)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return values
	}
	f, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	subsampling, err := LoadSubsampling(f)
	if err != nil {
		t.Fatal(err)
	}
	tower, err := LoadOfflineEncoderTower(f)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := LoadRNNTProjection(f)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(f)
	if err != nil {
		t.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		t.Fatalf("JFK WAV rate=%d err=%v", rate, err)
	}
	var frontend ASRMelChunkStream
	var chunks []ASRMelChunk
	for pos := 0; pos < 13000; {
		n := 397
		if n > 13000-pos {
			n = 13000 - pos
		}
		part, err := frontend.AppendPCM(pcm[pos : pos+n])
		if err != nil {
			t.Fatal(err)
		}
		chunks = append(chunks, part...)
		pos += n
	}
	last, err := frontend.Finish()
	if err != nil {
		t.Fatal(err)
	}
	chunks = append(chunks, last...)
	if len(chunks) != 3 {
		t.Fatalf("PCM produced %d chunks", len(chunks))
	}
	for _, lookahead := range []int{3} {
		inputRef := read("input", 12*1024)
		towerRef := read("tower", 12*1024)
		layer0Ref := read("layer0", 12*1024)
		rnntRef := read("rnnt", 12*640)
		sub := &SubsamplingStream{Model: subsampling}
		cached := &CachedEncoderTower{Tower: tower}
		exact := &CachedEncoderTower{Tower: tower}
		var layer0State Encoder0ChunkState
		greedy := &GreedyRNNTStream{Model: &GreedyRNNT{Decoder: decoder, Projection: projection}}
		var decisions, emittedFrames []int64
		for step, chunk := range chunks {
			if step == 0 {
				if _, err := sub.ForwardUnmaskedChunk(make([]float32, 26*128), 26); err == nil || sub.started {
					t.Fatal("accepted first generation chunk with wrong frame count")
				}
			} else if _, err := sub.ForwardUnmaskedChunk(make([]float32, 25*128), 25); err == nil || sub.last[0] == nil {
				t.Fatal("accepted subsequent generation chunk with wrong frame count")
			}
			features := chunk.Features
			frames := chunk.Frames
			if step == 0 {
				features = features[:25*128] // generate requires exactly 25 first rows
				frames = 25
			}
			input, err := sub.ForwardUnmaskedChunk(features, frames)
			if err != nil {
				t.Fatal(err)
			}
			// The released config sets scale_input=false (factor 1).
			if _, _, err := sub.ForwardMaskedChunk(features, frames, frames); err == nil || sub.mode != 2 {
				t.Fatal("mixed generation and masked subsampling cache modes")
			}
			comparePCMChunkStage(t, "input", lookahead, step, input, inputRef[step*4*1024:(step+1)*4*1024], 3e-3, 4e-5)
			got, err := cached.ForwardChunk(input, 4, lookahead)
			if err != nil {
				t.Fatalf("tower look=%d step=%d: %v", lookahead, step, err)
			}
			towerWant := towerRef[step*4*1024 : (step+1)*4*1024]
			layer0, layerErr := tower.layers[0].ForwardCachedChunk(input, 4, lookahead, &layer0State)
			if layerErr != nil {
				t.Fatal(layerErr)
			}
			comparePCMChunkStage(t, "layer0", lookahead, step, layer0, layer0Ref[step*4*1024:(step+1)*4*1024], 3e-4, 2e-5)
			comparePCMChunkStage(t, "tower", lookahead, step, got, towerWant, 3e-4, 2e-5)
			exactInput := inputRef[step*4*1024 : (step+1)*4*1024]
			exactGot, err := exact.ForwardChunk(exactInput, 4, lookahead)
			if err != nil {
				t.Fatal(err)
			}
			comparePCMChunkStage(t, "tower-exact-input", lookahead, step, exactGot, towerWant, 3e-4, 2e-5)
			_, _, encoded, err := projection.Project(got, 4, 101)
			if err != nil {
				t.Fatal(err)
			}
			comparePCMChunkStage(t, "rnnt", lookahead, step, encoded, rnntRef[step*4*640:(step+1)*4*640], 8e-4, 2e-5)
			tokens, positions, err := greedy.Append(encoded, 4)
			if err != nil {
				t.Fatal(err)
			}
			for i, token := range tokens {
				decisions = append(decisions, int64(token))
				emittedFrames = append(emittedFrames, positions[i])
			}
		}
		if err := greedy.Finish(); err != nil {
			t.Fatal(err)
		}
		// The pinned reference's generate() returns 13 blanks: its initial
		// BOS blank followed by these twelve frame decisions.
		if len(decisions) != 12 || len(emittedFrames) != 12 {
			t.Fatalf("PCM decisions=%v frames=%v", decisions, emittedFrames)
		}
		for i, token := range decisions {
			if token != rnntBlank || emittedFrames[i] != int64(i) {
				t.Fatalf("PCM frame=%d token=%d at=%d", i, token, emittedFrames[i])
			}
		}
		t.Logf("PCM 13k generated decisions=%v frames=%v", decisions, emittedFrames)
	}
}

func comparePCMChunkStage(t *testing.T, stage string, lookahead, step int, got, want []float32, absolute, relative float64) {
	t.Helper()
	var maxAbs, sumAbs float64
	var outside int
	for i, v := range got {
		diff := math.Abs(float64(v - want[i]))
		maxAbs = math.Max(maxAbs, diff)
		sumAbs += diff
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || diff > absolute+relative*math.Abs(float64(want[i])) {
			outside++
		}
	}
	t.Logf("%s look=%d step=%d max=%g mean=%g outside=%d", stage, lookahead, step, maxAbs, sumAbs/float64(len(got)), outside)
	if outside != 0 {
		t.Fatalf("%s look=%d step=%d differs from pinned PyTorch", stage, lookahead, step)
	}
}
