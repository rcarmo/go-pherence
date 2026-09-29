package nemotronasr

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestReleasedPCMGenerationJFKPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_GENERATION_LONG_REF")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_GENERATION_LONG_REF to pinned generate JSON")
	}
	model := releasedPCMGenerationModel(t)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var reference struct {
		Samples   int    `json:"samples"`
		MelValid  int    `json:"mel_valid"`
		Tokens    []int  `json:"tokens"`
		Durations []int  `json:"durations"`
		Text      string `json:"text"`
		Nonblank  int    `json:"nonblank"`
	}
	if err := json.Unmarshal(data, &reference); err != nil {
		t.Fatal(err)
	}
	if reference.MelValid < 1 || reference.MelValid > 10000 {
		t.Fatal("invalid pinned mel frame count")
	}
	encoderRows := 1 + (reference.MelValid-25+31)/32
	encoderRows *= 4
	readStage := func(name string) []float32 {
		t.Helper()
		f, err := os.Open(strings.TrimSuffix(path, ".json") + "." + name + ".f32.gz")
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
		if err != nil || len(data) != encoderRows*1024*4 {
			t.Fatalf("%s reference bytes=%d err=%v", name, len(data), err)
		}
		values := make([]float32, encoderRows*1024)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		}
		return values
	}
	inputRef := readStage("input")
	towerRef := readStage("tower")
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("unexpected JFK WAV rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	if reference.Samples == 16000*100 {
		original := pcm
		pcm = make([]float32, reference.Samples)
		for i := range pcm {
			pcm[i] = original[i%len(original)]
		}
	} else if reference.Samples == 16000*20 {
		pcm, rate, err = audio.WAV(filepath.Join("..", "..", "testdata", "podcast.wav"))
		if err != nil || rate != 16000 || len(pcm) < 320*rate {
			t.Fatalf("unexpected podcast WAV rate=%d samples=%d err=%v", rate, len(pcm), err)
		}
		pcm = pcm[300*rate : 320*rate]
	}
	if len(pcm) != reference.Samples || len(reference.Tokens) != len(reference.Durations) || len(reference.Tokens) == 0 || reference.Tokens[0] != rnntBlank {
		t.Fatalf("unexpected reference geometry: PCM=%d rate=%d tokens=%d durations=%d err=%v", len(pcm), rate, len(reference.Tokens), len(reference.Durations), err)
	}
	vocab, err := tokenizer.Load(filepath.Join("..", "..", "checkpoints", "nemotron", "asr", "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	var expectedFrames []int64
	var frame int64
	for _, duration := range reference.Durations[1:] { // first token is BOS blank
		expectedFrames = append(expectedFrames, frame)
		frame += int64(duration)
	}
	chunkSizes := []int{397, 4040, 5520}
	if reference.Samples != 176000 {
		chunkSizes = []int{80000}
	}
	if selected := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_GENERATION_PCM_CHUNK_SIZE"); selected != "" {
		size, err := strconv.Atoi(selected)
		if err != nil || size != 397 && size != 4040 && size != 5520 && size != 80000 {
			t.Fatal("invalid reference PCM chunk size")
		}
		chunkSizes = []int{size}
	}
	for _, chunkSize := range chunkSizes {
		s := &PCMGenerationStream{Model: model}
		chunkIndex := 0
		var maxInput, sumInput, maxTower, sumTower float64
		var outsideInput, outsideTower int
		s.onStage = func(stage string, values []float32) {
			var want []float32
			switch stage {
			case "subsampling":
				want = inputRef[chunkIndex*4*1024 : (chunkIndex+1)*4*1024]
			case "tower":
				want = towerRef[chunkIndex*4*1024 : (chunkIndex+1)*4*1024]
				chunkIndex++
			default:
				t.Fatalf("unexpected stage %s", stage)
			}
			for i, v := range values {
				d := math.Abs(float64(v - want[i]))
				if stage == "subsampling" {
					maxInput = math.Max(maxInput, d)
					sumInput += d
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-3+4e-5*math.Abs(float64(want[i])) {
						outsideInput++
						if outsideInput <= 4 {
							t.Logf("subsampling outlier chunk=%d row=%d col=%d native=%g ref=%g delta=%g", chunkIndex, i/1024, i%1024, v, want[i], d)
						}
					}
				} else {
					maxTower = math.Max(maxTower, d)
					sumTower += d
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-4+2e-5*math.Abs(float64(want[i])) {
						outsideTower++
					}
				}
			}
		}
		var decisions []int
		var frames []int64
		for offset := 0; offset < len(pcm); {
			end := offset + chunkSize
			if end > len(pcm) {
				end = len(pcm)
			}
			input := append([]float32(nil), pcm[offset:end]...)
			before := append([]float32(nil), input...)
			tokens, positions, err := s.AppendPCM(context.Background(), input)
			if err != nil || !reflect.DeepEqual(input, before) {
				t.Fatalf("chunk=%d at %d err=%v or caller input mutated", chunkSize, offset, err)
			}
			decisions = append(decisions, tokens...)
			frames = append(frames, positions...)
			offset = end
		}
		last, positions, err := s.Finish(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		decisions = append(decisions, last...)
		frames = append(frames, positions...)
		t.Logf("PCM JFK chunk=%d input max=%g mean=%g outside=%d tower max=%g mean=%g outside=%d", chunkSize, maxInput, sumInput/float64(encoderRows*1024), outsideInput, maxTower, sumTower/float64(encoderRows*1024), outsideTower)
		if chunkIndex != encoderRows/4 || outsideTower != 0 {
			t.Fatal("PCM-to-encoder tower parity failed")
		}
		if outsideInput != 0 {
			if reference.Samples != 16000*20 || outsideInput != 2 {
				t.Fatalf("unexpected PCM-to-subsampling outliers: %d", outsideInput)
			}
			t.Logf("PCM-to-subsampling stage UNQUALIFIED: %d podcast values outside provisional gate; retained for downstream decision analysis", outsideInput)
		}
		if len(decisions) != len(reference.Tokens)-1 || !reflect.DeepEqual(frames, expectedFrames) {
			t.Fatalf("chunk=%d decisions=%d want=%d frame parity=%v", chunkSize, len(decisions), len(reference.Tokens)-1, reflect.DeepEqual(frames, expectedFrames))
		}
		var mismatches, nonblank int
		for i, token := range decisions {
			if token != reference.Tokens[i+1] {
				mismatches++
				if mismatches < 6 {
					t.Logf("chunk=%d decision=%d frame=%d native=%d ref=%d", chunkSize, i, frames[i], token, reference.Tokens[i+1])
				}
			}
			if token != rnntBlank {
				nonblank++
			}
		}
		text, err := DecodeRNNTText(vocab, decisions)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("PCM JFK chunk=%d decisions=%d nonblank=%d mismatches=%d text=%q ref=%q", chunkSize, len(decisions), nonblank, mismatches, text, reference.Text)
		if mismatches != 0 || nonblank != reference.Nonblank || text != reference.Text || s.greedy.frames != int64(encoderRows) {
			t.Fatal("native JFK streaming transcription differs from pinned generate")
		}
	}
}
