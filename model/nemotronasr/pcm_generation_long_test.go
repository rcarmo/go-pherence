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
	"time"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
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
	var melRef []float32
	if reference.Samples == 16000*20 {
		melFile, err := os.Open(strings.TrimSuffix(path, ".json") + ".mel.f32.gz")
		if err != nil {
			t.Fatal(err)
		}
		defer melFile.Close()
		gz, err := gzip.NewReader(melFile)
		if err != nil {
			t.Fatal(err)
		}
		defer gz.Close()
		data, err := io.ReadAll(gz)
		if err != nil || len(data) != reference.MelValid*128*4 {
			t.Fatalf("mel reference bytes=%d err=%v", len(data), err)
		}
		melRef = make([]float32, reference.MelValid*128)
		for i := range melRef {
			melRef[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
	}
	if len(melRef) != 0 {
		var sub SubsamplingStream
		sub.Model = model.Subsampling
		var maxAbs, sumAbs float64
		var outside int
		for chunk, start := 0, 0; start < reference.MelValid; chunk++ {
			rows := 32
			if chunk == 0 {
				rows = 25
			}
			features := make([]float32, rows*128)
			end := min(start+rows, reference.MelValid)
			copy(features, melRef[start*128:end*128])
			projected, err := sub.ForwardUnmaskedChunk(features, rows)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range projected {
				want := inputRef[(chunk*4*1024)+i]
				d := math.Abs(float64(v - want))
				maxAbs = math.Max(maxAbs, d)
				sumAbs += d
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 3e-3+4e-5*math.Abs(float64(want)) {
					outside++
					if outside <= 4 {
						t.Logf("reference-mel subsampling outlier chunk=%d row=%d col=%d native=%g ref=%g delta=%g", chunk, i/1024, i%1024, v, want, d)
					}
				}
			}
			start = end
		}
		t.Logf("reference-mel subsampling max=%g mean=%g outside=%d", maxAbs, sumAbs/float64(len(inputRef)), outside)
		if outside != 0 {
			t.Fatal("reference-mel subsampling operator parity failed")
		}
	}
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
	backend := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_GENERATION_PROJECTOR")
	if backend != "" && backend != "ptx" && backend != "vulkan" {
		t.Fatalf("unsupported projection backend %q", backend)
	}
	for _, chunkSize := range chunkSizes {
		s := &PCMGenerationStream{Model: model}
		var projector *DeviceSubsamplingProjector
		if backend != "" {
			projector = &DeviceSubsamplingProjector{Backend: backend}
			defer projector.Close() // also release resources on an early test failure
			s.Projector = projector
		}
		if backend == "ptx" {
			previous := ptx.SetStatsEnabled(true)
			defer ptx.SetStatsEnabled(previous)
		}
		ptxBefore := ptx.StatsSnapshot()
		vkBefore := vulkan.VulkanMemoryStats()
		requestStart := time.Now()
		chunkIndex := 0
		var maxInput, sumInput, maxTower, sumTower, maxMel, sumMel float64
		var outsideInput, outsideTower, outsideMel, melRow int
		s.onStage = func(stage string, values []float32) {
			if stage == "mel" {
				if len(melRef) == 0 {
					return
				}
				for i, v := range values {
					row := melRow + i/128
					if row >= reference.MelValid {
						break
					}
					want := melRef[row*128+i%128]
					d := math.Abs(float64(v - want))
					maxMel = math.Max(maxMel, d)
					sumMel += d
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || d > 5e-4+1e-5*math.Abs(float64(want)) {
						outsideMel++
						if outsideMel <= 4 {
							t.Logf("mel outlier row=%d col=%d native=%g ref=%g delta=%g", row, i%128, v, want, d)
						}
					}
				}
				melRow += len(values) / 128
				return
			}
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
		if projector != nil {
			if projector.dispatches != encoderRows/4 {
				t.Fatalf("backend=%s dispatched %d of %d chunks", backend, projector.dispatches, encoderRows/4)
			}
			if closeErr := projector.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}
			if projector.ready || projector.ptxX != nil || projector.ptxW != nil || projector.ptxY != nil || projector.vkArena != nil || projector.vkOp != nil {
				t.Fatal("request GPU resources retained after close")
			}
			if backend == "ptx" {
				after := ptx.StatsSnapshot()
				if after.KernelLaunches-ptxBefore.KernelLaunches < uint64(projector.dispatches) || after.HostToDevice-ptxBefore.HostToDevice < uint64(projector.dispatches+1) || after.DeviceToHost-ptxBefore.DeviceToHost < uint64(projector.dispatches) || after.Mallocs-ptxBefore.Mallocs != after.Frees-ptxBefore.Frees {
					t.Fatalf("PTX dispatch/transfer/resource accounting before=%+v after=%+v", ptxBefore, after)
				}
			} else {
				after := vulkan.VulkanMemoryStats()
				if after.Bytes != vkBefore.Bytes || after.Allocations != vkBefore.Allocations || after.InFlight || after.Uncertain {
					t.Fatalf("Vulkan resource accounting before=%+v after=%+v", vkBefore, after)
				}
			}
		}
		if err != nil {
			t.Fatal(err)
		}
		decisions = append(decisions, last...)
		frames = append(frames, positions...)
		if len(melRef) > 0 {
			t.Logf("mel valid=%d observed=%d max=%g mean=%g outside=%d", reference.MelValid, melRow, maxMel, sumMel/float64(len(melRef)), outsideMel)
			if melRow != 25+(encoderRows/4-1)*32 || outsideMel != 0 {
				t.Fatal("PCM-to-mel stage parity failed")
			}
		}
		t.Logf("PCM JFK chunk=%d input max=%g mean=%g outside=%d tower max=%g mean=%g outside=%d", chunkSize, maxInput, sumInput/float64(encoderRows*1024), outsideInput, maxTower, sumTower/float64(encoderRows*1024), outsideTower)
		if chunkIndex != encoderRows/4 || outsideTower != 0 {
			t.Fatal("PCM-to-encoder tower parity failed")
		}
		if outsideInput != 0 {
			if reference.Samples != 16000*20 {
				t.Fatalf("PCM-to-subsampling stage parity failed: %d outliers", outsideInput)
			}
			// The podcast's composed stage is explicitly unqualified. Do not
			// treat a fixed number of outliers as an allowance or a passing gate.
			t.Logf("PCM-to-subsampling stage UNQUALIFIED: %d values outside provisional gate; retained for downstream decision analysis", outsideInput)
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
		t.Logf("PCM JFK backend=%q chunk=%d request_elapsed=%s decisions=%d nonblank=%d mismatches=%d text=%q ref=%q", backend, chunkSize, time.Since(requestStart), len(decisions), nonblank, mismatches, text, reference.Text)
		if mismatches != 0 || nonblank != reference.Nonblank || text != reference.Text || s.greedy.frames != int64(encoderRows) {
			t.Fatal("native JFK streaming transcription differs from pinned generate")
		}
	}
}
