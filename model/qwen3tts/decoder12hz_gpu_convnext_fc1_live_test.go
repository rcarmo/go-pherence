package qwen3tts

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// liveConvNeXtFC1GPU owns prepared buffers for exactly one diagnostic request.
// Its two 512-wide SGEMMs preserve Candle's partial-sum order.
type liveConvNeXtFC1GPU struct {
	a, part          [2][]float32
	devA, devB, devC [2]*nvidia.Buffer
}

func newLiveConvNeXtFC1GPU(t *testing.T, l talkerLinear) *liveConvNeXtFC1GPU {
	t.Helper()
	const rows, in, out, block = 256, 1024, 4096, 512
	if l.inDim != in || l.outDim != out || len(l.weight) != in*out || len(l.bias) != out {
		t.Fatal("unsupported live ConvNeXt FC1 geometry")
	}
	s := new(liveConvNeXtFC1GPU)
	t.Cleanup(s.close)
	for g := range s.a {
		var err error
		s.devA[g], err = nvidia.Malloc(rows * block)
		if err != nil {
			t.Fatal(err)
		}
		s.devB[g], err = nvidia.Malloc(block * out)
		if err != nil {
			t.Fatal(err)
		}
		s.devC[g], err = nvidia.Malloc(rows * out)
		if err != nil {
			t.Fatal(err)
		}
		weights := make([]float32, block*out)
		for o := 0; o < out; o++ {
			for k := 0; k < block; k++ {
				weights[k*out+o] = l.weight[o*in+g*block+k]
			}
		}
		if err := s.devB[g].Upload(weights); err != nil {
			t.Fatal(err)
		}
		s.a[g] = make([]float32, rows*block)
		s.part[g] = make([]float32, rows*out)
	}
	return s
}

func (s *liveConvNeXtFC1GPU) close() {
	for g := range s.a {
		if s.devA[g] != nil {
			s.devA[g].Free()
		}
		if s.devB[g] != nil {
			s.devB[g].Free()
		}
		if s.devC[g] != nil {
			s.devC[g].Free()
		}
	}
}

func (s *liveConvNeXtFC1GPU) forward(l talkerLinear, input []float32, rows int) ([]float32, error) {
	const expectedRows, in, out, block = 256, 1024, 4096, 512
	if rows != expectedRows || len(input) != rows*in || l.inDim != in || l.outDim != out {
		return nil, fmt.Errorf("invalid live ConvNeXt FC1 input rows=%d size=%d", rows, len(input))
	}
	for g := range s.a {
		for row := 0; row < rows; row++ {
			copy(s.a[g][row*block:(row+1)*block], input[row*in+g*block:row*in+(g+1)*block])
		}
		if err := s.devA[g].Upload(s.a[g]); err != nil {
			return nil, err
		}
		if err := nvidia.Sgemm(rows, out, block, 1, s.devA[g], s.devB[g], s.devC[g]); err != nil {
			return nil, err
		}
		if err := nvidia.SyncErr(); err != nil {
			return nil, err
		}
		if err := s.devC[g].Download(s.part[g]); err != nil {
			return nil, err
		}
	}
	result := make([]float32, rows*out)
	for i := range result {
		result[i] = float32(s.part[0][i]+s.part[1][i]) + l.bias[i%out]
	}
	return result, nil
}

func TestDecoderConvNeXtFC1LiveGPUWaveform(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for opt-in live FC1 GPU")
	}
	root, oracle, trace := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"), os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR"), os.Getenv("GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR")
	if root == "" || oracle == "" || trace == "" {
		t.Fatal("set approved model, sentence64 oracle and decoder trace directories")
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU requested but NVIDIA SGEMM unavailable")
	}
	for name, sha := range map[string]string{
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, name), sha, 0); err != nil {
			t.Fatal(err)
		}
	}
	_, rawCodes := sentence64Fixture(t)
	const rows, in, out = 256, 1024, 4096
	for _, file := range []struct {
		name, hash string
		count      int
	}{
		{"cn256-norm.full.f32le", "0b45759f311819bf1cb04ddf91ac298adcde78d9c481d2cbd3b9d76db53a106f", rows * in},
		{"cn256-fc1.full.f32le", "04de2e4687fd288d6d8b59c6d420c59341ce4afa0e5fd87878ce6d666aac7aeb", rows * out},
	} {
		if err := verifyReleasedFile(filepath.Join(trace, file.name), file.hash, int64(file.count*4)); err != nil {
			t.Fatal(err)
		}
	}
	wavePath := filepath.Join(oracle, "sentence_64_waveform.f32le")
	if err := verifyReleasedFile(wavePath, sentence64WaveSHA, 122880*4); err != nil {
		t.Fatal(err)
	}
	rawWave, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	rawInput, err := os.ReadFile(filepath.Join(trace, "cn256-norm.full.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	rawOutput, err := os.ReadFile(filepath.Join(trace, "cn256-fc1.full.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	codes := make([]uint32, len(rawCodes)/4)
	for i := range codes {
		codes[i] = binary.LittleEndian.Uint32(rawCodes[4*i:])
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	prev := nvidia.SetStatsEnabled(true)
	t.Cleanup(func() { nvidia.SetStatsEnabled(prev) })
	before := nvidia.StatsSnapshot()
	freeBefore, total := nvidia.MemInfo()
	if freeBefore == 0 || total == 0 {
		t.Fatal("GPU memory info unavailable")
	}
	gpu := newLiveConvNeXtFC1GPU(t, m.preUpsample[1].block.fc1)
	freePrepared, _ := nvidia.MemInfo()
	var calls int
	callback := func(l talkerLinear, input []float32, length int) ([]float32, error) {
		calls++
		if length != rows || len(input) != rows*in {
			t.Fatalf("unexpected FC1 live input rows=%d size=%d", length, len(input))
		}
		compare := func(label string, live []float32, raw []byte) (float64, int) {
			t.Helper()
			if len(live)*4 != len(raw) {
				t.Fatalf("%s reference length mismatch", label)
			}
			maxDiff, changed := float64(0), 0
			for i, v := range live {
				bits := binary.LittleEndian.Uint32(raw[4*i:])
				if math.Float32bits(v) != bits {
					changed++
				}
				diff := math.Abs(float64(v) - float64(math.Float32frombits(bits)))
				if math.IsNaN(diff) || math.IsInf(diff, 0) {
					t.Fatalf("%s nonfinite at %d", label, i)
				}
				maxDiff = max(maxDiff, diff)
			}
			t.Logf("%s vs pinned Rust max_abs=%.9g changed=%d/%d", label, maxDiff, changed, len(live))
			return maxDiff, changed
		}
		inputDrift, _ := compare("live ConvNeXt norm input", input, rawInput)
		cpu := make([]float32, rows*out)
		for row := 0; row < rows; row++ {
			if err := decoderLinearForward(l, cpu[row*out:(row+1)*out], input[row*in:(row+1)*in]); err != nil {
				t.Fatal(err)
			}
		}
		result, err := gpu.forward(l, input, length)
		if err != nil {
			return nil, err
		}
		maxPeer, changedPeer := float64(0), 0
		for i, v := range result {
			if math.Float32bits(v) != math.Float32bits(cpu[i]) {
				changedPeer++
			}
			diff := math.Abs(float64(v) - float64(cpu[i]))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatal("nonfinite GPU/CPU", i)
			}
			maxPeer = max(maxPeer, diff)
		}
		t.Logf("live ConvNeXt fc1 GPU/CPU max_abs=%.9g changed=%d/%d", maxPeer, changedPeer, len(result))
		outputDrift, _ := compare("live ConvNeXt fc1 GPU output", result, rawOutput)
		if inputDrift > 4e-6 || changedPeer != 0 || maxPeer != 0 || outputDrift > 4e-5 {
			t.Fatalf("live FC1 bounds input %.9g GPU/CPU %.9g output %.9g", inputDrift, maxPeer, outputDrift)
		}
		return result, nil
	}
	callBefore := nvidia.StatsSnapshot()
	wave, err := m.decodeCodesWithDiagnosticFC1(codes, 64, nil, nil, nil, callback)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(wave) != 122880 {
		t.Fatalf("FC1 calls=%d waveform samples=%d", calls, len(wave))
	}
	callAfter := nvidia.StatsSnapshot()
	if callAfter.KernelLaunches-callBefore.KernelLaunches != 2 || callAfter.HostToDevice-callBefore.HostToDevice != 2 || callAfter.DeviceToHost-callBefore.DeviceToHost != 2 || callAfter.Mallocs != callBefore.Mallocs || callAfter.Frees != callBefore.Frees {
		t.Fatalf("request GPU lifecycle before=%+v after=%+v", callBefore, callAfter)
	}
	maxWave := float64(0)
	for i, v := range wave {
		diff := math.Abs(float64(v) - float64(math.Float32frombits(binary.LittleEndian.Uint32(rawWave[4*i:]))))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatal("nonfinite waveform difference", i)
		}
		maxWave = max(maxWave, diff)
	}
	if maxWave > 1.6e-6 {
		t.Fatalf("GPU FC1 waveform max %.9g exceeds unchanged 1.6e-6", maxWave)
	}
	freeRequest, _ := nvidia.MemInfo()
	gpu.close()
	closed := nvidia.StatsSnapshot()
	freeClosed, _ := nvidia.MemInfo()
	if closed.Mallocs-before.Mallocs != 6 || closed.Frees-before.Frees != 6 || closed.MallocBytes-before.MallocBytes != 26214400 || closed.FreeBytes-before.FreeBytes != 26214400 {
		t.Fatalf("GPU buffer imbalance before=%+v after=%+v", before, closed)
	}
	t.Logf("live ConvNeXt FC1 GPU waveform max_abs=%.9g gate=1.6e-6 alloc/free=%d bytes free-before=%d free-prepared=%d free-request=%d free-closed=%d total=%d (not peak VRAM)", maxWave, closed.MallocBytes-before.MallocBytes, freeBefore, freePrepared, freeRequest, freeClosed, total)
}
