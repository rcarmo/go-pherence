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

// liveFinalConvGPU owns its buffers for one diagnostic request. This test-only
// adapter uses the same two 336-wide reductions as the pinned stage probe.
type liveFinalConvGPU struct {
	a                [2][]float32
	part             [2][]float32
	devA, devB, devC [2]*nvidia.Buffer
}

func newLiveFinalConvGPU(t testing.TB, c decoderConv1D) *liveFinalConvGPU {
	t.Helper()
	const tile, block = 128, 336
	if c.inChannels != 96 || c.outChannels != 1 || c.k != 7 || c.dilation != 1 || c.groups() != 1 || len(c.weight) != 2*block || len(c.bias) != 1 {
		t.Fatal("unexpected final decoder convolution geometry")
	}
	s := new(liveFinalConvGPU)
	t.Cleanup(s.close)
	for g := range s.a {
		var err error
		s.devA[g], err = nvidia.Malloc(tile * block)
		if err != nil {
			t.Fatal(err)
		}
		s.devB[g], err = nvidia.Malloc(block)
		if err != nil {
			t.Fatal(err)
		}
		s.devC[g], err = nvidia.Malloc(tile)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.devB[g].Upload(c.weight[g*block : (g+1)*block]); err != nil {
			t.Fatal(err)
		}
		s.a[g], s.part[g] = make([]float32, tile*block), make([]float32, tile)
	}
	return s
}

func (s *liveFinalConvGPU) close() {
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

func (s *liveFinalConvGPU) forward(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
	const tile, block, taps, channels = 128, 336, 7, 96
	if length != 122880 || len(input) != channels*length || c.inChannels != channels || c.outChannels != 1 || c.k != taps {
		return nil, 0, fmt.Errorf("invalid live final convolution geometry length=%d input=%d", length, len(input))
	}
	out := make([]float32, length)
	for pos := 0; pos < length; pos += tile {
		for g := range s.a {
			clear(s.a[g])
		}
		for row := 0; row < tile; row++ {
			for ic := 0; ic < channels; ic++ {
				for tap := 0; tap < taps; tap++ {
					src := pos + row + tap - (taps - 1)
					if src >= 0 {
						idx := ic*taps + tap
						s.a[idx/block][row*block+idx%block] = input[ic*length+src]
					}
				}
			}
		}
		for g := range s.a {
			if err := s.devA[g].Upload(s.a[g]); err != nil {
				return nil, 0, err
			}
			if err := nvidia.Sgemm(tile, 1, block, 1, s.devA[g], s.devB[g], s.devC[g]); err != nil {
				return nil, 0, err
			}
			if err := nvidia.SyncErr(); err != nil {
				return nil, 0, err
			}
			if err := s.devC[g].Download(s.part[g]); err != nil {
				return nil, 0, err
			}
		}
		for row := 0; row < tile; row++ {
			out[pos+row] = float32(s.part[0][row]+s.part[1][row]) + c.bias[0]
		}
	}
	return out, length, nil
}

func TestDecoderFinalConvLiveGPUWaveform(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in GPU waveform diagnostic")
	}
	root, oracle, trace := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"), os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR"), os.Getenv("GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR")
	if root == "" || oracle == "" || trace == "" {
		t.Fatal("set model, sentence64 oracle and decoder trace directories")
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU requested but NVIDIA SGEMM unavailable")
	}
	for name, sha := range map[string]string{"speech_tokenizer/config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167", "speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"} {
		if err := verifyReleasedFile(filepath.Join(root, name), sha, 0); err != nil {
			t.Fatal(err)
		}
	}
	_, rawCodes := sentence64Fixture(t)
	for _, file := range []struct {
		name, sha string
		size      int64
	}{
		{"finalsnake.full.f32le", "24230d220270640cfc161ffc90b1671111d12d4cbec823cefafa70e8107fda1d", 96 * 122880 * 4},
		{"finalconv.full.f32le", "e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb", 122880 * 4},
	} {
		if err := verifyReleasedFile(filepath.Join(trace, file.name), file.sha, file.size); err != nil {
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
	rawInput, err := os.ReadFile(filepath.Join(trace, "finalsnake.full.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	rawOutput, err := os.ReadFile(filepath.Join(trace, "finalconv.full.f32le"))
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
		t.Fatal("GPU memory information unavailable")
	}
	gpu := newLiveFinalConvGPU(t, m.finalConv)
	freePrepared, _ := nvidia.MemInfo()
	var calls int
	callback := func(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
		calls++
		maxInput, changedInput := float64(0), 0
		for i, v := range input {
			bits := binary.LittleEndian.Uint32(rawInput[4*i:])
			want := math.Float32frombits(bits)
			if math.Float32bits(v) != bits {
				changedInput++
			}
			diff := math.Abs(float64(v) - float64(want))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatal("nonfinite final input difference", i)
			}
			maxInput = max(maxInput, diff)
		}
		t.Logf("live final input vs Rust: max_abs=%.9g changed=%d/%d", maxInput, changedInput, len(input))
		cpu, n, err := c.forward(input, length)
		if err != nil {
			return nil, 0, err
		}
		out, n, err := gpu.forward(c, input, n)
		if err != nil {
			return nil, 0, err
		}
		maxGPUCPU, maxGPURust, changedGPUCPU := float64(0), float64(0), 0
		for i, v := range out {
			if math.Float32bits(v) != math.Float32bits(cpu[i]) {
				changedGPUCPU++
			}
			diff := math.Abs(float64(v) - float64(cpu[i]))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatal("nonfinite GPU/CPU difference", i)
			}
			maxGPUCPU = max(maxGPUCPU, diff)
			ref := math.Float32frombits(binary.LittleEndian.Uint32(rawOutput[4*i:]))
			diff = math.Abs(float64(v) - float64(ref))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatal("nonfinite GPU/Rust difference", i)
			}
			maxGPURust = max(maxGPURust, diff)
		}
		t.Logf("live final output: GPU/CPU max_abs=%.9g changed=%d/%d GPU/Rust max_abs=%.9g", maxGPUCPU, changedGPUCPU, len(out), maxGPURust)
		// Keep the live CPU/Rust boundary drift separate from the GPU/CPU
		// calculation error; neither intermediate is injected or substituted.
		if maxInput > 1e-3 || maxGPURust > 1.6e-6 || maxGPUCPU > 1e-6 {
			t.Fatalf("live final boundaries exceed input=1e-3, GPU/Rust=1.6e-6 or GPU/CPU=1e-6: %.9g %.9g %.9g", maxInput, maxGPURust, maxGPUCPU)
		}
		return out, n, nil
	}
	callBefore := nvidia.StatsSnapshot()
	wave, err := m.decodeCodesWithDiagnosticConvs(codes, 64, nil, nil, callback)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(wave) != 122880 {
		t.Fatalf("final GPU callback calls=%d waveform length=%d", calls, len(wave))
	}
	callAfter := nvidia.StatsSnapshot()
	if callAfter.Mallocs != callBefore.Mallocs || callAfter.Frees != callBefore.Frees || callAfter.KernelLaunches-callBefore.KernelLaunches != 1920 || callAfter.HostToDevice-callBefore.HostToDevice != 1920 || callAfter.DeviceToHost-callBefore.DeviceToHost != 1920 {
		t.Fatalf("request GPU lifecycle/transfers before=%+v after=%+v", callBefore, callAfter)
	}
	maxWave := float64(0)
	for i, v := range wave {
		ref := math.Float32frombits(binary.LittleEndian.Uint32(rawWave[4*i:]))
		diff := math.Abs(float64(v) - float64(ref))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatal("nonfinite waveform difference", i)
		}
		maxWave = max(maxWave, diff)
	}
	if maxWave > 1.6e-6 {
		t.Fatalf("live final GPU waveform max %.9g exceeds unchanged 1.6e-6", maxWave)
	}
	freeRequest, _ := nvidia.MemInfo()
	gpu.close()
	closed := nvidia.StatsSnapshot()
	freeClosed, _ := nvidia.MemInfo()
	if closed.Mallocs-before.Mallocs != 6 || closed.Frees-before.Frees != 6 || closed.MallocBytes-before.MallocBytes != 347776 || closed.FreeBytes-before.FreeBytes != 347776 {
		t.Fatalf("GPU buffer imbalance before=%+v after=%+v", before, closed)
	}
	t.Logf("live final GPU waveform: max_abs=%.9g gate=1.6e-6; buffers=6/6 bytes=347776 free-before=%d free-prepared=%d free-request=%d free-closed=%d total=%d; snapshots are not peak VRAM", maxWave, freeBefore, freePrepared, freeRequest, freeClosed, total)
}
