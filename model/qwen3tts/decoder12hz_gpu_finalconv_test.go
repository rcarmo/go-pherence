package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// TestDecoderFinalConvPinnedGPU is a single-stage diagnostic on the independent
// full Rust trace. It does not dispatch GPU synthesis or modify the decoder.
func TestDecoderFinalConvPinnedGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in GPU diagnostic")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	trace := os.Getenv("GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR")
	if root == "" || trace == "" {
		t.Fatal("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR and GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR")
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU test requested but NVIDIA SGEMM unavailable")
	}
	for name, sha := range map[string]string{
		"speech_tokenizer/config.json":       "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
		"speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
	} {
		if err := verifyReleasedFile(filepath.Join(root, name), sha, 0); err != nil {
			t.Fatal(err)
		}
	}
	sentence64Fixture(t)
	const length, inChannels, taps, tile = 122880, 96, 7, 128
	for _, f := range []struct {
		name, sha string
		size      int64
	}{
		{"finalsnake.full.f32le", "24230d220270640cfc161ffc90b1671111d12d4cbec823cefafa70e8107fda1d", length * inChannels * 4},
		{"finalconv.full.f32le", "e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb", length * 4},
	} {
		if err := verifyReleasedFile(filepath.Join(trace, f.name), f.sha, f.size); err != nil {
			t.Fatal(err)
		}
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	c := m.finalConv
	if c.inChannels != inChannels || c.outChannels != 1 || c.k != taps || c.dilation != 1 || c.groups() != 1 || len(c.weight) != inChannels*taps || len(c.bias) != 1 || decoderBlockSize(inChannels*taps) != 336 {
		t.Fatal("unexpected final-convolution geometry")
	}
	inputBytes, err := os.ReadFile(filepath.Join(trace, "finalsnake.full.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	input := make([]float32, len(inputBytes)/4)
	for i := range input {
		input[i] = math.Float32frombits(binary.LittleEndian.Uint32(inputBytes[4*i:]))
	}
	outputBytes, err := os.ReadFile(filepath.Join(trace, "finalconv.full.f32le"))
	if err != nil {
		t.Fatal(err)
	}
	cpu, n, err := c.forward(input, length)
	if err != nil || n != length || len(cpu) != length {
		t.Fatalf("CPU final convolution length=%d err=%v", n, err)
	}
	maxCPU := float64(0)
	for i, v := range cpu {
		diff := math.Abs(float64(v) - float64(math.Float32frombits(binary.LittleEndian.Uint32(outputBytes[4*i:]))))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatalf("nonfinite CPU/Rust difference at %d", i)
		}
		maxCPU = max(maxCPU, diff)
	}
	if maxCPU > 1e-6 {
		t.Fatalf("CPU final convolution vs pinned Rust max=%.9g > 1e-6", maxCPU)
	}

	const block = 336
	groups := inChannels * taps / block
	prev := nvidia.SetStatsEnabled(true)
	t.Cleanup(func() { nvidia.SetStatsEnabled(prev) })
	before := nvidia.StatsSnapshot()
	devA, devB, devC := make([]*nvidia.Buffer, groups), make([]*nvidia.Buffer, groups), make([]*nvidia.Buffer, groups)
	t.Cleanup(func() {
		for g := 0; g < groups; g++ {
			if devA[g] != nil {
				devA[g].Free()
			}
			if devB[g] != nil {
				devB[g].Free()
			}
			if devC[g] != nil {
				devC[g].Free()
			}
		}
	})
	a := make([][]float32, groups)
	part := make([][]float32, groups)
	for g := 0; g < groups; g++ {
		devA[g], err = nvidia.Malloc(tile * block)
		if err != nil {
			t.Fatal(err)
		}
		devB[g], err = nvidia.Malloc(block)
		if err != nil {
			t.Fatal(err)
		}
		devC[g], err = nvidia.Malloc(tile)
		if err != nil {
			t.Fatal(err)
		}
		if err := devB[g].Upload(c.weight[g*block : (g+1)*block]); err != nil {
			t.Fatal(err)
		}
		a[g], part[g] = make([]float32, tile*block), make([]float32, tile)
	}
	start := nvidia.StatsSnapshot()
	maxGPUCPU, maxGPURust, changedCPU := float64(0), float64(0), 0
	for pos := 0; pos < length; pos += tile {
		rows := min(tile, length-pos)
		for g := range a {
			clear(a[g])
		}
		for trow := 0; trow < rows; trow++ {
			for ic := 0; ic < inChannels; ic++ {
				for tap := 0; tap < taps; tap++ {
					src := pos + trow + tap - (taps - 1)
					if src >= 0 {
						idx := ic*taps + tap
						a[idx/block][trow*block+idx%block] = input[ic*length+src]
					}
				}
			}
		}
		for g := range a {
			if err := devA[g].Upload(a[g]); err != nil {
				t.Fatal(err)
			}
			if err := nvidia.Sgemm(tile, 1, block, 1, devA[g], devB[g], devC[g]); err != nil {
				t.Fatal(err)
			}
			if err := nvidia.SyncErr(); err != nil {
				t.Fatal(err)
			}
			if err := devC[g].Download(part[g]); err != nil {
				t.Fatal(err)
			}
		}
		for row := 0; row < rows; row++ {
			i := pos + row
			v := float32(part[0][row]+part[1][row]) + c.bias[0]
			ref := math.Float32frombits(binary.LittleEndian.Uint32(outputBytes[4*i:]))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("nonfinite GPU final convolution at %d", i)
			}
			if math.Float32bits(v) != math.Float32bits(cpu[i]) {
				changedCPU++
			}
			maxGPUCPU = max(maxGPUCPU, math.Abs(float64(v)-float64(cpu[i])))
			maxGPURust = max(maxGPURust, math.Abs(float64(v)-float64(ref)))
		}
	}
	after := nvidia.StatsSnapshot()
	wantLaunches := uint64(groups * (length / tile))
	if after.KernelLaunches-start.KernelLaunches != wantLaunches || after.HostToDevice-start.HostToDevice != wantLaunches || after.DeviceToHost-start.DeviceToHost != wantLaunches || after.Mallocs != start.Mallocs || after.Frees != start.Frees {
		t.Fatalf("GPU stage lifecycle/transfer mismatch start=%+v after=%+v", start, after)
	}
	for g := range devA {
		devA[g].Free()
		devB[g].Free()
		devC[g].Free()
	}
	closed := nvidia.StatsSnapshot()
	if closed.Mallocs-before.Mallocs != 6 || closed.Frees-before.Frees != 6 || closed.MallocBytes-before.MallocBytes != closed.FreeBytes-before.FreeBytes {
		t.Fatalf("GPU allocation/free imbalance before=%+v after=%+v", before, closed)
	}
	t.Logf("final convolution 64 frames: CPU/Rust max=%.9g GPU/CPU max=%.9g changed=%d/%d GPU/Rust max=%.9g launches=%d allocated/freed=%d bytes", maxCPU, maxGPUCPU, changedCPU, length, maxGPURust, wantLaunches, closed.MallocBytes-before.MallocBytes)
	// The two 336-wide GPU reductions may round differently from Candle's
	// CPU GEMM. Check both calculations against the same fixed stage bound;
	// retain the differing-bit count instead of claiming bitwise parity.
	if maxGPUCPU > 1e-6 || maxGPURust > 1e-6 {
		t.Fatalf("final convolution stage exceeds live CPU or pinned Rust 1e-6 gate")
	}
}
