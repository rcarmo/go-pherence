package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// TestDecoderThreeGPUConvsPinnedWaveform runs three request-local GPU
// convolutions within one live decoder waveform. No production dispatch uses it.
func TestDecoderThreeGPUConvsPinnedWaveform(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in decoder hybrid")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	oracle := os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	trace := os.Getenv("GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR")
	if root == "" || oracle == "" || trace == "" {
		t.Fatal("set model, sentence64 oracle and decoder trace directories")
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU gate requested but NVIDIA SGEMM unavailable")
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
	for _, stage := range []struct {
		dir, input, inputHash, output, outputHash string
		in, out                                   int
	}{
		{"decoder_preconv", "quantized.full.f32le", "4cce48ec0c48189105f28e24ab6404cadbdec7130f84a3f07c19979989298a9d", "preconv.full.f32le", "75c0a7feda1be45023577e1f2675f909b75351fd5bc977978e385597a4703685", 512 * 64, 1024 * 64},
		{"decoder_initconv", "upsample1.full.f32le", "5884389f6891c21e173029dab67d3d138a3dd8dc4b76611cc32b0abdf854f372", "decoderinit.full.f32le", "2dd3e28511ffb314f05ab258df7ed8b69d8663b29a33f04fe0dc9ddb7c5afc85", 1024 * 256, 1536 * 256},
	} {
		for _, file := range []struct {
			name, hash string
			count      int
		}{{stage.input, stage.inputHash, stage.in}, {stage.output, stage.outputHash, stage.out}} {
			if err := verifyReleasedFile(filepath.Join("testdata", "customvoice_0b6_ryan_hello", stage.dir, file.name), file.hash, int64(file.count*4)); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, file := range []struct {
		name, hash string
		count      int
	}{
		{"finalsnake.full.f32le", "24230d220270640cfc161ffc90b1671111d12d4cbec823cefafa70e8107fda1d", 96 * 122880},
		{"finalconv.full.f32le", "e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb", 122880},
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
	pre := newTestDecoderGPUConv(t, m.preConv, 64, "preconv", "quantized.full.f32le", "preconv.full.f32le")
	init := newTestDecoderGPUConv(t, m.decoderInit, 256, "initconv", "upsample1.full.f32le", "decoderinit.full.f32le")
	final := newLiveFinalConvGPU(t, m.finalConv)
	freePrepared, _ := nvidia.MemInfo()
	if freePrepared >= freeBefore {
		t.Fatal("GPU buffers were not allocated")
	}
	checkReference := func(stage, file string, live []float32, limit float64) {
		t.Helper()
		dir := trace
		if stage != "final" {
			dir = filepath.Join("testdata", "customvoice_0b6_ryan_hello", map[string]string{"preconv": "decoder_preconv", "initconv": "decoder_initconv"}[stage])
		}
		raw, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil || len(raw) != len(live)*4 {
			t.Fatalf("%s %s reference read/length: %v", stage, file, err)
		}
		changed, maxDiff := 0, float64(0)
		for i, v := range live {
			bits := binary.LittleEndian.Uint32(raw[4*i:])
			if math.Float32bits(v) != bits {
				changed++
			}
			diff := math.Abs(float64(v) - float64(math.Float32frombits(bits)))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatalf("nonfinite %s %s at %d", stage, file, i)
			}
			maxDiff = max(maxDiff, diff)
		}
		t.Logf("%s %s vs pinned Rust: max_abs=%.9g changed=%d/%d", stage, file, maxDiff, changed, len(live))
		if maxDiff > limit || (limit == 0 && changed != 0) {
			t.Fatalf("%s %s Rust boundary exceeds %.9g", stage, file, limit)
		}
	}
	preCalls, initCalls, finalCalls := 0, 0, 0
	preCallback := func(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
		preCalls++
		checkReference("preconv", pre.inputName, input, 0)
		out, n, err := pre.forward(c, input, length)
		if err != nil {
			return nil, 0, err
		}
		cpu, _, err := c.forward(input, length)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out {
			if math.Float32bits(v) != math.Float32bits(cpu[i]) {
				t.Fatalf("preconv GPU/CPU mismatch at %d", i)
			}
		}
		checkReference("preconv", pre.outputName, out, 0)
		return out, n, nil
	}
	initCallback := func(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
		initCalls++
		checkReference("initconv", init.inputName, input, 4e-5)
		out, n, err := init.forward(c, input, length)
		if err != nil {
			return nil, 0, err
		}
		cpu, _, err := c.forward(input, length)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out {
			if math.Float32bits(v) != math.Float32bits(cpu[i]) {
				t.Fatalf("initconv GPU/CPU mismatch at %d", i)
			}
		}
		checkReference("initconv", init.outputName, out, 1e-5)
		return out, n, nil
	}
	finalCallback := func(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
		finalCalls++
		checkReference("final", "finalsnake.full.f32le", input, 1e-3)
		out, n, err := final.forward(c, input, length)
		if err != nil {
			return nil, 0, err
		}
		cpu, _, err := c.forward(input, length)
		if err != nil {
			t.Fatal(err)
		}
		maxDiff, changed := float64(0), 0
		for i, v := range out {
			if math.Float32bits(v) != math.Float32bits(cpu[i]) {
				changed++
			}
			diff := math.Abs(float64(v) - float64(cpu[i]))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatal("nonfinite final GPU/CPU", i)
			}
			maxDiff = max(maxDiff, diff)
		}
		t.Logf("final live GPU/CPU max_abs=%.9g changed=%d/%d", maxDiff, changed, len(out))
		if maxDiff > 1e-6 {
			t.Fatalf("final GPU/CPU exceeds 1e-6: %.9g", maxDiff)
		}
		checkReference("final", "finalconv.full.f32le", out, 1.6e-6)
		return out, n, nil
	}
	callBefore := nvidia.StatsSnapshot()
	wave, err := m.decodeCodesWithDiagnosticConvs(codes, 64, preCallback, initCallback, finalCallback)
	if err != nil {
		t.Fatal(err)
	}
	if preCalls != 1 || initCalls != 1 || finalCalls != 1 || len(wave) != 122880 {
		t.Fatalf("stage calls=%d/%d/%d wave=%d", preCalls, initCalls, finalCalls, len(wave))
	}
	callAfter := nvidia.StatsSnapshot()
	if callAfter.Mallocs != callBefore.Mallocs || callAfter.Frees != callBefore.Frees || callAfter.KernelLaunches-callBefore.KernelLaunches != 1937 || callAfter.HostToDevice-callBefore.HostToDevice != 1937 || callAfter.DeviceToHost-callBefore.DeviceToHost != 1937 {
		t.Fatalf("request GPU lifecycle/transfers before=%+v after=%+v", callBefore, callAfter)
	}
	maxWave := float64(0)
	for i, v := range wave {
		diff := math.Abs(float64(v) - float64(math.Float32frombits(binary.LittleEndian.Uint32(rawWave[4*i:]))))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatal("nonfinite waveform", i)
		}
		maxWave = max(maxWave, diff)
	}
	if maxWave > 1.6e-6 {
		t.Fatalf("three-conv waveform max %.9g exceeds unchanged 1.6e-6", maxWave)
	}
	freeRequest, _ := nvidia.MemInfo()
	pre.close()
	init.close()
	final.close()
	closed := nvidia.StatsSnapshot()
	freeClosed, _ := nvidia.MemInfo()
	if closed.Mallocs-before.Mallocs != 57 || closed.Frees-before.Frees != 57 || closed.MallocBytes-before.MallocBytes != 81219200 || closed.FreeBytes-before.FreeBytes != 81219200 {
		t.Fatalf("GPU buffer imbalance before=%+v after=%+v", before, closed)
	}
	t.Logf("three GPU convolutions live decoder: waveform max_abs=%.9g gate=1.6e-6 launches=1937 alloc/free=81219200 bytes free-before=%d free-prepared=%d free-request=%d free-closed=%d total=%d (not peak VRAM)", maxWave, freeBefore, freePrepared, freeRequest, freeClosed, total)
}
