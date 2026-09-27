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

// testDecoderGPUConv owns one diagnostic request's immutable block weights,
// CUDA allocations and host scratch. A new instance is required per request;
// it does not change Decoder12HzCPU or enable production GPU dispatch.
type testDecoderGPUConv struct {
	name                  string
	length, in, out, taps int
	bias                  []float32
	inputName, outputName string
	devA, devB, devC      []*nvidia.Buffer
	a, partial            [][]float32
	sums, owned           []float32
}

func newTestDecoderGPUConv(t testing.TB, c decoderConv1D, length int, name, inputName, outputName string) *testDecoderGPUConv {
	t.Helper()
	if c.inChannels <= 0 || c.outChannels <= 0 || c.k <= 0 || c.dilation != 1 || c.groups() != 1 || c.inChannels*c.k%512 != 0 || length <= 0 || len(c.weight) != c.inChannels*c.k*c.outChannels || len(c.bias) != c.outChannels {
		t.Fatal("unsupported diagnostic decoder convolution geometry")
	}
	const block = 512
	groups := c.inChannels * c.k / block
	s := &testDecoderGPUConv{name: name, length: length, in: c.inChannels, out: c.outChannels, taps: c.k, bias: c.bias, inputName: inputName, outputName: outputName,
		devA: make([]*nvidia.Buffer, groups), devB: make([]*nvidia.Buffer, groups), devC: make([]*nvidia.Buffer, groups), a: make([][]float32, groups), partial: make([][]float32, groups), sums: make([]float32, length*c.outChannels), owned: make([]float32, length*c.outChannels)}
	t.Cleanup(s.close)
	k := c.inChannels * c.k
	for group := 0; group < groups; group++ {
		var err error
		s.devA[group], err = nvidia.Malloc(length * block)
		if err != nil {
			t.Fatal(err)
		}
		s.devB[group], err = nvidia.Malloc(block * c.outChannels)
		if err != nil {
			t.Fatal(err)
		}
		s.devC[group], err = nvidia.Malloc(length * c.outChannels)
		if err != nil {
			t.Fatal(err)
		}
		weights := make([]float32, block*c.outChannels)
		for out := 0; out < c.outChannels; out++ {
			for in := 0; in < block; in++ {
				weights[in*c.outChannels+out] = c.weight[out*k+group*block+in]
			}
		}
		if err := s.devB[group].Upload(weights); err != nil {
			t.Fatal(err)
		}
		s.a[group] = make([]float32, length*block)
		s.partial[group] = make([]float32, length*c.outChannels)
	}
	return s
}

func (s *testDecoderGPUConv) close() {
	for i := range s.devA {
		if s.devA[i] != nil {
			s.devA[i].Free()
		}
		if s.devB[i] != nil {
			s.devB[i].Free()
		}
		if s.devC[i] != nil {
			s.devC[i].Free()
		}
	}
}

func (s *testDecoderGPUConv) forward(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
	if length != s.length || len(input) != s.in*s.length || c.inChannels != s.in || c.outChannels != s.out || c.k != s.taps {
		return nil, 0, fmt.Errorf("invalid diagnostic %s input", s.name)
	}
	const block = 512
	for group := range s.a {
		clear(s.a[group])
	}
	for pos := 0; pos < length; pos++ {
		for ic := 0; ic < s.in; ic++ {
			for tap := 0; tap < s.taps; tap++ {
				src := pos + tap - (s.taps - 1)
				if src >= 0 {
					index := ic*s.taps + tap
					s.a[index/block][pos*block+index%block] = input[ic*length+src]
				}
			}
		}
	}
	for group := range s.a {
		if err := s.devA[group].Upload(s.a[group]); err != nil {
			return nil, 0, err
		}
		if err := nvidia.Sgemm(length, s.out, block, 1, s.devA[group], s.devB[group], s.devC[group]); err != nil {
			return nil, 0, err
		}
		if err := nvidia.SyncErr(); err != nil {
			return nil, 0, err
		}
		if err := s.devC[group].Download(s.partial[group]); err != nil {
			return nil, 0, err
		}
	}
	clear(s.sums)
	for group := range s.partial {
		for i, v := range s.partial[group] {
			s.sums[i] += v
		}
	}
	for out := 0; out < s.out; out++ {
		for pos := 0; pos < length; pos++ {
			s.owned[out*length+pos] = s.sums[pos*s.out+out] + s.bias[out]
		}
	}
	return append([]float32(nil), s.owned...), length, nil
}

func TestDecoderTwoGPUConvsPinnedWaveform(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in decoder-only hybrid gate")
	}
	root := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR")
	oracle := os.Getenv("GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	if root == "" || oracle == "" {
		t.Fatal("set GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR and GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR")
	}
	if !nvidia.SgemmReady() {
		t.Fatal("opt-in GPU gate requested but NVIDIA SGEMM unavailable")
	}
	for name, sha := range map[string]string{"speech_tokenizer/config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167", "speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"} {
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
	wavePath := filepath.Join(oracle, "sentence_64_waveform.f32le")
	if err := verifyReleasedFile(wavePath, sentence64WaveSHA, 64*1920*4); err != nil {
		t.Fatal(err)
	}
	rawWave, err := os.ReadFile(wavePath)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([]uint32, len(rawCodes)/4)
	for i := range codes {
		codes[i] = binary.LittleEndian.Uint32(rawCodes[i*4:])
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	prev := nvidia.SetStatsEnabled(true)
	t.Cleanup(func() { nvidia.SetStatsEnabled(prev) })
	before := nvidia.StatsSnapshot()
	freeBefore, totalBefore := nvidia.MemInfo()
	if freeBefore == 0 || totalBefore == 0 {
		t.Fatal("GPU memory information unavailable")
	}
	pre := newTestDecoderGPUConv(t, m.preConv, 64, "preconv", "quantized.full.f32le", "preconv.full.f32le")
	init := newTestDecoderGPUConv(t, m.decoderInit, 256, "initconv", "upsample1.full.f32le", "decoderinit.full.f32le")
	freePrepared, _ := nvidia.MemInfo()
	if freePrepared >= freeBefore {
		t.Fatal("GPU buffers were not allocated")
	}
	check := func(s *testDecoderGPUConv, c decoderConv1D, input, out []float32) {
		t.Helper()
		cpu, length, err := c.forward(input, s.length)
		if err != nil || length != s.length || len(cpu) != len(out) {
			t.Fatalf("%s CPU peer failed length=%d err=%v", s.name, length, err)
		}
		for i, want := range cpu {
			if math.Float32bits(out[i]) != math.Float32bits(want) {
				t.Fatalf("%s live GPU/CPU mismatch at %d got %.9g want %.9g", s.name, i, out[i], want)
			}
		}
		root := filepath.Join("testdata", "customvoice_0b6_ryan_hello", map[string]string{"preconv": "decoder_preconv", "initconv": "decoder_initconv"}[s.name])
		for _, item := range []struct {
			name string
			got  []float32
		}{{s.inputName, input}, {s.outputName, out}} {
			raw, err := os.ReadFile(filepath.Join(root, item.name))
			if err != nil || len(raw) != len(item.got)*4 {
				t.Fatalf("%s %s reference size/read error: %v", s.name, item.name, err)
			}
			maxDiff, changed := float64(0), 0
			for i, v := range item.got {
				bits := binary.LittleEndian.Uint32(raw[4*i:])
				diff := math.Abs(float64(v) - float64(math.Float32frombits(bits)))
				if math.IsNaN(diff) || math.IsInf(diff, 0) {
					t.Fatalf("%s %s nonfinite reference difference at %d", s.name, item.name, i)
				}
				if math.Float32bits(v) != bits {
					changed++
				}
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("%s %s vs pinned Rust: max_abs=%.9g changed=%d/%d", s.name, item.name, maxDiff, changed, len(item.got))
			// The second convolution receives the live CPU upstream output.
			// Measure its distinct Rust boundary drift; never replace this input
			// with the frozen Rust tensor to make the waveform gate pass.
			limit := 0.0
			if s.name == "initconv" {
				if item.name == s.inputName {
					limit = 4e-5
				} else {
					limit = 1e-5
				}
			}
			if maxDiff > limit || (s.name == "preconv" && changed != 0) {
				t.Fatalf("%s %s pinned Rust drift %.9g exceeds %.9g", s.name, item.name, maxDiff, limit)
			}
		}
	}
	var preCalls, initCalls int
	preCallback := func(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
		out, n, err := pre.forward(c, input, length)
		if err == nil {
			preCalls++
			check(pre, c, input, out)
		}
		return out, n, err
	}
	initCallback := func(c decoderConv1D, input []float32, length int) ([]float32, int, error) {
		out, n, err := init.forward(c, input, length)
		if err == nil {
			initCalls++
			check(init, c, input, out)
		}
		return out, n, err
	}
	callBefore := nvidia.StatsSnapshot()
	wave, err := m.decodeCodesWithConvs(codes, 64, preCallback, initCallback)
	if err != nil {
		t.Fatal(err)
	}
	if preCalls != 1 || initCalls != 1 || len(wave)*4 != len(rawWave) {
		t.Fatalf("hybrid stage calls=%d/%d waveform=%d", preCalls, initCalls, len(wave))
	}
	callAfter := nvidia.StatsSnapshot()
	if callAfter.Mallocs != callBefore.Mallocs || callAfter.Frees != callBefore.Frees || callAfter.KernelLaunches-callBefore.KernelLaunches != 17 || callAfter.HostToDevice-callBefore.HostToDevice != 17 || callAfter.DeviceToHost-callBefore.DeviceToHost != 17 {
		t.Fatalf("unexpected request GPU lifecycle/transfers before=%+v after=%+v", callBefore, callAfter)
	}
	maxErr := float64(0)
	for i, v := range wave {
		diff := math.Abs(float64(v) - float64(math.Float32frombits(binary.LittleEndian.Uint32(rawWave[4*i:]))))
		if math.IsNaN(diff) || math.IsInf(diff, 0) {
			t.Fatal("nonfinite waveform error", i)
		}
		if diff > maxErr {
			maxErr = diff
		}
	}
	if maxErr > 1.6e-6 {
		t.Fatalf("hybrid waveform max %.9g exceeds unchanged 1.6e-6", maxErr)
	}
	if freeAfter, _ := nvidia.MemInfo(); freeAfter != freePrepared {
		t.Logf("GPU free memory after hybrid request changed by %d bytes (diagnostic, allocator/cache may vary)", int64(freeAfter)-int64(freePrepared))
	}
	pre.close()
	init.close()
	after := nvidia.StatsSnapshot()
	if after.Mallocs-before.Mallocs != 51 || after.Frees-before.Frees != 51 || after.MallocBytes-before.MallocBytes != 80871424 || after.FreeBytes-before.FreeBytes != 80871424 || after.KernelLaunches-before.KernelLaunches != 17 {
		t.Fatalf("unexpected hybrid GPU lifecycle/launches before=%+v after=%+v", before, after)
	}
	freeClosed, _ := nvidia.MemInfo()
	t.Logf("hybrid decoder 64 frames: max waveform error %.9g (gate 1.6e-6), GPU alloc/free=%d bytes, free-before=%d free-prepared=%d free-closed=%d total=%d; snapshots do not measure in-flight peak VRAM", maxErr, after.MallocBytes-before.MallocBytes, freeBefore, freePrepared, freeClosed, totalBefore)
}
