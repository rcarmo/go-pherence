package qwen3tts

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// TestDecoderConvNeXtFC1PinnedGPU checks one batched pointwise projection on
// independent full Rust tensors. It never enables GPU decoder dispatch.
func TestDecoderConvNeXtFC1PinnedGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_QWEN3TTS_GPU_TEST") != "1" {
		t.Skip("set GO_PHERENCE_QWEN3TTS_GPU_TEST=1 for the opt-in GPU projection")
	}
	root, trace := os.Getenv("GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR"), os.Getenv("GO_PHERENCE_QWEN3TTS_DECODER_TRACE_DIR")
	if root == "" || trace == "" {
		t.Fatal("set approved model and independent decoder trace directories")
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
	sentence64Fixture(t)
	const rows, in, out = 256, 1024, 4096
	files := []struct {
		name, hash string
		count      int
	}{
		{"cn256-norm.full.f32le", "0b45759f311819bf1cb04ddf91ac298adcde78d9c481d2cbd3b9d76db53a106f", rows * in},
		{"cn256-fc1.full.f32le", "04de2e4687fd288d6d8b59c6d420c59341ce4afa0e5fd87878ce6d666aac7aeb", rows * out},
	}
	var traces [2][]float32
	for idx, file := range files {
		path := filepath.Join(trace, file.name)
		if err := verifyReleasedFile(path, file.hash, int64(file.count*4)); err != nil {
			t.Fatal(err)
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		traces[idx] = make([]float32, file.count)
		for i := range traces[idx] {
			traces[idx][i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	}
	m, err := LoadDecoder12HzCPUFromDir(root)
	if err != nil {
		t.Fatal(err)
	}
	l := m.preUpsample[1].block.fc1
	if l.inDim != in || l.outDim != out || len(l.weight) != in*out || len(l.bias) != out {
		t.Fatalf("unexpected ConvNeXt fc1 geometry %d/%d weights=%d bias=%d", l.inDim, l.outDim, len(l.weight), len(l.bias))
	}
	cpu := make([]float32, rows*out)
	for row := 0; row < rows; row++ {
		if err := decoderLinearForward(l, cpu[row*out:(row+1)*out], traces[0][row*in:(row+1)*in]); err != nil {
			t.Fatal(err)
		}
	}
	compare := func(label string, got, want []float32) (float64, int) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length %d != %d", label, len(got), len(want))
		}
		maxDiff, changed := float64(0), 0
		for i, v := range got {
			if math.Float32bits(v) != math.Float32bits(want[i]) {
				changed++
			}
			diff := math.Abs(float64(v) - float64(want[i]))
			if math.IsNaN(diff) || math.IsInf(diff, 0) {
				t.Fatalf("%s nonfinite difference at %d", label, i)
			}
			maxDiff = max(maxDiff, diff)
		}
		t.Logf("%s max_abs=%.9g changed=%d/%d", label, maxDiff, changed, len(got))
		return maxDiff, changed
	}
	cpuRust, cpuChanged := compare("CPU/Rust fc1", cpu, traces[1])
	if cpuChanged != 0 || cpuRust != 0 {
		t.Fatal("CPU fc1 no longer matches pinned Rust bitwise")
	}
	transposed := make([]float32, in*out)
	for o := 0; o < out; o++ {
		for i := 0; i < in; i++ {
			transposed[i*out+o] = l.weight[o*in+i]
		}
	}
	prev := nvidia.SetStatsEnabled(true)
	t.Cleanup(func() { nvidia.SetStatsEnabled(prev) })
	before := nvidia.StatsSnapshot()
	freeBefore, total := nvidia.MemInfo()
	if freeBefore == 0 || total == 0 {
		t.Fatal("GPU memory info unavailable")
	}
	gpu, err := nvidia.SgemmHost(rows, out, in, 1, traces[0], transposed)
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < rows; row++ {
		for o, bias := range l.bias {
			gpu[row*out+o] += bias
		}
	}
	after := nvidia.StatsSnapshot()
	if after.KernelLaunches-before.KernelLaunches != 1 || after.HostToDevice-before.HostToDevice != 2 || after.DeviceToHost-before.DeviceToHost != 1 || after.Mallocs-before.Mallocs != 3 || after.Frees-before.Frees != 3 || after.MallocBytes-before.MallocBytes != after.FreeBytes-before.FreeBytes {
		t.Fatalf("GPU transfer/allocation lifecycle before=%+v after=%+v", before, after)
	}
	freeAfter, _ := nvidia.MemInfo()
	gpuCPU, _ := compare("GPU/CPU fc1", gpu, cpu)
	gpuRust, _ := compare("GPU/Rust fc1", gpu, traces[1])
	t.Logf("ordinary full-width SGEMM diagnostic: GPU/CPU=%.9g GPU/Rust=%.9g (fixed 1e-6 stage gate)", gpuCPU, gpuRust)
	// Candle's CPU projection sums two 512-wide partials. Preserve that
	// partition and the F32 partial-addition order without relaxing the gate.
	const block = 512
	blocked := make([]float32, rows*out)
	for group := 0; group < in/block; group++ {
		a := make([]float32, rows*block)
		weights := make([]float32, block*out)
		for row := 0; row < rows; row++ {
			copy(a[row*block:(row+1)*block], traces[0][row*in+group*block:row*in+(group+1)*block])
		}
		for k := 0; k < block; k++ {
			copy(weights[k*out:(k+1)*out], transposed[(group*block+k)*out:(group*block+k+1)*out])
		}
		partial, err := nvidia.SgemmHost(rows, out, block, 1, a, weights)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range partial {
			blocked[i] += v
		}
	}
	for row := 0; row < rows; row++ {
		for o, bias := range l.bias {
			blocked[row*out+o] += bias
		}
	}
	blockedCPU, changedCPU := compare("blocked GPU/CPU fc1", blocked, cpu)
	blockedRust, changedRust := compare("blocked GPU/Rust fc1", blocked, traces[1])
	closed := nvidia.StatsSnapshot()
	if closed.KernelLaunches-before.KernelLaunches != 3 || closed.HostToDevice-before.HostToDevice != 6 || closed.DeviceToHost-before.DeviceToHost != 3 || closed.Mallocs-before.Mallocs != 9 || closed.Frees-before.Frees != 9 || closed.MallocBytes-before.MallocBytes != closed.FreeBytes-before.FreeBytes {
		t.Fatalf("blocked GPU lifecycle before=%+v after=%+v", before, closed)
	}
	if blockedCPU > 1e-6 || blockedRust > 1e-6 {
		t.Fatalf("blocked GPU fc1 exceeds fixed 1e-6 stage gate CPU=%.9g Rust=%.9g", blockedCPU, blockedRust)
	}
	t.Logf("batched 256x1024 -> 256x4096 fc1: blocked GPU differs CPU=%d Rust=%d values; 3 launches H2D=%d D2H=%d alloc/free=%d bytes; free-before=%d free-after-full-width=%d total=%d (not peak VRAM)", changedCPU, changedRust, closed.HostToDeviceBytes-before.HostToDeviceBytes, closed.DeviceToHostBytes-before.DeviceToHostBytes, closed.MallocBytes-before.MallocBytes, freeBefore, freeAfter, total)
}
