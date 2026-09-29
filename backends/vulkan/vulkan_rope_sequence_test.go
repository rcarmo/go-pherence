package vulkan

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestVulkanRoPESequenceShaderContract(t *testing.T) {
	contract, err := InspectVulkanShader(spirv_rope_sequence_f32)
	if err != nil {
		t.Fatal(err)
	}
	want := VulkanShaderContract{LocalSize: [3]uint32{256, 1, 1}, StorageBindings: 3, PushBytes: 16}
	if contract != want {
		t.Fatalf("shader contract %+v want %+v", contract, want)
	}
	if err := vkCheckShaderLimits(contract, offlineLimits()); err != nil {
		t.Fatal(err)
	}
}

// A bounded maximum-row allocation can be rebound to exact prefix views.
// The attention shader sees only the live keys; no padded row enters softmax.
func TestVulkanRoPEAttentionExactPrefixPlan(t *testing.T) {
	if !VulkanInit() {
		t.Skip("Vulkan unavailable")
	}
	ctx := context.Background()
	before := VulkanMemoryStats()
	arena, err := NewVkTensorArena(ctx, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	alloc := func(rows, width int) *VkTensorF32 {
		v, e := arena.AllocF32(ctx, rows, width)
		if e != nil {
			t.Fatal(e)
		}
		return v
	}
	q, k, v, out := alloc(17, 512), alloc(17, 512), alloc(17, 512), alloc(17, 512)
	freq := alloc(17, 64)
	rope, err := NewVkRoPESequenceF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	attention, err := NewVkAttentionF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	stage := func(rows int) []VkF32Stage {
		t.Helper()
		prefix := func(tensor *VkTensorF32) *VkTensorF32 {
			p, e := tensor.PrefixRows(ctx, rows)
			if e != nil {
				t.Fatal(e)
			}
			return p
		}
		pq, pk, pv, po, pf := prefix(q), prefix(k), prefix(v), prefix(out), prefix(freq)
		rq, e := rope.Stage(ctx, pq, pf, 8)
		if e != nil {
			t.Fatal(e)
		}
		rk, e := rope.Stage(ctx, pk, pf, 8)
		if e != nil {
			t.Fatal(e)
		}
		a, e := attention.Stage(ctx, po, pq, pk, pv, 8)
		if e != nil {
			t.Fatal(e)
		}
		return []VkF32Stage{rq, rk, a}
	}
	plan, err := NewVkF32Plan(ctx, stage(17))
	if err != nil {
		t.Fatal(err)
	}
	inputQ, inputK, inputV := make([]float32, 17*512), make([]float32, 17*512), make([]float32, 17*512)
	frequencies := make([]float32, 17*64)
	for row := 0; row < 17; row++ {
		for head := 0; head < 8; head++ {
			for dim := 0; dim < 64; dim++ {
				i := row*512 + head*64 + dim
				inputQ[i] = float32(((i*7)%101)-50) / 193
				inputK[i] = float32(((i*11)%109)-54) / 211
				inputV[i] = float32(((i*13)%113)-56) / 173
			}
		}
		for dim := 0; dim < 32; dim++ {
			angle := float64(float32(row) * float32(1/math.Pow(10000, float64(2*dim)/64)))
			frequencies[row*64+2*dim] = float32(math.Cos(angle))
			frequencies[row*64+2*dim+1] = float32(math.Sin(angle))
		}
	}
	for _, rows := range []int{17, 5, 11} {
		if err := plan.Rebind(ctx, stage(rows)); err != nil {
			t.Fatal(err)
		}
		prefix := func(tensor *VkTensorF32) *VkTensorF32 {
			view, err := tensor.PrefixRows(ctx, rows)
			if err != nil {
				t.Fatal(err)
			}
			return view
		}
		pq, pk, pv, po, pf := prefix(q), prefix(k), prefix(v), prefix(out), prefix(freq)
		for _, item := range []struct {
			tensor *VkTensorF32
			data   []float32
		}{{pq, inputQ[:rows*512]}, {pk, inputK[:rows*512]}, {pv, inputV[:rows*512]}, {pf, frequencies[:rows*64]}} {
			if err := item.tensor.Upload(ctx, item.data); err != nil {
				t.Fatal(err)
			}
		}
		if err := plan.Run(ctx); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, rows*512)
		if err := po.Download(ctx, got); err != nil {
			t.Fatal(err)
		}
		wantQ, wantK := append([]float32(nil), inputQ[:rows*512]...), append([]float32(nil), inputK[:rows*512]...)
		for row := 0; row < rows; row++ {
			for head := 0; head < 8; head++ {
				base := row*512 + head*64
				for dim := 0; dim < 32; dim++ {
					c, s := frequencies[row*64+2*dim], frequencies[row*64+2*dim+1]
					a, b := wantQ[base+dim], wantQ[base+dim+32]
					wantQ[base+dim], wantQ[base+dim+32] = a*c-b*s, b*c+a*s
					a, b = wantK[base+dim], wantK[base+dim+32]
					wantK[base+dim], wantK[base+dim+32] = a*c-b*s, b*c+a*s
				}
			}
		}
		var maxErr, sumErr float64
		for row := 0; row < rows; row++ {
			for head := 0; head < 8; head++ {
				base := row*512 + head*64
				scores := make([]float64, rows)
				for key := 0; key < rows; key++ {
					kb := key*512 + head*64
					var dot float64
					for dim := 0; dim < 64; dim++ {
						dot += float64(wantQ[base+dim]) * float64(wantK[kb+dim])
					}
					scores[key] = dot / 8
				}
				largest := scores[0]
				for _, score := range scores[1:] {
					largest = math.Max(largest, score)
				}
				var denom float64
				for key := range scores {
					scores[key] = math.Exp(scores[key] - largest)
					denom += scores[key]
				}
				for dim := 0; dim < 64; dim++ {
					var sum float64
					for key, weight := range scores {
						sum += weight * float64(inputV[key*512+head*64+dim])
					}
					i := base + dim
					delta := math.Abs(float64(got[i]) - sum/denom)
					maxErr = math.Max(maxErr, delta)
					sumErr += delta
					if delta > 3e-4+2e-5*math.Abs(sum/denom) {
						t.Fatalf("rows=%d row=%d head=%d dim=%d delta=%g", rows, row, head, dim, delta)
					}
				}
			}
		}
		t.Logf("rows=%d max_abs=%g mean_abs=%g", rows, maxErr, sumErr/float64(len(got)))
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rope.Close(); err != nil {
		t.Fatal(err)
	}
	if err := attention.Close(); err != nil {
		t.Fatal(err)
	}
	if err := arena.Close(); err != nil {
		t.Fatal(err)
	}
	after := VulkanMemoryStats()
	if before.Bytes != after.Bytes || before.Allocations != after.Allocations || after.InFlight || after.Uncertain {
		t.Fatalf("Vulkan resources before=%+v after=%+v", before, after)
	}
}

func TestVulkanRoPESequencePreparedRows(t *testing.T) {
	if !VulkanInit() {
		t.Skip("Vulkan unavailable")
	}
	ctx := context.Background()
	const rows, heads, headDim, half = 17, 8, 64, 32
	width := heads * headDim
	original := make([]float32, rows*width)
	for i := range original {
		original[i] = float32(math.Sin(float64(i) * 0.013))
	}
	freqs := make([]float32, rows*half*2)
	for row := 0; row < rows; row++ {
		for dim := 0; dim < half; dim++ {
			angle := float32(row) * float32(1/math.Pow(10000, float64(2*dim)/headDim))
			freqs[(row*half+dim)*2] = float32(math.Cos(float64(angle)))
			freqs[(row*half+dim)*2+1] = float32(math.Sin(float64(angle)))
		}
	}
	want := append([]float32(nil), original...)
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			for dim := 0; dim < half; dim++ {
				base := (row*heads+head)*headDim + dim
				a, b := want[base], want[base+half]
				c, s := freqs[(row*half+dim)*2], freqs[(row*half+dim)*2+1]
				want[base] = a*c - b*s
				want[base+half] = b*c + a*s
			}
		}
	}
	op, err := NewVkRoPESequenceF32(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer op.Close()
	arena, err := NewVkTensorArena(ctx, 2<<20)
	if err != nil {
		t.Fatal(err)
	}
	defer arena.Close()
	x, err := arena.AllocF32(ctx, rows, width)
	if err != nil {
		t.Fatal(err)
	}
	frequency, err := arena.AllocF32(ctx, rows, half*2)
	if err != nil {
		t.Fatal(err)
	}
	if err := x.Upload(ctx, original); err != nil {
		t.Fatal(err)
	}
	if err := frequency.Upload(ctx, freqs); err != nil {
		t.Fatal(err)
	}
	stage, err := op.Stage(ctx, x, frequency, heads)
	if err != nil {
		t.Fatal(err)
	}
	if stage.Groups[0] != 17 {
		t.Fatalf("groups=%v", stage.Groups)
	}
	plan, err := NewVkF32Plan(ctx, []VkF32Stage{stage})
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if err := plan.Close(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(want))
	if err := x.Download(ctx, got); err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i, value := range got {
		delta := math.Abs(float64(value - want[i]))
		maxAbs = math.Max(maxAbs, delta)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 2e-5 {
			t.Fatalf("row=%d col=%d delta=%g", i/width, i%width, delta)
		}
	}
	t.Logf("time-major RoPE rows=%d heads=%d max_abs=%g", rows, heads, maxAbs)
	if _, err := op.Stage(ctx, x, x, heads); err == nil {
		t.Fatal("accepted aliased frequency table")
	}
}

// Released-weight pre-RoPE Q/K from Nemotron diarization layer 1 exercise the
// same time-major layout consumed by the model, independently of test vectors.
func TestVulkanRoPESequenceReleasedLayer1(t *testing.T) {
	if !VulkanInit() {
		t.Skip("Vulkan unavailable")
	}
	ctx := context.Background()
	const rows, heads, dim, half = 138, 8, 64, 32
	width := heads * dim
	read := func(name string) []float32 {
		t.Helper()
		path := filepath.Join("..", "..", "model", "nemotrondiarization", "testdata", name)
		file, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		gz, err := gzip.NewReader(file)
		if err != nil {
			t.Fatal(err)
		}
		defer gz.Close()
		data, err := io.ReadAll(io.LimitReader(gz, int64(rows*width*4+1)))
		if err != nil || len(data) != rows*width*4 {
			t.Fatalf("fixture %s bytes=%d err=%v", name, len(data), err)
		}
		values := make([]float32, rows*width)
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		}
		return values
	}
	freqs := make([]float32, rows*half*2)
	for row := 0; row < rows; row++ {
		for d := 0; d < half; d++ {
			angle := float32(row) * float32(1/math.Pow(10000, float64(2*d)/dim))
			freqs[(row*half+d)*2] = float32(math.Cos(float64(angle)))
			freqs[(row*half+d)*2+1] = float32(math.Sin(float64(angle)))
		}
	}
	before := VulkanMemoryStats()
	closed := false
	arena, err := NewVkTensorArena(ctx, 2<<20)
	if err != nil {
		t.Fatal(err)
	}
	op, err := NewVkRoPESequenceF32(ctx)
	if err != nil {
		arena.Close()
		t.Fatal(err)
	}
	defer func() {
		if closed {
			return
		}
		if err := op.Close(); err != nil {
			t.Error(err)
		}
		if err := arena.Close(); err != nil {
			t.Error(err)
		}
	}()
	x, err := arena.AllocF32(ctx, rows, width)
	if err != nil {
		t.Fatal(err)
	}
	frequency, err := arena.AllocF32(ctx, rows, half*2)
	if err != nil {
		t.Fatal(err)
	}
	if err := frequency.Upload(ctx, freqs); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"jfk_full_layer1_q.f32.gz", "jfk_full_layer1_k.f32.gz"} {
		input := read(name)
		if err := x.Upload(ctx, input); err != nil {
			t.Fatal(err)
		}
		if err := op.Forward(ctx, x, frequency, heads); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(input))
		if err := x.Download(ctx, got); err != nil {
			t.Fatal(err)
		}
		var maxAbs float64
		for row := 0; row < rows; row++ {
			for head := 0; head < heads; head++ {
				for d := 0; d < half; d++ {
					i := (row*heads+head)*dim + d
					a, b := input[i], input[i+half]
					c, s := freqs[(row*half+d)*2], freqs[(row*half+d)*2+1]
					for _, pair := range [][2]float32{{got[i], a*c - b*s}, {got[i+half], b*c + a*s}} {
						delta := math.Abs(float64(pair[0] - pair[1]))
						maxAbs = math.Max(maxAbs, delta)
						if math.IsNaN(float64(pair[0])) || math.IsInf(float64(pair[0]), 0) || delta > 2e-5 {
							t.Fatalf("fixture=%s row=%d head=%d dim=%d delta=%g", name, row, head, d, delta)
						}
					}
				}
			}
		}
		t.Logf("fixture=%s max_abs=%g", name, maxAbs)
		if root := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_ROPE_REF"); root != "" {
			path := filepath.Join(root, name[:len(name)-len(".f32.gz")]+"_rope.f32.gz")
			file, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			gz, err := gzip.NewReader(file)
			if err != nil {
				file.Close()
				t.Fatal(err)
			}
			data, err := io.ReadAll(io.LimitReader(gz, int64(len(got)*4+1)))
			gz.Close()
			file.Close()
			if err != nil || len(data) != len(got)*4 {
				t.Fatalf("PyTorch RoPE fixture %s bytes=%d err=%v", name, len(data), err)
			}
			var maxReference, sumReference float64
			var outside int
			for i, value := range got {
				want := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
				delta := math.Abs(float64(value - want))
				maxReference = math.Max(maxReference, delta)
				sumReference += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
					outside++
				}
			}
			t.Logf("PyTorch fixture=%s max_abs=%g mean_abs=%g outside=%d", name, maxReference, sumReference/float64(len(got)), outside)
			if outside != 0 || sumReference/float64(len(got)) > 2e-5 {
				t.Fatal("released-weight RoPE differs from pinned PyTorch")
			}
		}
	}
	if err := op.Close(); err != nil {
		t.Fatal(err)
	}
	if err := arena.Close(); err != nil {
		t.Fatal(err)
	}
	closed = true
	after := VulkanMemoryStats()
	if after.Bytes != before.Bytes || after.Allocations != before.Allocations || after.InFlight || after.Uncertain {
		t.Fatalf("Vulkan resources before=%+v after=%+v", before, after)
	}
}
