package omnivoice

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestIsQ8ProjectionName(t *testing.T) {
	for _, tc := range []struct {
		name string
		want bool
	}{
		{name: "llm.layers.0.self_attn.q_proj.weight", want: true},
		{name: "llm.layers.12.self_attn.k_proj.weight", want: true},
		{name: "llm.layers.3.self_attn.v_proj.weight", want: true},
		{name: "llm.layers.1.self_attn.o_proj.weight", want: true},
		{name: "llm.layers.1.mlp.gate_proj.weight", want: true},
		{name: "llm.layers.1.mlp.up_proj.weight", want: true},
		{name: "llm.layers.1.mlp.down_proj.weight", want: true},
		{name: "llm.layers.x.self_attn.q_proj.weight", want: false},
		{name: "llm.layers.1.self_attn.q_norm.weight", want: false},
		{name: "llm.embed_tokens.weight", want: false},
	} {
		if got := IsQ8ProjectionName(tc.name); got != tc.want {
			t.Fatalf("IsQ8ProjectionName(%q)=%v want %v", tc.name, got, tc.want)
		}
	}
}

func TestExportGGUFQ8_0RoundTripSynthetic(t *testing.T) {
	cfg := q8CompatibleConfig(t)
	source := newPatternedWeights(t, cfg)
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := ExportGGUF(context.Background(), source, path, "q8_0"); err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if storage, ok := g.MetaString("omnivoice.storage"); !ok || storage != "q8_0" {
		t.Fatalf("omnivoice.storage=%q ok=%v", storage, ok)
	}
	qtypes := map[string]gguf.QuantType{}
	for _, tensor := range g.Tensors {
		qtypes[tensor.Name] = tensor.QType
	}
	got, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Close()

	for _, name := range source.file.Names() {
		wantRaw, wantDType, wantShape, err := source.file.GetRaw(name)
		if err != nil {
			t.Fatalf("source GetRaw %s: %v", name, err)
		}
		gotRaw, gotDType, gotShape, err := got.file.GetRaw(name)
		if err != nil {
			t.Fatalf("got GetRaw %s: %v", name, err)
		}
		if !reflect.DeepEqual(gotShape, wantShape) {
			t.Fatalf("%s shape=%v want %v", name, gotShape, wantShape)
		}
		if name == "codebook_layer_offsets" {
			if gotDType != "I64" || !bytes.Equal(gotRaw, wantRaw) {
				t.Fatalf("codebook dtype/raw mismatch got=%s", gotDType)
			}
			continue
		}
		if shouldQuantizeProjection(name, wantShape) {
			if gotDType != "Q8_0" {
				t.Fatalf("%s dtype=%s want Q8_0", name, gotDType)
			}
			if qtypes[name] != gguf.QuantQ8_0 {
				t.Fatalf("%s qtype=%s want Q8_0", name, qtypes[name])
			}
			stored, ok := got.QuantizedProjection(name)
			if !ok {
				t.Fatalf("QuantizedProjection(%s)=false", name)
			}
			expectedRaw, err := quantizeQ8_0Tensor(context.Background(), name, wantRaw, wantDType, wantShape)
			if err != nil {
				t.Fatalf("expected quantized raw %s: %v", name, err)
			}
			if !bytes.Equal(stored, expectedRaw) || !bytes.Equal(gotRaw, expectedRaw) {
				t.Fatalf("%s quantized raw mismatch", name)
			}
		} else {
			if gotDType != "F32" {
				t.Fatalf("%s dtype=%s want F32", name, gotDType)
			}
			if qtypes[name] != gguf.QuantF32 {
				t.Fatalf("%s qtype=%s want F32", name, qtypes[name])
			}
			if raw, ok := got.QuantizedProjection(name); ok || raw != nil {
				t.Fatalf("QuantizedProjection(%s) unexpectedly succeeded", name)
			}
		}
		wantF32, wantF32Shape, err := source.file.GetFloat32(name)
		if err != nil {
			t.Fatalf("source GetFloat32 %s: %v", name, err)
		}
		gotF32, gotF32Shape, err := got.file.GetFloat32(name)
		if err != nil {
			t.Fatalf("got GetFloat32 %s: %v", name, err)
		}
		if !reflect.DeepEqual(gotF32Shape, wantF32Shape) {
			t.Fatalf("%s float32 shape=%v want %v", name, gotF32Shape, wantF32Shape)
		}
		if shouldQuantizeProjection(name, wantShape) {
			expectedRaw, err := quantizeQ8_0Tensor(context.Background(), name, wantRaw, wantDType, wantShape)
			if err != nil {
				t.Fatalf("expected quantized raw %s: %v", name, err)
			}
			wantQ := make([]float32, len(gotF32))
			if err := dequantQ8_0Into(wantQ, expectedRaw); err != nil {
				t.Fatalf("dequant expected %s: %v", name, err)
			}
			if !reflect.DeepEqual(gotF32, wantQ) {
				t.Fatalf("%s dequant mismatch", name)
			}
		} else if !reflect.DeepEqual(gotF32, wantF32) {
			t.Fatalf("%s non-quant float mismatch", name)
		}
	}

	buf, err := got.NewLayerBuffer()
	if err != nil {
		t.Fatal(err)
	}
	if err := buf.Load(got, 0); err != nil {
		t.Fatal(err)
	}
	allocs := testing.AllocsPerRun(50, func() {
		if err := buf.Load(got, 0); err != nil {
			panic(err)
		}
	})
	if allocs != 0 {
		t.Fatalf("LayerBuffer.Load allocs=%g", allocs)
	}
	wantLayerQProj, _, err := got.file.GetFloat32("llm.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(buf.Tensors["self_attn.q_proj.weight"], wantLayerQProj) {
		t.Fatal("layer buffer q_proj mismatch")
	}
	row := make([]float32, cfg.LLMConfig.HiddenSize)
	if err := got.MatrixRowsInto(row, "llm.embed_tokens.weight", 0, 1); err != nil {
		t.Fatalf("MatrixRowsInto embed_tokens: %v", err)
	}
	if err := got.MatrixRowsInto(row, "llm.layers.0.self_attn.q_proj.weight", 0, 1); err == nil || !strings.Contains(err.Error(), "does not support quantized tensor") {
		t.Fatalf("MatrixRowsInto q_proj err=%v", err)
	}
	infos := got.file.TensorInfos()
	if infos["llm.layers.0.self_attn.q_proj.weight"].DType != "Q8_0" {
		t.Fatalf("TensorInfos q_proj dtype=%s want Q8_0", infos["llm.layers.0.self_attn.q_proj.weight"].DType)
	}
}

func TestExportGGUFQ8_0RetainsTinyProjectionMatricesAsF32(t *testing.T) {
	cfg := sampleConfig(t)
	source := newPatternedWeights(t, cfg)
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := ExportGGUF(context.Background(), source, path, "q8_0"); err != nil {
		t.Fatal(err)
	}
	got, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer got.Close()
	for _, name := range source.file.Names() {
		if !IsQ8ProjectionName(name) {
			continue
		}
		raw, dtype, shape, err := got.file.GetRaw(name)
		if err != nil {
			t.Fatalf("GetRaw %s: %v", name, err)
		}
		if shape[1]%q8_0BlockElems == 0 {
			t.Fatalf("test fixture unexpectedly quantizable for %s shape=%v", name, shape)
		}
		if dtype != "F32" {
			t.Fatalf("%s dtype=%s want F32", name, dtype)
		}
		if qraw, ok := got.QuantizedProjection(name); ok || qraw != nil {
			t.Fatalf("%s unexpectedly reported quantized projection", name)
		}
		wantF32, _, err := source.file.GetFloat32(name)
		if err != nil {
			t.Fatal(err)
		}
		gotF32, _, err := got.file.GetFloat32(name)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(gotF32, wantF32) {
			t.Fatalf("%s float mismatch", name)
		}
		if len(raw) != len(wantF32)*4 {
			t.Fatalf("%s raw len=%d want %d", name, len(raw), len(wantF32)*4)
		}
	}
}

func TestQuantizeQ8_0TensorRejectsNonFiniteExtremeAndCanceled(t *testing.T) {
	shape := []int{1, q8_0BlockElems}
	mkRaw := func(v float32) []byte {
		raw := make([]byte, len(shape)*shape[1]*4/len(shape))
		for i := 0; i < shape[1]; i++ {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		return raw
	}
	if _, err := quantizeQ8_0Tensor(context.Background(), "llm.layers.0.self_attn.q_proj.weight", mkRaw(float32(math.NaN())), "F32", shape); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("NaN err=%v", err)
	}
	if _, err := quantizeQ8_0Tensor(context.Background(), "llm.layers.0.self_attn.q_proj.weight", mkRaw(float32(math.Inf(1))), "F32", shape); err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("Inf err=%v", err)
	}
	if _, err := quantizeQ8_0Tensor(context.Background(), "llm.layers.0.self_attn.q_proj.weight", mkRaw(math.MaxFloat32), "F32", shape); err == nil || !strings.Contains(err.Error(), "stored scale out of range") {
		t.Fatalf("extreme err=%v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := quantizeQ8_0Tensor(ctx, "llm.layers.0.self_attn.q_proj.weight", mkRaw(1), "F32", shape); err == nil {
		t.Fatal("canceled context accepted")
	}
}

func TestOpenGGUFWeightsRejectsInvalidQ8ProjectionUse(t *testing.T) {
	cfg := q8CompatibleConfig(t)
	meta, tensors := buildTinyOmniVoiceGGUFFixture(t, cfg)

	t.Run("non projection q8", func(t *testing.T) {
		bad := cloneGGUFTensors(tensors)
		for i := range bad {
			if bad[i].spec.Name == "llm.embed_tokens.weight" {
				bad[i].spec.QType = gguf.QuantQ8_0
				bad[i].raw = make([]byte, (32*cfg.LLMConfig.VocabSize/q8_0BlockElems)*q8_0BlockBytes)
				break
			}
		}
		path := filepath.Join(t.TempDir(), "embed-q8.gguf")
		writeGGUFFixture(t, path, meta, bad)
		assertOpenGGUFWeightsError(t, path, "not an allowed decoder projection")
	})

	t.Run("misaligned q8 row width", func(t *testing.T) {
		bad := cloneGGUFTensors(tensors)
		for i := range bad {
			if bad[i].spec.Name == "llm.layers.0.self_attn.q_proj.weight" {
				bad[i].spec.QType = gguf.QuantQ8_0
				bad[i].spec.Shape = []uint64{31, bad[i].spec.Shape[1]}
				bad[i].raw = make([]byte, int((31*bad[i].spec.Shape[1]/q8_0BlockElems)*q8_0BlockBytes))
				break
			}
		}
		path := filepath.Join(t.TempDir(), "misaligned-q8.gguf")
		writeGGUFFixture(t, path, meta, bad)
		assertOpenGGUFWeightsError(t, path, "innermost dimension 31 is not a multiple of 32")
	})
}

type fakeTensorReader struct {
	names []string
	infos map[string]safetensors.TensorInfo
	raws  map[string][]byte
}

func (f *fakeTensorReader) Close() error { return nil }

func (f *fakeTensorReader) Names() []string {
	out := make([]string, len(f.names))
	copy(out, f.names)
	return out
}

func (f *fakeTensorReader) TensorInfos() map[string]safetensors.TensorInfo {
	out := make(map[string]safetensors.TensorInfo, len(f.infos))
	for name, info := range f.infos {
		out[name] = safetensors.TensorInfo{DType: info.DType, Shape: append([]int(nil), info.Shape...), DataOffsets: info.DataOffsets}
	}
	return out
}

func (f *fakeTensorReader) GetRaw(name string) ([]byte, string, []int, error) {
	info, ok := f.infos[name]
	if !ok {
		return nil, "", nil, fmt.Errorf("missing tensor %s", name)
	}
	raw := f.raws[name]
	return raw, info.DType, info.Shape, nil
}

func (f *fakeTensorReader) GetFloat32(name string) ([]float32, []int, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		return nil, nil, err
	}
	if dtype == "I64" {
		out := make([]float32, len(raw)/8)
		for i := range out {
			out[i] = float32(int64(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, shape, nil
	}
	count, err := tensorElementCount(shape)
	if err != nil {
		return nil, nil, err
	}
	out := make([]float32, count)
	if err := convertInto(out, raw, dtype); err != nil {
		return nil, nil, err
	}
	return out, shape, nil
}

func q8CompatibleConfig(t *testing.T) Config {
	t.Helper()
	cfg := sampleConfig(t)
	cfg.LLMConfig.HiddenSize = 32
	cfg.LLMConfig.HeadDim = 32
	cfg.LLMConfig.NumAttentionHeads = 1
	cfg.LLMConfig.NumKeyValueHeads = 1
	cfg.LLMConfig.IntermediateSize = 64
	cfg.LLMConfig.VocabSize = 64
	if err := cfg.Validate(); err != nil {
		t.Fatalf("cfg.Validate: %v", err)
	}
	return cfg
}

func newPatternedWeights(t *testing.T, cfg Config) *Weights {
	t.Helper()
	specs, err := expectedTensorSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(specs))
	for name := range specs {
		names = append(names, name)
	}
	sort.Strings(names)
	infos := make(map[string]safetensors.TensorInfo, len(specs))
	raws := make(map[string][]byte, len(specs))
	for i, name := range names {
		spec := specs[name]
		shape := int64sToInts(t, spec.Shape)
		if name == "codebook_layer_offsets" {
			raw, info, err := synthesizeCodebookOffsets(cfg)
			if err != nil {
				t.Fatal(err)
			}
			infos[name] = info
			raws[name] = raw
			continue
		}
		dtype := patternedTensorDType(name)
		vals := patternedTensorValues(t, shape, i+1)
		info := safetensors.TensorInfo{DType: dtype, Shape: shape}
		raw := encodePatternedTensorRaw(vals, dtype)
		info.DataOffsets = [2]int{0, len(raw)}
		infos[name] = info
		raws[name] = raw
	}
	return &Weights{file: &fakeTensorReader{names: names, infos: infos, raws: raws}, Config: cfg}
}

func patternedTensorDType(name string) string {
	switch {
	case IsQ8ProjectionName(name):
		return "F16"
	case name == "audio_embeddings.weight":
		return "BF16"
	case name == "llm.embed_tokens.weight":
		return "F32"
	case strings.HasSuffix(name, ".weight") && strings.Contains(name, "norm"):
		return "F32"
	default:
		return "F16"
	}
}

func patternedTensorValues(t *testing.T, shape []int, seed int) []float32 {
	t.Helper()
	count, err := tensorElementCount(shape)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, count)
	base := float32(seed%9) - 4
	for i := range out {
		step := float32((i%23)-11) * 0.0625
		v := base + step
		if i%2 == 1 {
			v = -v
		}
		if i%17 == 0 {
			v = 0
		}
		out[i] = v
	}
	return out
}

func encodePatternedTensorRaw(vals []float32, dtype string) []byte {
	switch dtype {
	case "F16":
		raw := make([]byte, len(vals)*2)
		for i, v := range vals {
			binary.LittleEndian.PutUint16(raw[i*2:], half.F32ToF16(v))
		}
		return raw
	case "BF16":
		raw := make([]byte, len(vals)*2)
		for i, v := range vals {
			binary.LittleEndian.PutUint16(raw[i*2:], uint16(math.Float32bits(v)>>16))
		}
		return raw
	case "F32":
		raw := make([]byte, len(vals)*4)
		for i, v := range vals {
			binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
		}
		return raw
	default:
		panic("unsupported dtype " + dtype)
	}
}

func TestGGUFQ8RejectsNonFiniteStoredScale(t *testing.T) {
	w := newPatternedWeights(t, q8CompatibleConfig(t))
	path := filepath.Join(t.TempDir(), "nonfinite.gguf")
	if err := ExportGGUF(context.Background(), w, path, "q8_0"); err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	offset := int64(-1)
	for _, tensor := range g.Tensors {
		if tensor.QType == gguf.QuantQ8_0 {
			offset = g.DataOffset + int64(tensor.Offset)
			break
		}
	}
	g.Close()
	if offset < 0 {
		t.Fatal("missing Q8 tensor")
	}
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.WriteAt([]byte{0, 0x7c}, offset)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := OpenWeights(path); err == nil {
		t.Fatal("nonfinite scale accepted")
	}
}
