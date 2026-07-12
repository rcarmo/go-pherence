package diffusiongemma

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestRunDenseMLPBatchParityWithRetainedQ4K(t *testing.T) {
	const (
		layer        = 0
		hiddenSize   = 256
		intermediate = 256
		positions    = 3
	)
	weights := newSyntheticBatchParityWeights(t, hiddenSize, layer)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.mlp.gate_proj.weight", gguf.QuantQ4_K, hiddenSize, intermediate)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.mlp.up_proj.weight", gguf.QuantQ4_K, hiddenSize, intermediate)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.mlp.down_proj.weight", gguf.QuantQ4_K, intermediate, hiddenSize)
	weights.addVector("model.decoder.layers.0.post_feedforward_layernorm_1.weight", makePattern(1, hiddenSize, 0.9, 0.0025))
	weights.bindLayerDenseMLP(layer)

	input := makePattern(7, positions*hiddenSize, -0.2, 0.01)
	residual := makePattern(11, positions*hiddenSize, 0.15, -0.007)
	got := ForwardScratch{Hidden: append([]float32(nil), input...), Residual: append([]float32(nil), residual...), MlpOut: make([]float32, len(input))}
	want := ForwardScratch{Hidden: append([]float32(nil), input...), Residual: append([]float32(nil), residual...), MlpOut: make([]float32, len(input))}
	op := LayerOp{Layer: layer, Kind: OpDenseMLP}
	if err := runDenseMLP(op, weights.TextWeights, got); err != nil {
		t.Fatal(err)
	}
	fallback := weights.withoutQuant()
	if err := runDenseMLP(op, fallback, want); err != nil {
		t.Fatal(err)
	}
	assertCloseSlice(t, got.MlpOut, want.MlpOut, 5e-2)
}

func TestRunSelfAttentionBatchParityWithRetainedQ4K(t *testing.T) {
	const (
		layer      = 0
		hiddenSize = 256
		positions  = 3
		headDim    = 64
	)
	weights := newSyntheticBatchParityWeights(t, hiddenSize, layer)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.self_attn.q_proj.weight", gguf.QuantQ4_K, hiddenSize, hiddenSize)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.self_attn.k_proj.weight", gguf.QuantQ4_K, hiddenSize, hiddenSize)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.self_attn.v_proj.weight", gguf.QuantQ4_K, hiddenSize, hiddenSize)
	weights.addQuantizedMatrix(t, "model.decoder.layers.0.self_attn.o_proj.weight", gguf.QuantQ4_K, hiddenSize, hiddenSize)
	weights.addVector("model.decoder.layers.0.self_attn.q_norm.weight", makePattern(3, headDim, 0.95, 0.001))
	weights.addVector("model.decoder.layers.0.self_attn.k_norm.weight", makePattern(5, headDim, 1.05, -0.0015))
	weights.bindLayerAttention(layer)

	input := makePattern(13, positions*hiddenSize, 0.05, 0.008)
	got := ForwardScratch{Hidden: append([]float32(nil), input...), SlidingWindow: 32}
	want := ForwardScratch{Hidden: append([]float32(nil), input...), SlidingWindow: 32}
	ctx := ForwardContext{EncoderSeqLen: 0}
	op := LayerOp{Layer: layer, Type: "sliding_attention", Kind: OpSelfAttention}
	if err := runSelfAttention(op, ctx, weights.TextWeights, got); err != nil {
		t.Fatal(err)
	}
	fallback := weights.withoutQuant()
	if err := runSelfAttention(op, ctx, fallback, want); err != nil {
		t.Fatal(err)
	}
	assertCloseSlice(t, got.Hidden, want.Hidden, 8e-2)
}

func TestMixedMatrixProjectBatchToPropagatesSupportedQuantErrors(t *testing.T) {
	m := &mixedMatrix{
		quant: &gguf.QuantMatrix{Name: "broken.q4k", QType: gguf.QuantQ4_K, InDim: 255, OutDim: 8, Raw: make([]byte, 8)},
		f32:   make([]float32, 255*8),
		rows:  8,
		cols:  255,
	}
	ok, err := m.projectBatchTo(make([]float32, 16), make([]float32, 2*255), 2)
	if ok {
		t.Fatal("projectBatchTo unexpectedly reported success")
	}
	if err == nil || err.Error() == "" {
		t.Fatal("projectBatchTo swallowed supported quant error")
	}
}

type syntheticBatchParityWeights struct {
	*TextWeights
	hiddenSize int
}

func newSyntheticBatchParityWeights(t testing.TB, hiddenSize, layer int) *syntheticBatchParityWeights {
	t.Helper()
	return &syntheticBatchParityWeights{TextWeights: &TextWeights{floatCache: map[string]FloatTensor{}, ggufQuant: map[string]*gguf.QuantMatrix{}, Layers: []LayerWeights{{Layer: layer}}}, hiddenSize: hiddenSize}
}

func (w *syntheticBatchParityWeights) addVector(name string, data []float32) {
	w.floatCache[name] = FloatTensor{Data: append([]float32(nil), data...), Shape: []int{len(data)}, DType: "F32"}
}

func (w *syntheticBatchParityWeights) addQuantizedMatrix(t testing.TB, name string, qtype gguf.QuantType, inDim, outDim int) {
	t.Helper()
	qm := syntheticQuantMatrix(t, qtype, inDim, outDim)
	f32, err := gguf.DequantToF32(qm.Raw, qm.QType, qm.InDim*qm.OutDim)
	if err != nil {
		t.Fatal(err)
	}
	w.floatCache[name] = FloatTensor{Data: f32, Shape: []int{outDim, inDim}, DType: "F32"}
	w.ggufQuant[name] = qm
}

func (w *syntheticBatchParityWeights) bindLayerDenseMLP(layer int) {
	w.Layers[0].Bindings = []TensorBinding{
		{TensorHandle: TensorHandle{Name: matrixName(layer, "mlp.gate_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "mlp.up_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "mlp.down_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "post_feedforward_layernorm_1.weight")}, Shape: []int{w.hiddenSize}, DType: "F32"},
	}
}

func (w *syntheticBatchParityWeights) bindLayerAttention(layer int) {
	w.Layers[0].Bindings = []TensorBinding{
		{TensorHandle: TensorHandle{Name: matrixName(layer, "self_attn.q_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "self_attn.k_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "self_attn.v_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "self_attn.o_proj.weight")}, Shape: []int{w.hiddenSize, w.hiddenSize}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "self_attn.q_norm.weight")}, Shape: []int{64}, DType: "F32"},
		{TensorHandle: TensorHandle{Name: matrixName(layer, "self_attn.k_norm.weight")}, Shape: []int{64}, DType: "F32"},
	}
}

func (w *syntheticBatchParityWeights) withoutQuant() *TextWeights {
	out := &TextWeights{floatCache: map[string]FloatTensor{}, ggufQuant: map[string]*gguf.QuantMatrix{}, Layers: make([]LayerWeights, len(w.Layers))}
	for k, v := range w.floatCache {
		out.floatCache[k] = FloatTensor{Data: append([]float32(nil), v.Data...), Shape: append([]int(nil), v.Shape...), DType: v.DType}
	}
	copy(out.Layers, w.Layers)
	for i := range out.Layers {
		out.Layers[i].Bindings = append([]TensorBinding(nil), w.Layers[i].Bindings...)
	}
	return out
}

func matrixName(layer int, suffix string) string {
	return "model.decoder.layers." + string(rune('0'+layer)) + "." + suffix
}

func makePattern(seed, n int, base, step float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = base + float32(((i+seed)%23)-11)*step
	}
	return out
}

func assertCloseSlice(t testing.TB, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("idx=%d got=%g want=%g diff=%g tol=%g", i, got[i], want[i], got[i]-want[i], tol)
		}
	}
}

func syntheticQuantMatrix(t testing.TB, qtype gguf.QuantType, inDim, outDim int) *gguf.QuantMatrix {
	t.Helper()
	m := &gguf.QuantMatrix{Name: "synthetic", QType: qtype, InDim: inDim, OutDim: outDim}
	rowBytes, err := m.RowBytes()
	if err != nil {
		t.Fatal(err)
	}
	m.Raw = make([]byte, rowBytes*outDim)
	for r := 0; r < outDim; r++ {
		row := m.Raw[r*rowBytes : (r+1)*rowBytes]
		switch qtype {
		case gguf.QuantQ4_K:
			for b := 0; b < inDim/256; b++ {
				blk := row[b*144 : (b+1)*144]
				binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(r+b)*0.002))
				binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32((r+b)%3)*0.001))
				for i := 0; i < 12; i++ {
					blk[4+i] = byte(1 + (i+r+b)%17)
				}
				for i := 0; i < 128; i++ {
					blk[16+i] = byte((i*7 + r*11 + b*13) & 0xff)
				}
			}
		default:
			t.Fatalf("unsupported qtype %v", qtype)
		}
	}
	return m
}
