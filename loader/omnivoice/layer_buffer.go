package omnivoice

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
)

// LayerBuffer holds one layer in a reusable float32 arena. A block referencing
// Tensors must finish execution before Load overwrites that layer's weights.
// It is not safe for concurrent load/execution. Names are precomputed once.
type LayerBuffer struct {
	Tensors  map[string][]float32
	arena    []float32
	names    [][]string
	suffixes []string
}

func (w *Weights) NewLayerBuffer() (*LayerBuffer, error) {
	shapes := ExpectedShapes(w.Config)
	suffixes := []string{"input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
	total := 0
	sizes := make([]int, len(suffixes))
	for i, name := range suffixes {
		shape := shapes["llm.layers.0."+name]
		n := int64(1)
		for _, dim := range shape {
			if dim <= 0 || n > int64(int(^uint(0)>>1)/4)/dim {
				return nil, fmt.Errorf("omnivoice: layer weight arena overflow")
			}
			n *= dim
		}
		if total > int(^uint(0)>>1)/4-int(n) {
			return nil, fmt.Errorf("omnivoice: layer weight arena overflow")
		}
		sizes[i] = int(n)
		total += int(n)
	}
	b := &LayerBuffer{Tensors: make(map[string][]float32, len(suffixes)), arena: make([]float32, total), suffixes: suffixes, names: make([][]string, w.Config.LLMConfig.NumHiddenLayers)}
	offset := 0
	for i, name := range suffixes {
		n := sizes[i]
		b.Tensors[name] = b.arena[offset : offset+n : offset+n]
		offset += n
	}
	for layer := range b.names {
		b.names[layer] = make([]string, len(suffixes))
		for i, suffix := range suffixes {
			b.names[layer][i] = "llm.layers." + strconv.Itoa(layer) + "." + suffix
		}
	}
	return b, nil
}
func (b *LayerBuffer) Bytes() int { return len(b.arena) * 4 }

// Load converts the requested layer in place without allocating on success.
// On failure contents are unspecified: do not run the block until Load succeeds.
func (b *LayerBuffer) Load(w *Weights, index int) error {
	if index < 0 || index >= len(b.names) {
		return fmt.Errorf("omnivoice: layer index out of bounds")
	}
	for i, name := range b.names[index] {
		raw, dtype, _, err := w.file.GetRaw(name)
		if err != nil {
			return err
		}
		if err = convertInto(b.Tensors[b.suffixes[i]], raw, dtype); err != nil {
			return err
		}
	}
	return nil
}
func convertInto(dst []float32, raw []byte, dtype string) error {
	size := 2
	if dtype == "F32" {
		size = 4
	}
	if len(raw)%size != 0 || len(raw)/size != len(dst) {
		return fmt.Errorf("omnivoice: weight conversion length mismatch")
	}
	switch dtype {
	case "F16":
		if !simd.F16LittleEndianToF32(dst, raw) {
			return fmt.Errorf("omnivoice: weight conversion length mismatch")
		}
	case "BF16":
		for i := range dst {
			dst[i] = half.BF16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "F32":
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	default:
		return fmt.Errorf("omnivoice: unsupported float dtype %s", dtype)
	}
	return nil
}
