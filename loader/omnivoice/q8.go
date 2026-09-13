package omnivoice

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

const (
	q8_0BlockElems = 32
	q8_0BlockBytes = 34
)

func IsQ8ProjectionName(name string) bool {
	const prefix = "llm.layers."
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	rest := name[len(prefix):]
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 {
		return false
	}
	if _, err := strconv.Atoi(rest[:dot]); err != nil {
		return false
	}
	switch rest[dot+1:] {
	case "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight":
		return true
	default:
		return false
	}
}

func exportQTypeForTensor(format, name string, shape []int) (gguf.QuantType, error) {
	switch format {
	case "f16":
		return gguf.QuantF16, nil
	case "f32":
		return gguf.QuantF32, nil
	case "q8_0":
		if shouldQuantizeProjection(name, shape) {
			return gguf.QuantQ8_0, nil
		}
		return gguf.QuantF32, nil
	default:
		return 0, fmt.Errorf("omnivoice: GGUF format must be f16, f32, or q8_0")
	}
}

func shouldQuantizeProjection(name string, shape []int) bool {
	return IsQ8ProjectionName(name) && len(shape) == 2 && shape[0] > 0 && shape[1] >= q8_0BlockElems && shape[1]%q8_0BlockElems == 0
}

func validateQ8ProjectionTensor(name string, shape []uint64) error {
	if !IsQ8ProjectionName(name) {
		return fmt.Errorf("Q8_0 tensor %q is not an allowed decoder projection", name)
	}
	if len(shape) != 2 {
		return fmt.Errorf("Q8_0 tensor %q shape %v is not a matrix", name, shape)
	}
	if shape[0] == 0 || shape[1] == 0 {
		return fmt.Errorf("Q8_0 tensor %q shape %v has zero dimension", name, shape)
	}
	if shape[0]%q8_0BlockElems != 0 {
		return fmt.Errorf("Q8_0 tensor %q innermost dimension %d is not a multiple of %d", name, shape[0], q8_0BlockElems)
	}
	return nil
}

func validationDTypeForTensor(dtype string) string {
	if dtype == "Q8_0" {
		return "F32"
	}
	return dtype
}

func floatDTypeWidth(dtype string) (int, error) {
	switch dtype {
	case "F16", "BF16":
		return 2, nil
	case "F32":
		return 4, nil
	default:
		return 0, fmt.Errorf("omnivoice: unsupported float dtype %s", dtype)
	}
}

func tensorElementCount(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("omnivoice: empty tensor shape")
	}
	n := 1
	for _, dim := range shape {
		if dim <= 0 || n > int(^uint(0)>>1)/dim {
			return 0, fmt.Errorf("omnivoice: invalid tensor shape %v", shape)
		}
		n *= dim
	}
	return n, nil
}

func quantizeQ8_0Tensor(ctx context.Context, name string, raw []byte, dtype string, shape []int) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !shouldQuantizeProjection(name, shape) {
		return nil, fmt.Errorf("omnivoice: tensor %q is not eligible for Q8_0 export", name)
	}
	width, err := floatDTypeWidth(dtype)
	if err != nil {
		return nil, err
	}
	n, err := tensorElementCount(shape)
	if err != nil {
		return nil, err
	}
	if n > int(^uint(0)>>1)/width {
		return nil, fmt.Errorf("omnivoice: tensor %q raw size overflow", name)
	}
	rows, cols := shape[0], shape[1]
	want := n * width
	if len(raw) != want {
		return nil, fmt.Errorf("omnivoice: tensor %q raw length %d, want %d", name, len(raw), want)
	}
	rowBytes, err := gguf.TensorRawBytes(gguf.QuantQ8_0, cols)
	if err != nil {
		return nil, fmt.Errorf("omnivoice: tensor %q: %w", name, err)
	}
	if rows > int(^uint(0)>>1)/rowBytes {
		return nil, fmt.Errorf("omnivoice: tensor %q Q8_0 size overflow", name)
	}
	out := make([]byte, rows*rowBytes)
	var block [q8_0BlockElems]float32
	checks := 0
	for row := 0; row < rows; row++ {
		rowBase := row * cols * width
		outBase := row * rowBytes
		for col := 0; col < cols; col += q8_0BlockElems {
			if checks&127 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			checks++
			for i := 0; i < q8_0BlockElems; i++ {
				v, err := decodeTensorFloat(raw[rowBase+(col+i)*width:], dtype)
				if err != nil {
					return nil, fmt.Errorf("omnivoice: tensor %q row %d col %d: %w", name, row, col+i, err)
				}
				block[i] = v
			}
			if err := encodeQ8_0Block(out[outBase+(col/q8_0BlockElems)*q8_0BlockBytes:], block[:]); err != nil {
				return nil, fmt.Errorf("omnivoice: tensor %q row %d block %d: %w", name, row, col/q8_0BlockElems, err)
			}
		}
	}
	return out, nil
}

func decodeTensorFloat(raw []byte, dtype string) (float32, error) {
	switch dtype {
	case "F32":
		if len(raw) < 4 {
			return 0, fmt.Errorf("short F32 value")
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(raw[:4])), nil
	case "F16":
		if len(raw) < 2 {
			return 0, fmt.Errorf("short F16 value")
		}
		return half.F16ToF32(binary.LittleEndian.Uint16(raw[:2])), nil
	case "BF16":
		if len(raw) < 2 {
			return 0, fmt.Errorf("short BF16 value")
		}
		return half.BF16ToF32(binary.LittleEndian.Uint16(raw[:2])), nil
	default:
		return 0, fmt.Errorf("unsupported float dtype %s", dtype)
	}
}

func encodeQ8_0Block(dst []byte, vals []float32) error {
	if len(dst) < q8_0BlockBytes {
		return fmt.Errorf("short Q8_0 destination")
	}
	if len(vals) != q8_0BlockElems {
		return fmt.Errorf("Q8_0 block has %d values, want %d", len(vals), q8_0BlockElems)
	}
	var amax float32
	for i, v := range vals {
		if !isFiniteFloat32(v) {
			return fmt.Errorf("non-finite value at %d", i)
		}
		av := float32(math.Abs(float64(v)))
		if av > amax {
			amax = av
		}
	}
	if amax == 0 {
		binary.LittleEndian.PutUint16(dst[:2], 0)
		for i := 2; i < q8_0BlockBytes; i++ {
			dst[i] = 0
		}
		return nil
	}
	d := amax / 127
	if !isFiniteFloat32(d) || d == 0 {
		return fmt.Errorf("Q8_0 scale out of range")
	}
	dh := half.F32ToF16(d)
	dStored := half.F16ToF32(dh)
	if !isFiniteFloat32(dStored) || dStored == 0 {
		return fmt.Errorf("Q8_0 stored scale out of range")
	}
	binary.LittleEndian.PutUint16(dst[:2], dh)
	id := float32(1) / d
	for i, v := range vals {
		q := int(math.Round(float64(v * id)))
		if q > 127 {
			q = 127
		}
		if q < -127 {
			q = -127
		}
		dst[2+i] = byte(int8(q))
	}
	return nil
}

func dequantQ8_0Into(dst []float32, raw []byte) error {
	if len(dst)%q8_0BlockElems != 0 {
		return fmt.Errorf("omnivoice: Q8_0 tensor len=%d not multiple of %d", len(dst), q8_0BlockElems)
	}
	want := len(dst) / q8_0BlockElems * q8_0BlockBytes
	if len(raw) != want {
		return fmt.Errorf("omnivoice: Q8_0 raw length %d, want %d", len(raw), want)
	}
	for b := 0; b < len(dst)/q8_0BlockElems; b++ {
		blk := raw[b*q8_0BlockBytes:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[:2]))
		base := b * q8_0BlockElems
		for i := 0; i < q8_0BlockElems; i++ {
			dst[base+i] = d * float32(int8(blk[2+i]))
		}
	}
	return nil
}

func isFiniteFloat32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

func (w *Weights) QuantizedProjection(name string) ([]byte, bool) {
	if w == nil || w.file == nil || !IsQ8ProjectionName(name) {
		return nil, false
	}
	reader, ok := w.file.(interface{ QuantizedProjection(string) ([]byte, bool) })
	if !ok {
		return nil, false
	}
	return reader.QuantizedProjection(name)
}
