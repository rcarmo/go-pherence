package ideogram4

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
)

// RawTensorSource abstracts a (sharded) safetensors file: it returns raw bytes,
// the safetensors dtype string, and the shape for a tensor name. Both
// *safetensors.File and *safetensors.ShardedFile satisfy this via GetRaw.
type RawTensorSource interface {
	GetRaw(name string) ([]byte, string, []int, error)
}

// LoadFP8Linear materializes a single FP8Linear from a tensor source using the
// `<prefix>.weight` (E4M3 bytes) and `<prefix>.weight_scale` tensors named by
// the spec. The scale tensor may be F32 (per-tensor scalar or per-row) or F8.
func LoadFP8Linear(src RawTensorSource, spec LinearSpec) (*FP8Linear, error) {
	if src == nil {
		return nil, fmt.Errorf("ideogram4 fp8 load %q: nil tensor source", spec.Prefix)
	}
	wbytes, wdtype, wshape, err := src.GetRaw(spec.Weight)
	if err != nil {
		return nil, fmt.Errorf("ideogram4 fp8 weight %q: %w", spec.Weight, err)
	}
	if !isFP8DType(wdtype) {
		return nil, fmt.Errorf("ideogram4 fp8 weight %q dtype=%s want F8_E4M3*", spec.Weight, wdtype)
	}
	if err := checkLinearShape(spec, wshape); err != nil {
		return nil, err
	}
	sbytes, sdtype, sshape, err := src.GetRaw(spec.WeightScale)
	if err != nil {
		return nil, fmt.Errorf("ideogram4 fp8 scale %q: %w", spec.WeightScale, err)
	}
	scale, err := decodeScale(spec, sbytes, sdtype, sshape)
	if err != nil {
		return nil, err
	}
	var bias []float32
	if bb, bd, _, berr := src.GetRaw(spec.Prefix + ".bias"); berr == nil {
		bias, err = decodeFloatVec(bb, bd, spec.OutDim)
		if err != nil {
			return nil, fmt.Errorf("ideogram4 fp8 bias %q: %w", spec.Prefix, err)
		}
	}
	return NewFP8Linear(spec, wbytes, scale, bias)
}

// decodeFloatVec decodes a small F32/F16/BF16 vector of length n.
func decodeFloatVec(b []byte, dtype string, n int) ([]float32, error) {
	out := make([]float32, n)
	switch dtype {
	case "F32":
		if len(b) != n*4 {
			return nil, fmt.Errorf("vec bytes=%d want=%d", len(b), n*4)
		}
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
	case "BF16":
		if len(b) != n*2 {
			return nil, fmt.Errorf("vec bytes=%d want=%d", len(b), n*2)
		}
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[i*2:])) << 16)
		}
	case "F16":
		if len(b) != n*2 {
			return nil, fmt.Errorf("vec bytes=%d want=%d", len(b), n*2)
		}
		for i := 0; i < n; i++ {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(b[i*2:]))
		}
	default:
		return nil, fmt.Errorf("unsupported vec dtype %s", dtype)
	}
	return out, nil
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	}
	return math.Float32frombits(sign | ((exp + 112) << 23) | (mant << 13))
}

// LoadLayerFP8Linears loads every required FP8 linear for the transformer
// (globals plus all per-layer matrices) from the source, returning them keyed
// by prefix. Missing tensors are reported with their prefix.
func LoadLayerFP8Linears(src RawTensorSource, cfg Config) (map[string]*FP8Linear, error) {
	specs, err := RequiredLinearSpecs(cfg)
	if err != nil {
		return nil, err
	}
	out := make(map[string]*FP8Linear, len(specs))
	for _, spec := range specs {
		lin, err := LoadFP8Linear(src, spec)
		if err != nil {
			return nil, err
		}
		out[spec.Prefix] = lin
	}
	return out, nil
}

func isFP8DType(dtype string) bool {
	switch dtype {
	case "F8_E4M3", "F8_E4M3FN":
		return true
	default:
		return false
	}
}

func checkLinearShape(spec LinearSpec, shape []int) error {
	if len(shape) != 2 {
		return fmt.Errorf("ideogram4 fp8 weight %q shape=%v want [out,in]", spec.Weight, shape)
	}
	if shape[0] != spec.OutDim || shape[1] != spec.InDim {
		return fmt.Errorf("ideogram4 fp8 weight %q shape=%v want [%d,%d]", spec.Weight, shape, spec.OutDim, spec.InDim)
	}
	return nil
}

// decodeScale converts a scale tensor into per-tensor (len 1) or per-row
// (len OutDim) float32 scales.
func decodeScale(spec LinearSpec, b []byte, dtype string, shape []int) ([]float32, error) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(shape) == 0 {
		n = 1
	}
	if n != 1 && n != spec.OutDim {
		return nil, fmt.Errorf("ideogram4 fp8 scale %q numel=%d want 1 or %d", spec.WeightScale, n, spec.OutDim)
	}
	out := make([]float32, n)
	switch dtype {
	case "F32":
		if len(b) != n*4 {
			return nil, fmt.Errorf("ideogram4 fp8 scale %q bytes=%d want=%d", spec.WeightScale, len(b), n*4)
		}
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		}
	case "F8_E4M3", "F8_E4M3FN":
		if len(b) != n {
			return nil, fmt.Errorf("ideogram4 fp8 scale %q bytes=%d want=%d", spec.WeightScale, len(b), n)
		}
		for i := 0; i < n; i++ {
			out[i] = fp8.DecodeE4M3(b[i])
		}
	default:
		return nil, fmt.Errorf("ideogram4 fp8 scale %q dtype=%s unsupported", spec.WeightScale, dtype)
	}
	return out, nil
}
