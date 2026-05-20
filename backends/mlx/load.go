package mlx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// LoadWeight loads an MLX affine quantized weight from safetensors.
// prefix is e.g. "model.layers.0.self_attn.q_proj"
func LoadWeight(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, prefix string, outDim, inDim, groupSize, bits int) (*QuantWeight, error) {
	if bits <= 0 || bits > 32 || 32%bits != 0 {
		return nil, fmt.Errorf("invalid MLX bits=%d", bits)
	}
	if groupSize <= 0 {
		return nil, fmt.Errorf("invalid MLX groupSize=%d", groupSize)
	}
	packFactor := 32 / bits
	// Load packed weight: [outDim, inDim/packFactor] as uint32. Prefer the
	// safetensors shape when available so callers cannot accidentally use a
	// matching element count with the wrong logical row/column dimensions.
	raw, dtype, shape, err := f.GetRaw(prefix + ".weight")
	if err != nil {
		return nil, fmt.Errorf("load %s.weight: %w", prefix, err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("MLX weight raw byte length %d is not divisible by 4", len(raw))
	}

	if len(shape) == 2 {
		shapeOut, shapePackedIn := shape[0], shape[1]
		if shapeOut <= 0 || shapePackedIn <= 0 {
			return nil, fmt.Errorf("MLX weight invalid shape %v", shape)
		}
		shapeIn, ok := checkedMulInt(shapePackedIn, packFactor)
		if !ok {
			return nil, fmt.Errorf("MLX weight shape %v overflows inDim with packFactor=%d", shape, packFactor)
		}
		if shapeIn%groupSize != 0 {
			return nil, fmt.Errorf("MLX weight shape %v implies inDim=%d not divisible by groupSize=%d", shape, shapeIn, groupSize)
		}
		outDim = shapeOut
		inDim = shapeIn
	} else if len(shape) != 0 {
		return nil, fmt.Errorf("MLX weight shape rank %d unsupported for %s.weight", len(shape), prefix)
	}
	if inDim <= 0 || outDim <= 0 {
		return nil, fmt.Errorf("invalid MLX dims outDim=%d inDim=%d", outDim, inDim)
	}
	if inDim%packFactor != 0 {
		return nil, fmt.Errorf("MLX inDim=%d is not divisible by packFactor=%d", inDim, packFactor)
	}
	if inDim%groupSize != 0 {
		return nil, fmt.Errorf("MLX inDim=%d is not divisible by groupSize=%d", inDim, groupSize)
	}
	numGroups := inDim / groupSize

	var weight []uint32
	if dtype == "U32" || dtype == "I32" {
		n := len(raw) / 4
		weight = make([]uint32, n)
		for i := 0; i < n; i++ {
			weight[i] = binary.LittleEndian.Uint32(raw[i*4:])
		}
	} else {
		return nil, fmt.Errorf("MLX weight dtype %s not supported (expected U32/I32)", dtype)
	}

	expectedN, ok := checkedMulInt(outDim, inDim/packFactor)
	if !ok {
		return nil, fmt.Errorf("MLX weight expected size overflows out=%d in=%d packFactor=%d", outDim, inDim, packFactor)
	}
	if len(weight) != expectedN {
		return nil, fmt.Errorf("MLX weight shape mismatch: got %d, expected %d (%dx%d)", len(weight), expectedN, outDim, inDim/packFactor)
	}

	expectedScaleN, ok := checkedMulInt(outDim, numGroups)
	if !ok {
		return nil, fmt.Errorf("MLX scale/bias expected size overflows out=%d groups=%d", outDim, numGroups)
	}

	// Load scales: [outDim, numGroups]
	scales, err := loadMLXFloat(f, prefix+".scales", expectedScaleN)
	if err != nil {
		return nil, err
	}

	// Load biases: [outDim, numGroups]
	biases, err := loadMLXFloat(f, prefix+".biases", expectedScaleN)
	if err != nil {
		return nil, err
	}

	return &QuantWeight{
		Weight:    weight,
		Scales:    scales,
		Biases:    biases,
		OutDim:    outDim,
		InDim:     inDim,
		Groups:    numGroups,
		GroupSize: groupSize,
		Bits:      bits,
	}, nil
}

// loadMLXFloat loads a scale/bias tensor, accepting only F32/F16/BF16.
func loadMLXFloat(f interface {
	GetFloat32(name string) ([]float32, []int, error)
	GetRaw(name string) ([]byte, string, []int, error)
}, name string, expectedN int) ([]float32, error) {
	raw, dtype, shape, err := f.GetRaw(name)
	if err != nil {
		// Fallback for minimal test doubles that do not expose raw dtype.
		data, shape, f32Err := f.GetFloat32(name)
		if f32Err != nil {
			return nil, fmt.Errorf("load %s: %w", name, err)
		}
		if err := validateMLXFloatLen(name, len(data), shape, expectedN); err != nil {
			return nil, err
		}
		return data, nil
	}

	var out []float32
	switch dtype {
	case "F32":
		if len(raw)%4 != 0 {
			return nil, fmt.Errorf("%s F32 raw byte length %d is not divisible by 4", name, len(raw))
		}
		n := len(raw) / 4
		out = make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("%s F16 raw byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out = make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		if len(raw)%2 != 0 {
			return nil, fmt.Errorf("%s BF16 raw byte length %d is not divisible by 2", name, len(raw))
		}
		n := len(raw) / 2
		out = make([]float32, n)
		for i := 0; i < n; i++ {
			bits := uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16
			out[i] = math.Float32frombits(bits)
		}
	default:
		return nil, fmt.Errorf("unsupported dtype %s for %s", dtype, name)
	}
	if err := validateMLXFloatLen(name, len(out), shape, expectedN); err != nil {
		return nil, err
	}
	return out, nil
}

func validateMLXFloatLen(name string, got int, shape []int, expectedN int) error {
	if expectedN < 0 {
		return fmt.Errorf("%s invalid expected length %d", name, expectedN)
	}
	if len(shape) > 0 {
		shapeN := 1
		for _, d := range shape {
			if d <= 0 {
				return fmt.Errorf("%s invalid shape %v", name, shape)
			}
			var ok bool
			shapeN, ok = checkedMulInt(shapeN, d)
			if !ok {
				return fmt.Errorf("%s shape %v element count overflows", name, shape)
			}
		}
		if shapeN != got {
			return fmt.Errorf("%s shape %v has %d elements, raw data has %d", name, shape, shapeN, got)
		}
	}
	if got != expectedN {
		return fmt.Errorf("%s length mismatch: got %d, expected %d", name, got, expectedN)
	}
	return nil
}
