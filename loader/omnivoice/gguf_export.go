package omnivoice

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

// ExportGGUF writes a backbone-only GGUF v3 checkpoint. Codec/tokenizer assets
// remain in the original model directory. format is f16, f32, or q8_0. q8_0
// quantizes only the seven decoder projection matrices whose innermost source
// dimension is a multiple of 32; all other tensors are written as F32 to retain
// source precision without padding shapes. The destination must not exist. Only
// one converted tensor is materialised at a time. The derived integer offsets
// are stored implicitly in the validated config.
func ExportGGUF(ctx context.Context, w *Weights, path, format string) error {
	if ctx == nil || w == nil || w.file == nil {
		return fmt.Errorf("omnivoice: nil export context/weights")
	}
	if _, err := exportQTypeForTensor(format, "", nil); err != nil {
		return err
	}
	if err := w.CheckCodebookOffsets(); err != nil {
		return err
	}
	if err := w.Config.Validate(); err != nil {
		return err
	}
	cfg, err := json.Marshal(w.Config)
	if err != nil {
		return err
	}
	infos := w.file.TensorInfos()
	specs := make([]gguf.TensorSpec, 0, len(infos)-1)
	for _, name := range w.file.Names() {
		if name == "codebook_layer_offsets" {
			continue
		}
		info := infos[name]
		qt, err := exportQTypeForTensor(format, name, info.Shape)
		if err != nil {
			return err
		}
		shape := make([]uint64, len(info.Shape))
		for i, d := range info.Shape {
			shape[len(shape)-1-i] = uint64(d)
		}
		specs = append(specs, gguf.TensorSpec{Name: name, Shape: shape, QType: qt})
	}
	meta := []gguf.MetadataEntry{
		{Key: "general.architecture", Value: "omnivoice"},
		{Key: "omnivoice.schema_version", Value: uint32(1)},
		{Key: "omnivoice.config_json", Value: string(cfg)},
		{Key: "omnivoice.layout", Value: "safetensors-names-reversed-dimensions"},
		{Key: "omnivoice.storage", Value: format},
	}
	return gguf.WriteV3(ctx, path, meta, specs, func(ctx context.Context, _ int, t gguf.TensorSpec) (io.Reader, error) {
		raw, dtype, shape, err := w.file.GetRaw(t.Name)
		if err != nil {
			return nil, err
		}
		switch t.QType {
		case gguf.QuantQ8_0:
			if dtype == "Q8_0" {
				return bytes.NewReader(raw), nil
			}
			out, err := quantizeQ8_0Tensor(ctx, t.Name, raw, dtype, shape)
			if err != nil {
				return nil, err
			}
			return bytes.NewReader(out), nil
		case gguf.QuantF32:
			if dtype == "F32" {
				return bytes.NewReader(raw), nil
			}
		case gguf.QuantF16:
			if dtype == "F16" {
				return bytes.NewReader(raw), nil
			}
		default:
			return nil, fmt.Errorf("omnivoice: unsupported export qtype %s", t.QType)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := tensorElementCount(shape)
		if err != nil {
			return nil, err
		}
		if dtype == "Q8_0" {
			decoded := make([]float32, n)
			if err := dequantQ8_0Into(decoded, raw); err != nil {
				return nil, fmt.Errorf("omnivoice: tensor %q: %w", t.Name, err)
			}
			if t.QType == gguf.QuantF32 {
				out := make([]byte, n*4)
				for i, v := range decoded {
					if i&16383 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
				}
				return bytes.NewReader(out), nil
			}
			raw = make([]byte, n*2)
			for i, v := range decoded {
				if i&16383 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				binary.LittleEndian.PutUint16(raw[i*2:], half.F32ToF16(v))
			}
			return bytes.NewReader(raw), nil
		}
		inputWidth, err := floatDTypeWidth(dtype)
		if err != nil {
			return nil, err
		}
		width := 2
		if t.QType == gguf.QuantF32 {
			width = 4
		}
		if n > int(^uint(0)>>1)/width || len(raw) != n*inputWidth {
			return nil, fmt.Errorf("invalid export size")
		}
		out := make([]byte, n*width)
		for i := 0; i < n; i++ {
			if i&16383 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			v, err := decodeTensorFloat(raw[i*inputWidth:], dtype)
			if err != nil {
				return nil, fmt.Errorf("omnivoice: tensor %q[%d]: %w", t.Name, i, err)
			}
			if width == 4 {
				binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
			} else {
				binary.LittleEndian.PutUint16(out[i*2:], half.F32ToF16(v))
			}
		}
		return bytes.NewReader(out), nil
	})
}
