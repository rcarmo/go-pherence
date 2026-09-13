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
// remain in the original model directory. format is f16 or f32. The destination
// must not exist. Only one converted tensor is materialised at a time.
// The derived integer offsets are stored implicitly in the validated config.
func ExportGGUF(ctx context.Context, w *Weights, path, format string) error {
	if ctx == nil || w == nil || w.file == nil {
		return fmt.Errorf("omnivoice: nil export context/weights")
	}
	var qt gguf.QuantType
	var width int
	switch format {
	case "f16":
		qt = gguf.QuantF16
		width = 2
	case "f32":
		qt = gguf.QuantF32
		width = 4
	default:
		return fmt.Errorf("omnivoice: GGUF format must be f16 or f32")
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
		raw, dtype, _, err := w.file.GetRaw(t.Name)
		if err != nil {
			return nil, err
		}
		if (dtype == "F16" && width == 2) || (dtype == "F32" && width == 4) {
			return bytes.NewReader(raw), nil
		}
		inputWidth := 2
		if dtype == "F32" {
			inputWidth = 4
		} else if dtype != "F16" && dtype != "BF16" {
			return nil, fmt.Errorf("unsupported export dtype %s", dtype)
		}
		n := len(raw) / inputWidth
		if len(raw)%inputWidth != 0 || n > int(^uint(0)>>1)/width {
			return nil, fmt.Errorf("invalid export size")
		}
		out := make([]byte, n*width)
		for i := 0; i < n; i++ {
			if i%16384 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			var v float32
			switch dtype {
			case "F32":
				v = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			case "F16":
				v = half.F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
			case "BF16":
				v = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
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
