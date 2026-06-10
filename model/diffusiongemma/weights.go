package diffusiongemma

import (
	"fmt"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// TensorBinding joins a planned tensor handle with safetensors metadata. Data is
// not loaded until a caller explicitly asks the underlying ShardedFile for it.
type TensorBinding struct {
	TensorHandle
	DType string `json:"dtype,omitempty"`
	Shape []int  `json:"shape,omitempty"`
}

// LayerWeights is metadata for one DiffusionGemma text decoder layer.
type LayerWeights struct {
	Layer    int             `json:"layer"`
	Type     string          `json:"type,omitempty"`
	Bindings []TensorBinding `json:"bindings"`
}

// TextWeights is a non-eager binding of the DiffusionGemma text tensor plan to
// a sharded safetensors file. It owns the open shard handles and must be closed.
type TextWeights struct {
	Plan    TextTensorPlan  `json:"plan"`
	Globals []TensorBinding `json:"globals"`
	Layers  []LayerWeights  `json:"layers"`
	shards  *safetensors.ShardedFile
}

func OpenTextWeights(modelDir string, shape Shape) (*TextWeights, error) {
	plan, ok, err := TextTensorPlanFromModelDir(modelDir, shape)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("DiffusionGemma safetensors index not found in %s", modelDir)
	}
	if !plan.Ready {
		return nil, fmt.Errorf("DiffusionGemma text tensor plan is incomplete: missing %v", plan.Missing)
	}
	shards, err := safetensors.OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	infos := shards.TensorInfos()
	out := &TextWeights{Plan: plan, shards: shards}
	for _, h := range plan.Globals {
		b, err := bindTensorHandle(h, infos)
		if err != nil {
			_ = out.Close()
			return nil, err
		}
		out.Globals = append(out.Globals, b)
	}
	for _, lp := range plan.Layers {
		lw := LayerWeights{Layer: lp.Layer, Type: lp.Type}
		for _, h := range lp.Handles {
			b, err := bindTensorHandle(h, infos)
			if err != nil {
				_ = out.Close()
				return nil, err
			}
			lw.Bindings = append(lw.Bindings, b)
		}
		out.Layers = append(out.Layers, lw)
	}
	return out, nil
}

func bindTensorHandle(h TensorHandle, infos map[string]safetensors.TensorInfo) (TensorBinding, error) {
	info, ok := infos[h.Name]
	if !ok {
		return TensorBinding{}, fmt.Errorf("DiffusionGemma tensor %q missing from opened shards", h.Name)
	}
	return TensorBinding{TensorHandle: h, DType: info.DType, Shape: append([]int(nil), info.Shape...)}, nil
}

func (w *TextWeights) Close() error {
	if w == nil || w.shards == nil {
		return nil
	}
	return w.shards.Close()
}

func (w *TextWeights) RawTensor(name string) ([]byte, string, []int, error) {
	if w == nil || w.shards == nil {
		return nil, "", nil, fmt.Errorf("nil DiffusionGemma text weights")
	}
	return w.shards.GetRaw(name)
}

func (w *TextWeights) RawTensorRow(name string, row int) ([]byte, string, []int, error) {
	raw, dtype, shape, err := w.RawTensor(name)
	if err != nil {
		return nil, "", nil, err
	}
	if len(shape) != 2 {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q shape %v is not rank-2", name, shape)
	}
	if row < 0 || row >= shape[0] {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q row %d outside [0,%d)", name, row, shape[0])
	}
	elemSize, ok := diffusionGemmaDTypeSize(dtype)
	if !ok {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q unsupported dtype %s", name, dtype)
	}
	rowBytes := shape[1] * elemSize
	start := row * rowBytes
	end := start + rowBytes
	if start < 0 || end < start || end > len(raw) {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q row byte range [%d,%d) exceeds %d", name, start, end, len(raw))
	}
	return raw[start:end], dtype, []int{shape[1]}, nil
}

func diffusionGemmaDTypeSize(dtype string) (int, bool) {
	switch dtype {
	case "F32", "I32", "U32":
		return 4, true
	case "BF16", "F16", "I16", "U16":
		return 2, true
	case "I8", "U8", "BOOL":
		return 1, true
	default:
		return 0, false
	}
}
