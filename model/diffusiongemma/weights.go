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
