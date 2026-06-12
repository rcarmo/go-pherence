package diffusiongemma

import (
	"fmt"
	"path/filepath"
	"unsafe"

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
	Plan                   TextTensorPlan  `json:"plan"`
	Globals                []TensorBinding `json:"globals"`
	Layers                 []LayerWeights  `json:"layers"`
	shards                 *safetensors.ShardedFile
	floatCache             map[string]FloatTensor
	q80ResidentLayerPrefix int
}

type FloatTensor struct {
	Data  []float32 `json:"-"`
	Shape []int     `json:"shape"`
	DType string    `json:"dtype"`
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
	out := &TextWeights{Plan: plan, shards: shards, floatCache: map[string]FloatTensor{}}
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

func (w *TextWeights) CachedFloatTensor(name string) (FloatTensor, error) {
	if w == nil {
		return FloatTensor{}, fmt.Errorf("nil DiffusionGemma text weights")
	}
	if t, ok := w.floatCache[name]; ok {
		return t, nil
	}
	raw, dtype, shape, err := w.RawTensor(name)
	if err != nil {
		return FloatTensor{}, err
	}
	n := 1
	for _, dim := range shape {
		if dim <= 0 {
			return FloatTensor{}, fmt.Errorf("DiffusionGemma tensor %q invalid shape %v", name, shape)
		}
		n *= dim
	}
	out := make([]float32, n)
	if err := decodeFloatRowTo(out, raw, dtype); err != nil {
		return FloatTensor{}, err
	}
	t := FloatTensor{Data: out, Shape: append([]int(nil), shape...), DType: dtype}
	w.floatCache[name] = t
	return t, nil
}

func (w *TextWeights) ClearFloatCache() {
	if w != nil {
		w.floatCache = map[string]FloatTensor{}
		w.q80ResidentLayerPrefix = 0
		k3ClearQ80CacheForWeights(w)
	}
}

func (w *TextWeights) EvictFloatTensor(name string) bool {
	if w == nil || w.floatCache == nil {
		return false
	}
	if _, ok := w.floatCache[name]; !ok {
		return false
	}
	delete(w.floatCache, name)
	k3EvictQ80Tensor(w, name)
	return true
}

func (w *TextWeights) EvictLayer(layer int) int {
	if w == nil || layer < 0 || layer >= len(w.Layers) {
		return 0
	}
	evicted := 0
	keepQ80 := layer < w.q80ResidentLayerPrefix
	for _, b := range w.Layers[layer].Bindings {
		if w.floatCache != nil {
			if _, ok := w.floatCache[b.Name]; ok {
				delete(w.floatCache, b.Name)
				evicted++
			}
		}
		if !keepQ80 && k3EvictQ80Tensor(w, b.Name) {
			evicted++
		}
	}
	if !keepQ80 {
		evicted += k3EvictQ80Layer(w, layer)
	}
	return evicted
}

func (w *TextWeights) RetainGlobalsAndLayerPrefix(layers int) int {
	if w == nil {
		return 0
	}
	keep := map[string]bool{}
	for _, b := range w.Globals {
		keep[b.Name] = true
	}
	if layers > len(w.Layers) {
		layers = len(w.Layers)
	}
	if layers < 0 {
		layers = 0
	}
	for i := 0; i < layers; i++ {
		for _, b := range w.Layers[i].Bindings {
			keep[b.Name] = true
		}
	}
	evicted := 0
	for name := range w.floatCache {
		if !keep[name] {
			delete(w.floatCache, name)
			evicted++
		}
	}
	return evicted
}

func (w *TextWeights) FloatCacheEntries() int {
	if w == nil {
		return 0
	}
	return len(w.floatCache)
}

func (w *TextWeights) FloatCacheBytes() int64 {
	if w == nil {
		return 0
	}
	var total int64
	for _, t := range w.floatCache {
		total += int64(len(t.Data)) * 4
	}
	return total
}

func (w *TextWeights) PreloadGlobals() error {
	if w == nil {
		return fmt.Errorf("nil DiffusionGemma text weights")
	}
	for _, b := range w.Globals {
		if _, err := w.CachedFloatTensor(b.Name); err != nil {
			return err
		}
	}
	return nil
}

func (w *TextWeights) PreloadLayer(layer int) error {
	if w == nil {
		return fmt.Errorf("nil DiffusionGemma text weights")
	}
	if layer < 0 || layer >= len(w.Layers) {
		return fmt.Errorf("DiffusionGemma layer %d outside [0,%d)", layer, len(w.Layers))
	}
	for _, b := range w.Layers[layer].Bindings {
		if _, err := w.CachedFloatTensor(b.Name); err != nil {
			return err
		}
	}
	return nil
}

func (w *TextWeights) PreloadLayerRange(start, count int) error {
	if count < 0 {
		return fmt.Errorf("DiffusionGemma negative layer preload count %d", count)
	}
	for i := 0; i < count; i++ {
		if err := w.PreloadLayer(start + i); err != nil {
			return err
		}
	}
	return nil
}

func (w *TextWeights) EagerLoad() (int64, error) {
	if w == nil || w.shards == nil {
		return 0, nil
	}
	return w.shards.EagerLoad()
}

func (w *TextWeights) RawTensor(name string) ([]byte, string, []int, error) {
	if w == nil || w.shards == nil {
		return nil, "", nil, fmt.Errorf("nil DiffusionGemma text weights")
	}
	return w.shards.GetRaw(name)
}

// RawBF16Tensor returns the raw BF16 weight data as []uint16 without decoding.
// Uses unsafe reinterpretation of the mmap'd bytes to avoid copying.
// Returns nil if the tensor is not BF16.
func (w *TextWeights) RawBF16Tensor(name string) ([]uint16, []int, error) {
	raw, dtype, shape, err := w.RawTensor(name)
	if err != nil {
		return nil, nil, err
	}
	if dtype != "BF16" {
		return nil, nil, nil
	}
	n := len(raw) / 2
	out := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), n)
	return out, shape, nil
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
	case "I8", "U8", "BOOL", "F8_E4M3", "F8_E4M3FN":
		return 1, true
	default:
		return 0, false
	}
}
