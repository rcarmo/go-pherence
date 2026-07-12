package diffusiongemma

import (
	"fmt"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/gguf"
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
	Plan           TextTensorPlan  `json:"plan"`
	Globals        []TensorBinding `json:"globals"`
	Layers         []LayerWeights  `json:"layers"`
	shards         *safetensors.ShardedFile
	floatCache     map[string]FloatTensor
	cacheMu        sync.RWMutex
	noEvict        bool                         // GGUF mode: all weights pre-cached, cannot reload from shards
	ggufQuant      map[string]*gguf.QuantMatrix // canonical 2D GGUF matrices keyed by safetensors binding name
	ggufTokenEmbd  *gguf.QuantMatrix            // original quantized tied token_embd.weight for GGUF LM-head parity
	IndexedExperts bool                         // FP8 mode: experts are per-expert tensors resolved by FP8ExpertIndex, not fused BF16 bindings
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
	out := &TextWeights{Plan: plan, shards: shards, floatCache: map[string]FloatTensor{}, IndexedExperts: plan.IndexedExperts}
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
	if w == nil {
		return nil
	}
	w.cacheMu.Lock()
	w.floatCache = nil
	w.ggufQuant = nil
	w.ggufTokenEmbd = nil
	w.cacheMu.Unlock()
	if w.shards == nil {
		return nil
	}
	err := w.shards.Close()
	w.shards = nil
	return err
}

func (w *TextWeights) CachedFloatTensor(name string) (FloatTensor, error) {
	if w == nil {
		return FloatTensor{}, fmt.Errorf("nil DiffusionGemma text weights")
	}
	w.cacheMu.RLock()
	if t, ok := w.floatCache[name]; ok {
		w.cacheMu.RUnlock()
		return t, nil
	}
	w.cacheMu.RUnlock()

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
	w.cacheMu.Lock()
	w.floatCache[name] = t
	w.cacheMu.Unlock()
	return t, nil
}

func (w *TextWeights) ClearFloatCache() {
	if w != nil {
		w.cacheMu.Lock()
		w.floatCache = map[string]FloatTensor{}
		w.cacheMu.Unlock()
	}
}

func (w *TextWeights) EvictFloatTensor(name string) bool {
	if w == nil || w.floatCache == nil || w.noEvict {
		return false
	}
	w.cacheMu.Lock()
	_, ok := w.floatCache[name]
	if ok {
		delete(w.floatCache, name)
	}
	w.cacheMu.Unlock()
	return ok
}

func (w *TextWeights) EvictLayer(layer int) int {
	if w == nil || layer < 0 || layer >= len(w.Layers) {
		return 0
	}
	evicted := 0
	for _, b := range w.Layers[layer].Bindings {
		if w.EvictFloatTensor(b.Name) {
			evicted++
		}
	}
	return evicted
}

func (w *TextWeights) RetainGlobalsAndLayerPrefix(layers int) int {
	if w == nil || w.noEvict {
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
	w.cacheMu.RLock()
	n := len(w.floatCache)
	w.cacheMu.RUnlock()
	return n
}

func (w *TextWeights) FloatCacheBytes() int64 {
	if w == nil {
		return 0
	}
	w.cacheMu.RLock()
	var total int64
	for _, t := range w.floatCache {
		total += int64(len(t.Data)) * 4
	}
	w.cacheMu.RUnlock()
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
	if w == nil {
		return nil, "", nil, fmt.Errorf("nil DiffusionGemma text weights")
	}
	if w.shards != nil {
		return w.shards.GetRaw(name)
	}
	// GGUF mode: no shards; synthesize raw bytes from float cache
	w.cacheMu.RLock()
	t, ok := w.floatCache[name]
	w.cacheMu.RUnlock()
	if !ok {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q not in float cache (GGUF mode, no shards)", name)
	}
	// Return F32 data as raw bytes.
	want, ok := tensorElementCount(t.Shape)
	if !ok || want != len(t.Data) {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q cached shape %v has %d elements", name, t.Shape, len(t.Data))
	}
	if len(t.Data) == 0 {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q is empty", name)
	}
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&t.Data[0])), len(t.Data)*4)
	return raw, "F32", t.Shape, nil
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
	want, ok := tensorElementCount(shape)
	if !ok || want <= 0 {
		return nil, nil, fmt.Errorf("DiffusionGemma tensor %q invalid BF16 shape %v", name, shape)
	}
	needBytes, ok := checked.MulInt(want, 2)
	if !ok || len(raw) < needBytes {
		return nil, nil, fmt.Errorf("DiffusionGemma tensor %q BF16 bytes=%d want at least %d for shape %v", name, len(raw), needBytes, shape)
	}
	out := unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), want)
	return out, shape, nil
}

func (w *TextWeights) GGUFQuantMatrix(name string) *gguf.QuantMatrix {
	if w == nil {
		return nil
	}
	w.cacheMu.RLock()
	qm := w.ggufQuant[name]
	w.cacheMu.RUnlock()
	return qm
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
	rowBytes, okRowBytes := checked.MulInt(shape[1], elemSize)
	start, okStart := checked.MulInt(row, rowBytes)
	end := start + rowBytes
	if !okRowBytes || !okStart || start < 0 || end < start || end > len(raw) {
		return nil, "", nil, fmt.Errorf("DiffusionGemma tensor %q row byte range [%d,%d) exceeds %d", name, start, end, len(raw))
	}
	return raw[start:end], dtype, []int{shape[1]}, nil
}

func tensorElementCount(shape []int) (int, bool) {
	if len(shape) == 0 {
		return 0, false
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return 0, false
		}
		var ok bool
		n, ok = checked.MulInt(n, d)
		if !ok {
			return 0, false
		}
	}
	return n, true
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

// PreloadLayerLightweight preloads only small tensors (norms, scalars, router)
// for a layer, skipping large projection and MoE weights that the GPU path
// handles via FP8.
func (w *TextWeights) PreloadLayerLightweight(layer int) error {
	if w == nil {
		return fmt.Errorf("nil DiffusionGemma text weights")
	}
	if layer < 0 || layer >= len(w.Layers) {
		return fmt.Errorf("DiffusionGemma layer %d outside [0,%d)", layer, len(w.Layers))
	}
	for _, b := range w.Layers[layer].Bindings {
		switch b.Group {
		case TensorAttention, TensorMoE:
			continue // skip large projections handled by GPU FP8
		}
		// Also skip large decoder-layer tensors (MLP projections)
		if b.Group == TensorDecoderLayer {
			size := 1
			for _, d := range b.Shape {
				size *= d
			}
			if size > 1_000_000 {
				continue
			}
		}
		if _, err := w.CachedFloatTensor(b.Name); err != nil {
			return err
		}
	}
	return nil
}

// PreloadLayerRangeLightweight preloads small tensors for a range of layers.
func (w *TextWeights) PreloadLayerRangeLightweight(start, count int) error {
	for i := 0; i < count; i++ {
		if err := w.PreloadLayerLightweight(start + i); err != nil {
			return err
		}
	}
	return nil
}
