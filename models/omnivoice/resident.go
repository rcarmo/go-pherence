package omnivoice

import (
	"context"
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

var residentLayerSuffixes = [...]string{
	"input_layernorm.weight",
	"post_attention_layernorm.weight",
	"self_attn.q_norm.weight",
	"self_attn.k_norm.weight",
	"self_attn.q_proj.weight",
	"self_attn.k_proj.weight",
	"self_attn.v_proj.weight",
	"self_attn.o_proj.weight",
	"mlp.gate_proj.weight",
	"mlp.up_proj.weight",
	"mlp.down_proj.weight",
}

// residentCache owns immutable float32 decoder layers built once from the
// checkpoint. Blocks reference the buffer tensors directly; keeping both makes
// the slice lifetime explicit for shared sibling backbones.
type residentCache struct {
	buffers     []*loader.LayerBuffer
	blocks      []*Block
	bytes       int64
	packedBytes int64
}

// ResidentRequiredBytes returns the float32 bytes needed to cache every decoder
// layer once. It excludes Go headers and the streamed single-layer arena that
// every backbone already owns.
func (b *Backbone) ResidentRequiredBytes() int64 {
	if b == nil || b.weights == nil {
		return 0
	}
	required, err := residentRequiredBytes(b.weights.Config)
	if err != nil {
		return 0
	}
	return required
}

// ResidentPrepackedRequiredBytes returns the total bytes needed for a resident
// decoder cache retaining both raw float32 weights and any prepacked full
// 16-column projection panels.
func (b *Backbone) ResidentPrepackedRequiredBytes() int64 {
	if b == nil || b.weights == nil {
		return 0
	}
	required, err := residentPrepackedRequiredBytes(b.weights.Config)
	if err != nil {
		return 0
	}
	return required
}

// PrepackedBytes returns the currently attached resident prepacked-panel bytes,
// excluding Go headers and the always-present streamed arena.
func (b *Backbone) PrepackedBytes() int64 {
	if b == nil || b.resident == nil {
		return 0
	}
	return b.resident.packedBytes
}

// ResidentBytes returns the shared resident decoder-cache bytes currently held
// by this backbone, excluding Go headers and the always-present streamed arena.
func (b *Backbone) ResidentBytes() int64 {
	if b == nil || b.resident == nil {
		return 0
	}
	return b.resident.bytes + b.resident.packedBytes
}

// EnableResident converts every decoder layer into an immutable float32 cache.
// The operation is transactional: cancellation or any error leaves the backbone
// in streamed mode with no partially enabled resident state attached.
func (b *Backbone) EnableResident(ctx context.Context, maxBytes int64) error {
	if b == nil || b.weights == nil || ctx == nil {
		return fmt.Errorf("omnivoice: nil context")
	}
	if b.directQ8 != nil {
		return fmt.Errorf("omnivoice: resident layers incompatible with direct Q8")
	}
	if b.resident != nil {
		return nil
	}
	required, err := residentRequiredBytes(b.weights.Config)
	if err != nil {
		return err
	}
	if maxBytes < required {
		return fmt.Errorf("omnivoice: resident cache needs %d bytes, budget %d", required, maxBytes)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	resident, err := newResidentCache(ctx, b.weights)
	if err != nil {
		return err
	}
	b.resident = resident
	return nil
}

// EnableResidentPrepacked builds a resident decoder cache retaining the raw
// float32 layer tensors plus prepacked full 16-column projection panels.
// Upgrading an existing raw resident cache is transactional and does not mutate
// already attached sibling caches on cancellation or error.
func (b *Backbone) EnableResidentPrepacked(ctx context.Context, maxBytes int64) error {
	if b == nil || b.weights == nil || ctx == nil {
		return fmt.Errorf("omnivoice: nil context")
	}
	if b.directQ8 != nil {
		return fmt.Errorf("omnivoice: resident layers incompatible with direct Q8")
	}
	if b.resident != nil && b.resident.packedBytes != 0 {
		return nil
	}
	required, err := residentPrepackedRequiredBytes(b.weights.Config)
	if err != nil {
		return err
	}
	if maxBytes < required {
		return fmt.Errorf("omnivoice: resident prepacked cache needs %d bytes, budget %d", required, maxBytes)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	base := b.resident
	if base == nil {
		base, err = newResidentCache(ctx, b.weights)
		if err != nil {
			return err
		}
	}
	resident, err := newResidentPrepackedCache(ctx, base, b.weights.Config.LLMConfig)
	if err != nil {
		return err
	}
	b.resident = resident
	return nil
}

func newResidentCache(ctx context.Context, weights *loader.Weights) (*residentCache, error) {
	layers := weights.Config.LLMConfig.NumHiddenLayers
	cache := &residentCache{
		buffers: make([]*loader.LayerBuffer, 0, layers),
		blocks:  make([]*Block, 0, layers),
	}
	for i := 0; i < layers; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		buffer, err := weights.NewLayerBuffer()
		if err != nil {
			return nil, err
		}
		if err = buffer.Load(weights, i); err != nil {
			return nil, err
		}
		block, err := NewBlock(weights.Config.LLMConfig, buffer.Tensors)
		if err != nil {
			return nil, err
		}
		cache.buffers = append(cache.buffers, buffer)
		cache.blocks = append(cache.blocks, block)
		cache.bytes += int64(buffer.Bytes())
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	return cache, nil
}

func newResidentPrepackedCache(ctx context.Context, base *residentCache, cfg loader.LLMConfig) (*residentCache, error) {
	if base == nil {
		return nil, fmt.Errorf("omnivoice: nil resident cache")
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	shapes := blockWeightShapes(cfg)
	cache := &residentCache{
		buffers: append([]*loader.LayerBuffer(nil), base.buffers...),
		blocks:  make([]*Block, len(base.blocks)),
		bytes:   base.bytes,
	}
	for i, block := range base.blocks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		prepacked := make(map[string][]float32, len(blockPrepackedSuffixes))
		for _, suffix := range blockPrepackedSuffixes {
			shape := shapes[suffix]
			packed, err := simd.PackSgemmNTWeights(block.weights[suffix], shape[0], shape[1], shape[1])
			if err != nil {
				return nil, err
			}
			if len(packed) == 0 {
				continue
			}
			packedBytes := int64(len(packed)) * 4
			if cache.packedBytes > maxInt64-packedBytes {
				return nil, fmt.Errorf("omnivoice: resident cache size overflow")
			}
			cache.packedBytes += packedBytes
			prepacked[suffix] = packed
		}
		if len(prepacked) == 0 {
			prepacked = nil
		}
		cache.blocks[i] = block.cloneWithPrepacked(prepacked)
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	return cache, nil
}

func residentRequiredBytes(config loader.Config) (int64, error) {
	shapes := loader.ExpectedShapes(config)
	if shapes == nil {
		return 0, fmt.Errorf("omnivoice: invalid resident cache config")
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	var elemsPerLayer int64
	for _, suffix := range residentLayerSuffixes {
		shape, ok := shapes["llm.layers.0."+suffix]
		if !ok {
			return 0, fmt.Errorf("omnivoice: missing resident tensor shape %s", suffix)
		}
		n, err := residentTensorElemCount(shape)
		if err != nil {
			return 0, err
		}
		if elemsPerLayer > maxInt64-n {
			return 0, fmt.Errorf("omnivoice: resident cache size overflow")
		}
		elemsPerLayer += n
	}
	if elemsPerLayer > maxInt64/4 {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	perLayerBytes := elemsPerLayer * 4
	numLayers := int64(config.LLMConfig.NumHiddenLayers)
	if numLayers <= 0 || perLayerBytes > maxInt64/numLayers {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	return perLayerBytes * numLayers, nil
}

func residentPrepackedRequiredBytes(config loader.Config) (int64, error) {
	raw, err := residentRequiredBytes(config)
	if err != nil {
		return 0, err
	}
	packed, err := residentPackedRequiredBytes(config)
	if err != nil {
		return 0, err
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	if raw > maxInt64-packed {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	return raw + packed, nil
}

func residentPackedRequiredBytes(config loader.Config) (int64, error) {
	shapes := loader.ExpectedShapes(config)
	if shapes == nil {
		return 0, fmt.Errorf("omnivoice: invalid resident cache config")
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	var elemsPerLayer int64
	for _, suffix := range blockPrepackedSuffixes {
		shape, ok := shapes["llm.layers.0."+suffix]
		if !ok {
			return 0, fmt.Errorf("omnivoice: missing resident tensor shape %s", suffix)
		}
		n, err := residentPrepackedElemCount(shape)
		if err != nil {
			return 0, err
		}
		if elemsPerLayer > maxInt64-n {
			return 0, fmt.Errorf("omnivoice: resident cache size overflow")
		}
		elemsPerLayer += n
	}
	if elemsPerLayer > maxInt64/4 {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	perLayerBytes := elemsPerLayer * 4
	numLayers := int64(config.LLMConfig.NumHiddenLayers)
	if numLayers <= 0 || perLayerBytes > maxInt64/numLayers {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	return perLayerBytes * numLayers, nil
}

func residentTensorElemCount(shape []int64) (int64, error) {
	const maxInt64 = int64(^uint64(0) >> 1)
	n := int64(1)
	for _, dim := range shape {
		if dim <= 0 || n > maxInt64/dim {
			return 0, fmt.Errorf("omnivoice: resident cache size overflow")
		}
		n *= dim
	}
	return n, nil
}

func residentPrepackedElemCount(shape []int64) (int64, error) {
	if len(shape) != 2 {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	const maxInt64 = int64(^uint64(0) >> 1)
	out, in := shape[0], shape[1]
	if out <= 0 || in <= 0 {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	fullPanels := out / 16
	if fullPanels == 0 {
		return 0, nil
	}
	if fullPanels > maxInt64/16 {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	packedCols := fullPanels * 16
	if packedCols > maxInt64/in {
		return 0, fmt.Errorf("omnivoice: resident cache size overflow")
	}
	return packedCols * in, nil
}
