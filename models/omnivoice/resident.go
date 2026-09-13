package omnivoice

import (
	"context"
	"fmt"

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
	buffers []*loader.LayerBuffer
	blocks  []*Block
	bytes   int64
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

// ResidentBytes returns the shared resident decoder-cache bytes currently held
// by this backbone, excluding Go headers and the always-present streamed arena.
func (b *Backbone) ResidentBytes() int64 {
	if b == nil || b.resident == nil {
		return 0
	}
	return b.resident.bytes
}

// EnableResident converts every decoder layer into an immutable float32 cache.
// The operation is transactional: cancellation or any error leaves the backbone
// in streamed mode with no partially enabled resident state attached.
func (b *Backbone) EnableResident(ctx context.Context, maxBytes int64) error {
	if b == nil || b.weights == nil || ctx == nil {
		return fmt.Errorf("omnivoice: nil context")
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
		n := int64(1)
		for _, dim := range shape {
			if dim <= 0 || n > maxInt64/dim {
				return 0, fmt.Errorf("omnivoice: resident cache size overflow")
			}
			n *= dim
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
