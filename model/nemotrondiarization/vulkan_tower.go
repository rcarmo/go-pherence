package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanAudioTower runs the released layer 0 on CPU and the remaining 30
// unmasked layers with device-resident activations. Each layer has a bounded
// fixed-row plan; only the first layer input and final-normalised output cross
// the host/device boundary. Plans still fence between layers. This does not
// include the streaming speaker cache, upsampler or classifier. Call Close.
type VulkanAudioTower struct {
	first                  *Layer0Complete
	layers                 []*VulkanAudioLayer
	final                  *vk.VkF32Plan
	output                 *vk.VkTensorF32
	finalNorm              *vk.VkLayerNormF32
	finalWeight, finalBias *vk.VkTensorF32
	owners                 []interface{ Close() error }
	rows                   int
	mu                     sync.Mutex // serializes plan use and teardown
	closed                 bool
}

func (t *VulkanAudioTower) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	// A later layer can bind an earlier layer's output as input. Release
	// owners in strict reverse order; if one fails, keep it and every
	// upstream owner alive for a retry after VulkanDrain.
	for i := len(t.owners) - 1; i >= 0; i-- {
		if t.owners[i] == nil {
			continue
		}
		if err := t.owners[i].Close(); err != nil {
			return err
		}
		t.owners[i] = nil
	}
	t.closed = true
	t.owners, t.layers, t.final, t.output, t.first = nil, nil, nil, nil, nil
	t.finalNorm, t.finalWeight, t.finalBias = nil, nil, nil
	return nil
}

func NewVulkanAudioTower(ctx context.Context, source *OfflineAudioTower, rows int) (result *VulkanAudioTower, err error) {
	if ctx == nil || source == nil || source.first == nil || len(source.remaining) != 30 ||
		len(source.finalWeight) != projectedWidth || len(source.finalBias) != projectedWidth || rows < 1 || rows > maxPreparedDiarizationRows {
		return nil, fmt.Errorf("invalid Nemotron Vulkan tower model or window")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t := &VulkanAudioTower{first: source.first, rows: rows, layers: make([]*VulkanAudioLayer, 0, 30)}
	defer func() {
		if err != nil {
			err = errors.Join(err, t.Close())
			if !t.closed {
				result = t // retain cleanup ownership for retry
			}
		}
	}()
	for index, model := range source.remaining {
		var previous *vk.VkTensorF32
		if index != 0 {
			previous = t.layers[index-1].output
		}
		var layer *VulkanAudioLayer
		layer, err = newVulkanAudioLayerFromTensor(ctx, model, rows, previous)
		if layer != nil {
			t.layers = append(t.layers, layer)
			t.owners = append(t.owners, layer)
		}
		if err != nil {
			return nil, fmt.Errorf("Nemotron Vulkan tower layer %d: %w", index+1, err)
		}
	}
	// Two F32 vectors plus one [rows,512] output, rounded up to MiB.
	// The fixed 1 MiB arena cannot hold the final output when the cache grows.
	// Leave a bounded allowance for device storage-buffer offset alignment.
	finalBytes := ((rows*projectedWidth*4 + 8192 + ((1 << 20) - 1)) / (1 << 20)) * (1 << 20)
	arena, err := vk.NewVkTensorArena(ctx, finalBytes)
	if err != nil {
		return nil, err
	}
	t.owners = append(t.owners, arena)
	alloc := func(shape ...int) (*vk.VkTensorF32, error) { return arena.AllocF32(ctx, shape...) }
	weight, err := alloc(projectedWidth)
	if err != nil {
		return nil, err
	}
	bias, err := alloc(projectedWidth)
	if err != nil {
		return nil, err
	}
	for _, values := range [][]float32{source.finalWeight, source.finalBias} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("non-finite Nemotron Vulkan tower final weights")
			}
		}
	}
	if err = weight.Upload(ctx, source.finalWeight); err != nil {
		return nil, err
	}
	if err = bias.Upload(ctx, source.finalBias); err != nil {
		return nil, err
	}
	t.output, err = alloc(rows, projectedWidth)
	if err != nil {
		return nil, err
	}
	norm, err := vk.NewVkLayerNormF32(ctx)
	if err != nil {
		return nil, err
	}
	t.owners = append(t.owners, norm)
	t.finalNorm, t.finalWeight, t.finalBias = norm, weight, bias
	stage, err := norm.Stage(ctx, t.output, t.layers[29].output, weight, bias, 1e-5)
	if err != nil {
		return nil, err
	}
	t.final, err = vk.NewVkF32Plan(ctx, []vk.VkF32Stage{stage})
	if err != nil {
		return nil, err
	}
	t.owners = append(t.owners, t.final)
	return t, nil
}

// Forward accepts a fully valid [rows,512] stacking window and returns an
// independent final-normalised tower output. No input is retained or mutated.
func (t *VulkanAudioTower) Forward(ctx context.Context, stacked []float32) ([]float32, error) {
	return t.forward(ctx, stacked, nil)
}

// ForwardRows runs an exact prefix of a maximum-row resident tower. Plans are
// rebound before execution; the attention stage receives precisely rows keys
// and RoPE starts at position zero. The caller's input and output are owned.
func (t *VulkanAudioTower) ForwardRows(ctx context.Context, stacked []float32, rows int) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("nil Nemotron Vulkan tower")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed || t.first == nil || t.final == nil || t.finalNorm == nil || ctx == nil || rows < 1 || rows > t.rows || len(stacked) != rows*projectedWidth || len(t.layers) != 30 {
		return nil, fmt.Errorf("invalid Nemotron Vulkan tower prefix input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, value := range stacked {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron Vulkan tower prefix input")
		}
	}
	first, err := t.first.ForwardOffline(stacked, rows)
	if err != nil {
		return nil, err
	}
	input, err := t.layers[0].input.PrefixRows(ctx, rows)
	if err != nil {
		return nil, err
	}
	if err := input.Upload(ctx, first); err != nil {
		return nil, err
	}
	for index, layer := range t.layers {
		var upstream *vk.VkTensorF32
		if index != 0 {
			upstream, err = t.layers[index-1].output.PrefixRows(ctx, rows)
			if err != nil {
				return nil, err
			}
		}
		if err := layer.rebindRows(ctx, rows, upstream); err != nil {
			return nil, fmt.Errorf("Nemotron Vulkan tower rebind layer %d: %w", index+1, err)
		}
		if err := layer.runResident(ctx); err != nil {
			return nil, fmt.Errorf("Nemotron Vulkan tower layer %d: %w", index+1, err)
		}
	}
	output, err := t.output.PrefixRows(ctx, rows)
	if err != nil {
		return nil, err
	}
	hidden, err := t.layers[29].output.PrefixRows(ctx, rows)
	if err != nil {
		return nil, err
	}
	stage, err := t.finalNorm.Stage(ctx, output, hidden, t.finalWeight, t.finalBias, 1e-5)
	if err != nil {
		return nil, err
	}
	if err := t.final.Rebind(ctx, []vk.VkF32Stage{stage}); err != nil {
		return nil, err
	}
	if err := t.final.Run(ctx); err != nil {
		return nil, err
	}
	result := make([]float32, rows*projectedWidth)
	if err := output.Download(ctx, result); err != nil {
		return nil, err
	}
	for _, value := range result {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron Vulkan tower prefix output")
		}
	}
	return result, nil
}

// observe is for pinned intermediate fixture checks. Production calls do not
// download layer outputs. It runs only after a layer's fence has completed.
// The observer may download selected tensors; it must not retain them.
func (t *VulkanAudioTower) forward(ctx context.Context, stacked []float32, observe func(int, *vk.VkTensorF32) error) ([]float32, error) {
	if t == nil {
		return nil, fmt.Errorf("nil Nemotron Vulkan tower")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed || t.first == nil || t.final == nil || len(t.layers) != 30 || ctx == nil || len(stacked) != t.rows*projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron Vulkan tower input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, value := range stacked {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron Vulkan tower input")
		}
	}
	first, err := t.first.ForwardOffline(stacked, t.rows)
	if err != nil {
		return nil, err
	}
	if err := t.layers[0].input.Upload(ctx, first); err != nil {
		return nil, err
	}
	for index, layer := range t.layers {
		if err := layer.runResident(ctx); err != nil {
			return nil, fmt.Errorf("Nemotron Vulkan tower layer %d: %w", index+1, err)
		}
		if observe != nil {
			if err := observe(index+1, layer.output); err != nil {
				return nil, err
			}
		}
	}
	if err := t.final.Run(ctx); err != nil {
		return nil, err
	}
	result := make([]float32, t.rows*projectedWidth)
	if err := t.output.Download(ctx, result); err != nil {
		return nil, err
	}
	for _, value := range result {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("non-finite Nemotron Vulkan tower output")
		}
	}
	return result, nil
}
