package community1

import (
	"context"
	"errors"
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanEmbedding is an explicit hybrid Community-1 owner: the fixed-frame CNN
// trunk is resident Vulkan, while mask-dependent StatsPool and final projection
// use the existing checked CPU implementation. Copies share one serialized
// lifetime. No model or service default selects this path.
type VulkanEmbedding struct{ s *vulkanEmbeddingState }

type VulkanEmbeddingStats struct {
	Trunk           VulkanResNetStats
	ProjectionBytes uint64
}

type vulkanEmbeddingState struct {
	gate             chan struct{}
	stopping, closed bool
	trunk            *VulkanResNetTrunk
	projection       HeadLinear
	embedDim         int
	stats            VulkanEmbeddingStats
}

type vulkanResNetFactory func(context.Context, *WeSpeakerResNet34, int) (*VulkanResNetTrunk, error)

// NewVulkanEmbedding copies the host projection and constructs the resident
// trunk. Source and frames are fully validated before Vulkan allocation. If
// construction and rollback both fail, it returns a stopping owner whose Close
// can retry the retained trunk cleanup.
func NewVulkanEmbedding(ctx context.Context, source *WeSpeakerResNet34, frames int) (*VulkanEmbedding, error) {
	return newVulkanEmbedding(ctx, source, frames, NewVulkanResNetTrunk)
}

func newVulkanEmbedding(ctx context.Context, source *WeSpeakerResNet34, frames int, makeTrunk vulkanResNetFactory) (result *VulkanEmbedding, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: nil source")
	}
	if makeTrunk == nil {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: nil trunk constructor")
	}
	shape, err := source.FrameShape(frames)
	if err != nil {
		return nil, err
	}
	features := shape.Channels * shape.Frequency
	if len(source.projection.Weight) != source.cfg.EmbedDim*2*features || len(source.projection.Bias) != source.cfg.EmbedDim {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: invalid projection")
	}
	projection := HeadLinear{Weight: append([]float32(nil), source.projection.Weight...), Bias: append([]float32(nil), source.projection.Bias...)}
	if err := finiteBlock(ctx, projection.Weight); err != nil {
		return nil, err
	}
	if err := finiteBlock(ctx, projection.Bias); err != nil {
		return nil, err
	}
	s := &vulkanEmbeddingState{gate: make(chan struct{}, 1), projection: projection, embedDim: source.cfg.EmbedDim}
	owner := &VulkanEmbedding{s: s}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Community-1 Vulkan embedding: rollback: %w", closeErr))
			}
		}
	}()
	// Adopt a partial owner before checking the constructor error. Resident trunk
	// construction can return a stopping owner when its own rollback must retry.
	s.trunk, err = makeTrunk(ctx, source, frames)
	if err != nil {
		return nil, err
	}
	if s.trunk == nil {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: trunk constructor returned nil")
	}
	s.stats = VulkanEmbeddingStats{Trunk: s.trunk.Stats(), ProjectionBytes: uint64(len(projection.Weight)+len(projection.Bias)) * 4}
	return owner, nil
}

func (e *VulkanEmbedding) acquire(ctx context.Context) (*vulkanEmbeddingState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e == nil || e.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: uninitialized owner")
	}
	s := e.s
	select {
	case s.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-s.gate
			return nil, err
		}
		return s, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (e *VulkanEmbedding) Stats() VulkanEmbeddingStats {
	if e == nil || e.s == nil {
		return VulkanEmbeddingStats{}
	}
	return e.s.stats
}

// Forward runs the shared resident trunk once, then performs checked host pooling
// and projection for zero or more masks. It returns raw unnormalised embeddings
// and the same support metadata as WeSpeakerResNet34.Forward.
func (e *VulkanEmbedding) Forward(ctx context.Context, fbank []float32, frames int, masks []float32, speakers, maskFrames int) (*WeSpeakerEmbeddingResult, error) {
	s, err := e.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed {
		return nil, vk.ErrVulkanClosed
	}
	if frames != s.stats.Trunk.Frames {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: frame geometry mismatch")
	}
	shape := s.stats.Trunk.Output
	if masks == nil {
		if speakers != 0 || maskFrames != 0 || shape.Frames < 2 {
			return nil, fmt.Errorf("unweighted embedding requires at least two CNN frames and no mask geometry")
		}
	} else if speakers < 1 || speakers > 8 || maskFrames < 1 || maskFrames > 4096 || len(masks) != speakers*maskFrames {
		return nil, fmt.Errorf("invalid embedding masks")
	}
	if err := finitePool(ctx, masks, true); err != nil {
		return nil, err
	}
	features, shape, err := s.trunk.ForwardFrames(ctx, fbank, frames)
	if err != nil {
		return nil, err
	}
	pooled, err := StatsPool(ctx, features, masks, StatsPoolConfig{Features: shape.Channels * shape.Frequency, Frames: shape.Frames, Speakers: speakers, MaskFrames: maskFrames})
	if err != nil {
		return nil, err
	}
	rows := max(1, speakers)
	columns := 2 * shape.Channels * shape.Frequency
	if len(s.projection.Weight) != s.embedDim*columns || len(s.projection.Bias) != s.embedDim {
		return nil, fmt.Errorf("Community-1 Vulkan embedding: closed/invalid projection")
	}
	output := make([]float32, rows*s.embedDim)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		destination := output[row*s.embedDim : (row+1)*s.embedDim]
		input := pooled.Statistics[row*columns : (row+1)*columns]
		if !simd.GemvRows(destination, input, s.projection.Weight, s.embedDim, columns) {
			return nil, fmt.Errorf("Community-1 Vulkan embedding: projection rejected")
		}
		for i := range destination {
			destination[i] += s.projection.Bias[i]
		}
	}
	if err := finiteBlock(ctx, output); err != nil {
		return nil, err
	}
	return &WeSpeakerEmbeddingResult{Embeddings: output, WeightSum: pooled.WeightSum, NonzeroFrames: pooled.NonzeroFrames}, ctx.Err()
}

func (e *VulkanEmbedding) Close() error {
	if e == nil || e.s == nil {
		return nil
	}
	s, err := e.acquire(context.Background())
	if err != nil {
		return err
	}
	defer func() { <-s.gate }()
	if s.closed {
		return nil
	}
	s.stopping = true
	if s.trunk != nil {
		if err := s.trunk.Close(); err != nil {
			return err
		}
	}
	s.closed = true
	s.trunk = nil
	s.projection = HeadLinear{}
	s.embedDim = 0
	return nil
}
