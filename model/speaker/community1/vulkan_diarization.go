package community1

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// VulkanDiarization is the explicit fixed-window Community-1 hybrid owner used
// by service integrations. CPU SincNet, mask selection, pooling, projection and
// postprocessing surround resident Vulkan LSTM/CNN work. The complete owner is
// serialized because its native children share one process-global Vulkan lane.
// It performs no device initialisation and is never selected implicitly.
type VulkanDiarization struct{ s *vulkanDiarizationState }

type VulkanDiarizationStats struct {
	WindowSamples, SegmentationFrames, FbankFrames int
	LocalSpeakers, EmbeddingDimension              int
	Segmentation                                   VulkanSegmentationStats
	Embedding                                      VulkanEmbeddingStats
}

type vulkanDiarizationSegmentation interface {
	ForwardPCM(context.Context, []float32, SincNetMode, HeadMode) (*SegmentationPCMResult, error)
	Close() error
	Stats() VulkanSegmentationStats
	Grid() SincNetGrid
}
type vulkanDiarizationEmbedding interface {
	Forward(context.Context, []float32, int, []float32, int, int) (*WeSpeakerEmbeddingResult, error)
	Close() error
	Stats() VulkanEmbeddingStats
}
type vulkanDiarizationFactories struct {
	segmentation func(context.Context, *SegmentationCheckpoint, []float32, int) (vulkanDiarizationSegmentation, error)
	embedding    func(context.Context, *WeSpeakerResNet34, int) (vulkanDiarizationEmbedding, error)
}
type vulkanDiarizationState struct {
	gate             chan struct{}
	stopping, closed bool
	segmentation     vulkanDiarizationSegmentation
	embedding        vulkanDiarizationEmbedding
	plda             *PreparedPLDA
	stats            VulkanDiarizationStats
}

// NewVulkanDiarization validates fixed geometry before constructing native
// owners. Native construction order is segmentation then embedding; Close uses
// the reverse order and is retryable. The returned model owns copied neural
// parameters and the supplied immutable PLDA reference exclusively.
func NewVulkanDiarization(ctx context.Context, segmentation *SegmentationCheckpoint, filters []float32, embedding *WeSpeakerResNet34, plda *PreparedPLDA, windowSamples int) (*VulkanDiarization, error) {
	return newVulkanDiarization(ctx, segmentation, filters, embedding, plda, windowSamples, vulkanDiarizationFactories{
		segmentation: func(ctx context.Context, source *SegmentationCheckpoint, filters []float32, samples int) (vulkanDiarizationSegmentation, error) {
			return NewVulkanSegmentationPCM(ctx, source, filters, samples)
		},
		embedding: func(ctx context.Context, source *WeSpeakerResNet34, frames int) (vulkanDiarizationEmbedding, error) {
			return NewVulkanEmbedding(ctx, source, frames)
		},
	})
}

func newVulkanDiarization(ctx context.Context, segmentation *SegmentationCheckpoint, filters []float32, embedding *WeSpeakerResNet34, plda *PreparedPLDA, windowSamples int, factories vulkanDiarizationFactories) (result *VulkanDiarization, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if segmentation == nil || segmentation.recurrent == nil || segmentation.head == nil || embedding == nil || factories.segmentation == nil || factories.embedding == nil || windowSamples < audio.WeSpeakerWindowSamples || windowSamples > audio.WeSpeakerMaxSamples {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: invalid models/window")
	}
	if plda != nil && plda.cfg.InputDim != embedding.cfg.EmbedDim {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: PLDA/embedding dimensions differ")
	}
	grid, err := segmentation.Grid(windowSamples)
	if err != nil {
		return nil, err
	}
	fbankFrames := 1 + (windowSamples-audio.WeSpeakerWindowSamples)/audio.WeSpeakerHopSamples
	if _, err = embedding.FrameShape(fbankFrames); err != nil {
		return nil, err
	}
	state := &vulkanDiarizationState{gate: make(chan struct{}, 1), plda: plda}
	owner := &VulkanDiarization{s: state}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Community-1 Vulkan diarization: rollback: %w", closeErr))
			}
		}
	}()
	if state.segmentation, err = factories.segmentation(ctx, segmentation, filters, windowSamples); err != nil {
		return nil, err
	}
	if state.segmentation == nil {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: segmentation constructor returned nil")
	}
	if state.embedding, err = factories.embedding(ctx, embedding, fbankFrames); err != nil {
		return nil, err
	}
	if state.embedding == nil {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: embedding constructor returned nil")
	}
	state.stats = VulkanDiarizationStats{
		WindowSamples: windowSamples, SegmentationFrames: grid.Frames, FbankFrames: fbankFrames,
		LocalSpeakers: segmentation.cfg.Head.Speakers, EmbeddingDimension: embedding.cfg.EmbedDim,
		Segmentation: state.segmentation.Stats(), Embedding: state.embedding.Stats(),
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return owner, nil
}

func (m *VulkanDiarization) acquire(ctx context.Context) (*vulkanDiarizationState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: uninitialized owner")
	}
	s := m.s
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

func (m *VulkanDiarization) Stats() VulkanDiarizationStats {
	if m == nil || m.s == nil {
		return VulkanDiarizationStats{}
	}
	return m.s.stats
}

// RunPCM executes the same window/mask/postprocessing policy as
// ExperimentalDiarization, with explicit CPU SincNet/head modes and no CPU or
// automatic fallback for the resident recurrent/CNN operators.
func (m *VulkanDiarization) RunPCM(ctx context.Context, reader DiarizationPCMReader, samples int64, cfg DiarizationPCMConfig, sincMode SincNetMode, headMode HeadMode) (*DiarizationPCMResult, error) {
	s, err := m.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed || s.segmentation == nil || s.embedding == nil || reader == nil {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: closed/invalid input")
	}
	if cfg.WindowSamples != s.stats.WindowSamples || (sincMode != SincNetScalarFMA && sincMode != SincNetSIMDFMA) || (headMode != HeadScalar && headMode != HeadSIMD) {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: invalid policy/mode")
	}
	windows, err := PlanDiarizationWindows(samples, cfg.WindowSamples, cfg.StepSamples)
	if err != nil {
		return nil, err
	}
	grid := s.segmentation.Grid()
	if grid.Frames < 2 || grid.Frames > MaxPowersetFrames || len(windows)*grid.Frames > (1<<24)/8 || grid.Frames != s.stats.SegmentationFrames {
		return nil, fmt.Errorf("diarization segmentation frame bound")
	}
	if cfg.MinimumEmbeddingSamples < 1 || cfg.MinimumEmbeddingSamples > cfg.WindowSamples {
		return nil, fmt.Errorf("invalid embedding sample policy")
	}
	local, dim := s.stats.LocalSpeakers, s.stats.EmbeddingDimension
	if local < 1 || local > 8 || dim < 1 || dim > 512 {
		return nil, fmt.Errorf("invalid diarization model dimensions")
	}
	if cfg.MinSpeakers < 1 || cfg.MaxSpeakers < cfg.MinSpeakers || cfg.MaxSpeakers > 64 || cfg.NumSpeakers < 0 || cfg.NumSpeakers > 64 || cfg.AHCThreshold < 0 || cfg.Fa <= 0 || cfg.Fb <= 0 || cfg.MinDurationOff < 0 || cfg.MinDurationOff > 30 {
		return nil, fmt.Errorf("invalid diarization postprocess policy")
	}
	for _, value := range []float64{cfg.AHCThreshold, cfg.Fa, cfg.Fb, cfg.MinDurationOff} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("nonfinite diarization policy")
		}
	}
	post := PostprocessConfig{Reconstruction: ReconstructionConfig{Chunks: len(windows), Frames: grid.Frames, Speakers: local, Start: 0, ChunkDuration: float64(cfg.WindowSamples) / 16000, ChunkStep: float64(cfg.StepSamples) / 16000, FrameDuration: float64(grid.ReceptiveField) / 16000, FrameStep: float64(grid.Step) / 16000, MaxSpeakers: cfg.MaxSpeakers, TiePolicy: cfg.TiePolicy}, EmbeddingDimension: dim, MinSpeakers: cfg.MinSpeakers, NumSpeakers: cfg.NumSpeakers, AHCThreshold: cfg.AHCThreshold, Fa: cfg.Fa, Fb: cfg.Fb, MinDurationOff: cfg.MinDurationOff, Constrained: cfg.Constrained}
	if _, _, err = reconstructionGrid(post.Reconstruction); err != nil {
		return nil, err
	}
	powerset, err := NewPowerset(local, s.stats.Segmentation.MaxActive)
	if err != nil || powerset.Classes() != s.stats.Segmentation.Classes {
		return nil, fmt.Errorf("Community-1 Vulkan diarization: powerset geometry changed")
	}
	rows := len(windows) * local
	result := &DiarizationPCMResult{Windows: windows, Grid: grid, LocalSpeakers: local, EmbeddingDimension: dim, Segmentations: make([]float32, len(windows)*grid.Frames*local), Embeddings: make([]float32, rows*dim), WeightSum: make([]float32, rows), SelectedFrames: make([]int, rows), NonzeroFrames: make([]int, rows), UsedOverlapExcluded: make([]bool, rows)}
	pcm := make([]float32, cfg.WindowSamples)
	for index, window := range windows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		clear(pcm)
		n, readErr := reader.ReadSamplesAt(ctx, pcm[:window.Samples], window.Start)
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return nil, readErr
		}
		if n != window.Samples {
			return nil, fmt.Errorf("diarization PCM extent mismatch: %w", io.ErrUnexpectedEOF)
		}
		scores, err := s.segmentation.ForwardPCM(ctx, pcm, sincMode, headMode)
		if err != nil {
			return nil, err
		}
		if scores.Grid != grid || scores.Classes != powerset.Classes() {
			return nil, fmt.Errorf("diarization segmentation geometry changed")
		}
		binary, err := powerset.Decode(ctx, scores.LogProbabilities, grid.Frames, PowersetHard)
		if err != nil {
			return nil, err
		}
		copy(result.Segmentations[index*grid.Frames*local:], binary)
		masks, err := SelectEmbeddingMasks(ctx, binary, EmbeddingMaskConfig{grid.Frames, local, cfg.WindowSamples, cfg.MinimumEmbeddingSamples, cfg.ExcludeOverlap})
		if err != nil {
			return nil, err
		}
		copy(result.SelectedFrames[index*local:], masks.SelectedFrames)
		copy(result.UsedOverlapExcluded[index*local:], masks.UsedOverlapExcluded)
		fbank, frames, err := audio.WeSpeakerFbank(ctx, pcm)
		if err != nil {
			return nil, err
		}
		if frames != s.stats.FbankFrames {
			return nil, fmt.Errorf("Community-1 Vulkan diarization: Fbank geometry changed")
		}
		embedded, err := s.embedding.Forward(ctx, fbank, frames, masks.Masks, local, grid.Frames)
		if err != nil {
			return nil, err
		}
		copy(result.Embeddings[index*local*dim:], embedded.Embeddings)
		copy(result.WeightSum[index*local:], embedded.WeightSum)
		copy(result.NonzeroFrames[index*local:], embedded.NonzeroFrames)
	}
	result.Postprocess, err = PostprocessCommunity1(ctx, result.Segmentations, result.Embeddings, s.plda, post)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// Close stops admission and closes embedding then segmentation. A failed child
// remains reachable for retry. PLDA is released only after native closure proves
// complete; this method never drains or recreates the global Vulkan device.
func (m *VulkanDiarization) Close() error {
	if m == nil || m.s == nil {
		return nil
	}
	s, err := m.acquire(context.Background())
	if err != nil {
		return err
	}
	defer func() { <-s.gate }()
	if s.closed {
		return nil
	}
	s.stopping = true
	var failures []error
	if s.embedding != nil {
		if err := s.embedding.Close(); err != nil {
			failures = append(failures, err)
		} else {
			s.embedding = nil
		}
	}
	if s.segmentation != nil {
		if err := s.segmentation.Close(); err != nil {
			failures = append(failures, err)
		} else {
			s.segmentation = nil
		}
	}
	if len(failures) != 0 {
		return errors.Join(failures...)
	}
	s.plda = nil
	s.closed = true
	return nil
}
