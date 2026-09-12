package community1

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// DiarizationPCMReader reads immutable mono16k float32 PCM by absolute sample.
// media.PCMReader implements it. The caller owns/ closes it and supplies the
// verified sample extent; no media decoding or source-time mapping occurs here.
// Each requested in-extent read MUST fill dst (EOF may accompany a full read).
// This is an absolute-sample must-fill contract, not an io.Reader stream.
type DiarizationPCMReader interface {
	ReadSamplesAt(context.Context, []float32, int64) (int, error)
}

// DiarizationWindow records the source extent and right padding of one window.
// Windows follow unfold(window,step), with one padded orphan at complete*step
// only when input<window or (input-window)%step!=0. No duplicate exact-end window.
type DiarizationWindow struct {
	Start            int64
	Samples, Padding int
}

// PlanDiarizationWindows is bounded to128 windows and four hours of canonical
// input. Empty input is rejected explicitly. Window<=160000, step<=window.
// This bounds retained results; it is not a production long-file scheduler.
func PlanDiarizationWindows(samples int64, window, step int) ([]DiarizationWindow, error) {
	if samples < 1 || samples > 14400*16000 || window < 1 || window > 160000 || step < 1 || step > window {
		return nil, fmt.Errorf("invalid diarization PCM/window bounds")
	}
	full := int64(0)
	if samples >= int64(window) {
		full = 1 + (samples-int64(window))/int64(step)
	}
	orphan := samples < int64(window) || (samples-int64(window))%int64(step) != 0
	count := full
	if orphan {
		count++
	}
	if count > 128 {
		return nil, fmt.Errorf("experimental diarization exceeds128 windows")
	}
	windows := make([]DiarizationWindow, 0, int(count))
	for i := int64(0); i < count; i++ {
		start := i * int64(step)
		n := min(int64(window), samples-start)
		if n < 1 {
			return nil, fmt.Errorf("invalid trailing diarization window")
		}
		windows = append(windows, DiarizationWindow{start, int(n), window - int(n)})
	}
	return windows, nil
}

// DiarizationPCMConfig is explicit experimental policy. MinimumEmbeddingSamples
// controls overlap-clean mask selection, NOT generic speech validity. The caller
// must qualify it against the embedding frontend. Training admission remains
// PostprocessCommunity1's clean ratio0.2; unsupported KMeans/ties are explicit.
// NumSpeakers>0 overrides min/max during postprocessing, matching the underlying
// explicit count policy. Minimum clean support can exceed available speech and
// legitimately chooses the all-speech mask; it is not a guaranteed admission.
// No defaults or inferred model identity. Step/window are canonical samples.
type DiarizationPCMConfig struct {
	WindowSamples, StepSamples, MinimumEmbeddingSamples int
	ExcludeOverlap                                      bool
	MinSpeakers, MaxSpeakers, NumSpeakers               int
	AHCThreshold, Fa, Fb, MinDurationOff                float64
	Constrained                                         bool
	TiePolicy                                           ReconstructionTiePolicy
}

// ExperimentalDiarization holds immutable experimental segmentation/embedding
// wrappers and prepared PLDA. Their unresolved strict numerical gates still
// apply. This composition is not a production-qualified Community-1 pipeline.
// Nil PLDA is permitted for silence/single-training-row paths only; multirow
// postprocessing fails explicitly. No hidden reference runtime or fallback.
type ExperimentalDiarization struct {
	segmentation *ExperimentalSegmentation
	embedding    *ExperimentalEmbedding
	plda         *PreparedPLDA
}

func NewExperimentalDiarization(ctx context.Context, segmentation *ExperimentalSegmentation, embedding *ExperimentalEmbedding, plda *PreparedPLDA) (*ExperimentalDiarization, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if segmentation == nil || segmentation.checkpoint == nil || segmentation.frontend == nil || segmentation.checkpoint.head == nil || segmentation.checkpoint.recurrent == nil || embedding == nil || embedding.model == nil {
		return nil, fmt.Errorf("invalid experimental diarization models")
	}
	if plda != nil && plda.cfg.InputDim != embedding.model.cfg.EmbedDim {
		return nil, fmt.Errorf("diarization PLDA/embedding dimensions differ")
	}
	return &ExperimentalDiarization{segmentation, embedding, plda}, nil
}

// DiarizationPCMResult owns all diagnostic arrays and postprocessed turns.
// Segmentations are [windows,frames,localSpeakers]; embeddings are raw
// [windows,localSpeakers,dimension]. SelectedFrames counts pre-resize masks;
// WeightSum/NonzeroFrames count CNN mask support. Neither is global identity.
// Turns are frame-centre based, may extend into padded regions, and are NOT
// source-PTS mapped, clipped to audio duration, renamed or label-reordered.
type DiarizationPCMResult struct {
	Windows                              []DiarizationWindow
	Grid                                 SincNetGrid
	LocalSpeakers, EmbeddingDimension    int
	Segmentations, Embeddings, WeightSum []float32
	SelectedFrames, NonzeroFrames        []int
	UsedOverlapExcluded                  []bool
	Postprocess                          *PostprocessResult
}

// DiarizationPCMObserver receives stage notifications: read, segmentation,
// masks, embedding per window; then postprocess stages with window=-1. An error
// is preserved and prevents later work. No partial result escapes cancellation.
type DiarizationPCMObserver func(stage string, window int) error

func (m *ExperimentalDiarization) RunPCM(ctx context.Context, reader DiarizationPCMReader, samples int64, cfg DiarizationPCMConfig, segModes SegmentationModes, embeddingMode WeSpeakerBlockMode) (*DiarizationPCMResult, error) {
	return m.RunPCMObserved(ctx, reader, samples, cfg, segModes, embeddingMode, nil)
}
func (m *ExperimentalDiarization) RunPCMObserved(ctx context.Context, reader DiarizationPCMReader, samples int64, cfg DiarizationPCMConfig, segModes SegmentationModes, embeddingMode WeSpeakerBlockMode, observe DiarizationPCMObserver) (*DiarizationPCMResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.segmentation == nil || m.segmentation.checkpoint == nil || m.segmentation.frontend == nil || m.segmentation.checkpoint.head == nil || m.segmentation.checkpoint.recurrent == nil || m.embedding == nil || m.embedding.model == nil || reader == nil {
		return nil, fmt.Errorf("invalid experimental diarization input")
	}
	if (segModes.SincNet != SincNetScalarFMA && segModes.SincNet != SincNetSIMDFMA) || (segModes.LSTM != LSTMScalar && segModes.LSTM != LSTMSIMD) || (segModes.Head != HeadScalar && segModes.Head != HeadSIMD) || (embeddingMode != WeSpeakerBlockScalar && embeddingMode != WeSpeakerBlockSIMD) {
		return nil, fmt.Errorf("invalid diarization CPU mode")
	}
	windows, err := PlanDiarizationWindows(samples, cfg.WindowSamples, cfg.StepSamples)
	if err != nil {
		return nil, err
	}
	grid, err := m.segmentation.frontend.Grid(cfg.WindowSamples)
	if err != nil {
		return nil, err
	}
	if grid.Frames < 2 || grid.Frames > MaxPowersetFrames || len(windows)*grid.Frames > (1<<24)/8 {
		return nil, fmt.Errorf("diarization segmentation frame bound")
	}
	if cfg.WindowSamples < audio.WeSpeakerWindowSamples || cfg.MinimumEmbeddingSamples < 1 || cfg.MinimumEmbeddingSamples > cfg.WindowSamples {
		return nil, fmt.Errorf("invalid embedding sample policy")
	}
	if _, err = m.embedding.model.FrameShape(1 + (cfg.WindowSamples-audio.WeSpeakerWindowSamples)/audio.WeSpeakerHopSamples); err != nil {
		return nil, err
	}
	local := m.segmentation.checkpoint.cfg.Head.Speakers
	dim := m.embedding.model.cfg.EmbedDim
	if local < 1 || local > 8 || dim < 1 || dim > 512 {
		return nil, fmt.Errorf("invalid diarization model dimensions")
	}
	if cfg.MinSpeakers < 1 || cfg.MaxSpeakers < cfg.MinSpeakers || cfg.MaxSpeakers > 64 || cfg.NumSpeakers < 0 || cfg.NumSpeakers > 64 || cfg.AHCThreshold < 0 || cfg.Fa <= 0 || cfg.Fb <= 0 || cfg.MinDurationOff < 0 || cfg.MinDurationOff > 30 {
		return nil, fmt.Errorf("invalid diarization postprocess policy")
	}
	for _, v := range []float64{cfg.AHCThreshold, cfg.Fa, cfg.Fb, cfg.MinDurationOff} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("nonfinite diarization policy")
		}
	}
	post := PostprocessConfig{Reconstruction: ReconstructionConfig{Chunks: len(windows), Frames: grid.Frames, Speakers: local, Start: 0, ChunkDuration: float64(cfg.WindowSamples) / 16000, ChunkStep: float64(cfg.StepSamples) / 16000, FrameDuration: float64(grid.ReceptiveField) / 16000, FrameStep: float64(grid.Step) / 16000, MaxSpeakers: cfg.MaxSpeakers, TiePolicy: cfg.TiePolicy}, EmbeddingDimension: dim, MinSpeakers: cfg.MinSpeakers, NumSpeakers: cfg.NumSpeakers, AHCThreshold: cfg.AHCThreshold, Fa: cfg.Fa, Fb: cfg.Fb, MinDurationOff: cfg.MinDurationOff, Constrained: cfg.Constrained}
	if _, _, err = reconstructionGrid(post.Reconstruction); err != nil {
		return nil, err
	}
	powerset, err := NewPowerset(local, m.segmentation.checkpoint.cfg.Head.MaxActive)
	if err != nil {
		return nil, err
	}
	rows := len(windows) * local
	result := &DiarizationPCMResult{Windows: windows, Grid: grid, LocalSpeakers: local, EmbeddingDimension: dim, Segmentations: make([]float32, len(windows)*grid.Frames*local), Embeddings: make([]float32, rows*dim), WeightSum: make([]float32, rows), SelectedFrames: make([]int, rows), NonzeroFrames: make([]int, rows), UsedOverlapExcluded: make([]bool, rows)}
	notify := func(stage string, index int) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if observe != nil {
			if err := observe(stage, index); err != nil {
				return err
			}
		}
		return ctx.Err()
	}
	pcm := make([]float32, cfg.WindowSamples)
	for index, window := range windows {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		clear(pcm)
		n, readErr := reader.ReadSamplesAt(ctx, pcm[:window.Samples], window.Start)
		// Reading an exact advertised extent must fill it. EOF alongside a full
		// final read is allowed; short reads/truncation never become extra padding.
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			return nil, readErr
		}
		if n != window.Samples {
			return nil, fmt.Errorf("diarization PCM extent mismatch: %w", io.ErrUnexpectedEOF)
		}
		if err = notify("read", index); err != nil {
			return nil, err
		}
		scores, e := m.segmentation.ForwardPCM(ctx, pcm, segModes)
		if e != nil {
			return nil, e
		}
		if scores.Grid != grid || scores.Classes != powerset.Classes() {
			return nil, fmt.Errorf("diarization segmentation geometry changed")
		}
		binary, e := powerset.Decode(ctx, scores.LogProbabilities, grid.Frames, PowersetHard)
		if e != nil {
			return nil, e
		}
		copy(result.Segmentations[index*grid.Frames*local:], binary)
		if err = notify("segmentation", index); err != nil {
			return nil, err
		}
		masks, e := SelectEmbeddingMasks(ctx, binary, EmbeddingMaskConfig{grid.Frames, local, cfg.WindowSamples, cfg.MinimumEmbeddingSamples, cfg.ExcludeOverlap})
		if e != nil {
			return nil, e
		}
		copy(result.SelectedFrames[index*local:], masks.SelectedFrames)
		copy(result.UsedOverlapExcluded[index*local:], masks.UsedOverlapExcluded)
		if err = notify("masks", index); err != nil {
			return nil, err
		}
		trunk, e := m.embedding.ForwardPCMFrames(ctx, pcm, embeddingMode)
		if e != nil {
			return nil, e
		}
		embedded, e := m.embedding.EmbedFrames(ctx, trunk, masks.Masks, local, grid.Frames, embeddingMode)
		if e != nil {
			return nil, e
		}
		copy(result.Embeddings[index*local*dim:], embedded.Embeddings)
		copy(result.WeightSum[index*local:], embedded.WeightSum)
		copy(result.NonzeroFrames[index*local:], embedded.NonzeroFrames)
		if err = notify("embedding", index); err != nil {
			return nil, err
		}
	}
	result.Postprocess, err = PostprocessCommunity1Observed(ctx, result.Segmentations, result.Embeddings, m.plda, post, func(stage string) error { return notify(stage, -1) })
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
