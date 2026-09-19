package community1

import (
	"context"
	"fmt"
)

// ExperimentalSegmentation composes lowered SincNet, recurrent and powerset-head
// inference on one complete mono16k PCM window. It is not production-qualified:
// intermediate boundaries still fail the pinned Torch/MKL strict numerical gate,
// including a silent-window boundary. Local masks and matching final outputs
// alone do not qualify global diarization, DER or other checkpoints.
// All parameters are immutable; call scratch/results are owned independently.
// The supplied checkpoint is retained (its weights are already owned); the
// frontend's weights and lowered filters are copied by the constructor.
// This type is separate from the feature-only SegmentationCheckpoint API.
type ExperimentalSegmentation struct {
	checkpoint *SegmentationCheckpoint
	frontend   *SincNet
}

// SegmentationModes explicitly selects CPU kernels. The assembled experimental
// path requires ordered-FMA SincNet; it never falls back to another runtime.
// Zero-valued modes are invalid here; choose each mode deliberately.
type SegmentationModes struct {
	SincNet SincNetMode
	LSTM    LSTMMode
	Head    HeadMode
}

// SegmentationObservers synchronously receive transient read-only intermediate
// arrays. They must neither retain nor mutate them; copy to retain. Callbacks
// already delivered are not retracted if later computation is cancelled.
type SegmentationObservers struct {
	SincNet SincNetObserver
	LSTM    LSTMObserver
	Head    HeadObserver
}

// SegmentationPCMResult owns frame-major [Grid.Frames,Classes] log probabilities.
// Classes are local-speaker powerset classes, not persistent speaker identities.
// Grid centres are zero-based canonical samples; no source timebase mapping,
// window padding, overlap merge, resampling or implicit silence skipping occurs.
type SegmentationPCMResult struct {
	Grid             SincNetGrid
	Classes          int
	LogProbabilities []float32
}

// NewExperimentalSegmentation requires a checked checkpoint and explicit lowered
// [80,251] filters. Caller must verify their common checkpoint identity/lowering
// policy; shape/finiteness checks cannot establish semantic correspondence.
// No model files, processes, drivers or worker pools are opened here.
func NewExperimentalSegmentation(ctx context.Context, checkpoint *SegmentationCheckpoint, filters []float32) (*ExperimentalSegmentation, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if checkpoint == nil || checkpoint.recurrent == nil || checkpoint.head == nil {
		return nil, fmt.Errorf("invalid segmentation checkpoint")
	}
	frontend, err := NewSincNetWithFilters(ctx, checkpoint.cfg.SincNetStride, checkpoint.sincnet, filters)
	if err != nil {
		return nil, err
	}
	return &ExperimentalSegmentation{checkpoint: checkpoint, frontend: frontend}, nil
}

// ForwardPCM consumes an entire immutable mono16k window, bounded by SincNet's
// Grid (at most160000 samples). It uses zero initial recurrent state. Whole-window
// normalization and reverse recurrence make arbitrary chunking non-equivalent.
// Cancellation/error returns nil, never a partial output; calls in progress and
// observers are synchronous. Callers own admission/concurrency and source PCM.
func (m *ExperimentalSegmentation) ForwardPCM(ctx context.Context, pcm []float32, modes SegmentationModes) (*SegmentationPCMResult, error) {
	return m.ForwardPCMObserved(ctx, pcm, modes, SegmentationObservers{})
}
func (m *ExperimentalSegmentation) ForwardPCMObserved(ctx context.Context, pcm []float32, modes SegmentationModes, observe SegmentationObservers) (*SegmentationPCMResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.frontend == nil || m.checkpoint == nil || m.checkpoint.recurrent == nil || m.checkpoint.head == nil {
		return nil, fmt.Errorf("invalid experimental segmentation model")
	}
	if (modes.SincNet != SincNetScalarFMA && modes.SincNet != SincNetSIMDFMA) || (modes.LSTM != LSTMScalar && modes.LSTM != LSTMSIMD) || (modes.Head != HeadScalar && modes.Head != HeadSIMD) {
		return nil, fmt.Errorf("invalid experimental segmentation modes")
	}
	features, grid, err := m.frontend.ForwardObserved(ctx, pcm, modes.SincNet, observe.SincNet)
	if err != nil {
		return nil, err
	}
	sequence, err := m.checkpoint.recurrent.ForwardObserved(ctx, features, grid.Frames, nil, nil, modes.LSTM, observe.LSTM)
	if err != nil {
		return nil, err
	}
	scores, err := m.checkpoint.head.ForwardObserved(ctx, sequence.Output, grid.Frames, modes.Head, observe.Head)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &SegmentationPCMResult{Grid: grid, Classes: m.checkpoint.head.Classes(), LogProbabilities: scores}, nil
}
