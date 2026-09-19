package community1

import (
	"context"
	"fmt"

	"github.com/rcarmo/go-pherence/loader/audio"
)

// ExperimentalEmbedding composes the fixed WeSpeaker mono16k frontend with a
// checked ResNet34 whose weights are already owned. Reduced-width 80-bin models
// are accepted for tests; constructing this wrapper does not identify a trained
// checkpoint. It is not production-qualified: trained Fbank, trunk
// and soft-mask support comparisons still fail their strict reference gates.
// Matching raw embeddings alone does not establish DER or speaker identity.
// The retained model is immutable; callers own admission and concurrency.
type ExperimentalEmbedding struct{ model *WeSpeakerResNet34 }

// EmbeddingPCMFrames owns immutable trunk features tied to the exact wrapper
// that produced them. Fields are private to prevent mutation or model mixups.
// Dropping this object releases its Go-owned features; no native resource exists.
// Multiple mask sets may reuse it without recomputing Fbank or the CNN. This
// reuse is valid only for the same complete PCM window/centering policy.
type EmbeddingPCMFrames struct {
	owner                *ExperimentalEmbedding
	values               []float32
	shape                CHWShape
	samples, fbankFrames int
}

// Shape returns value metadata only; feature storage is not exposed.
func (f *EmbeddingPCMFrames) Shape() (CHWShape, int, int) {
	if f == nil {
		return CHWShape{}, 0, 0
	}
	return f.shape, f.samples, f.fbankFrames
}

type EmbeddingPCMObservers struct {
	Fbank audio.WeSpeakerFbankObserver
	Trunk WeSpeakerResNetObserver
}

// NewExperimentalEmbedding retains checked weights; it does not open files or
// infer checkpoint identity. Callers must verify the checkpoint's fixed frontend
// configuration and weight hashes. The wrapper only accepts 80-bin models but
// deliberately does not enforce production channel/embedding widths.
func NewExperimentalEmbedding(ctx context.Context, m *WeSpeakerResNet34) (*ExperimentalEmbedding, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.cfg.MelBins != audio.WeSpeakerMelBands {
		return nil, fmt.Errorf("experimental embedding requires checked 80-bin model")
	}
	shape, err := m.FrameShape(9)
	if err != nil {
		return nil, err
	}
	if err = m.validateEmbeddingInput(ctx, shape, nil, 0, 0); err != nil {
		return nil, err
	}
	return &ExperimentalEmbedding{model: m}, nil
}

// ForwardPCMFrames runs Fbank and all16 residual blocks once for one complete
// immutable mono16k window of400..160000 samples. It neither pads/resamples nor
// skips silence; whole-window centering makes arbitrary chunking non-equivalent.
// Observers are synchronous read-only transient views and must copy to retain.
// Cancellation/error returns nil, even if earlier observers already ran.
func (m *ExperimentalEmbedding) ForwardPCMFrames(ctx context.Context, pcm []float32, mode WeSpeakerBlockMode) (*EmbeddingPCMFrames, error) {
	return m.ForwardPCMFramesObserved(ctx, pcm, mode, EmbeddingPCMObservers{})
}
func (m *ExperimentalEmbedding) ForwardPCMFramesObserved(ctx context.Context, pcm []float32, mode WeSpeakerBlockMode, observe EmbeddingPCMObservers) (*EmbeddingPCMFrames, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.model == nil || m.model.cfg.MelBins != 80 {
		return nil, fmt.Errorf("invalid experimental embedding model")
	}
	if !validWeSpeakerBlockMode(mode) {
		return nil, fmt.Errorf("invalid embedding CPU mode")
	}
	// Check bounds/graph before expensive frontend work.
	if len(pcm) < audio.WeSpeakerWindowSamples || len(pcm) > audio.WeSpeakerMaxSamples {
		return nil, fmt.Errorf("invalid embedding PCM extent")
	}
	if _, err := m.model.FrameShape(1 + (len(pcm)-audio.WeSpeakerWindowSamples)/audio.WeSpeakerHopSamples); err != nil {
		return nil, err
	}
	fbank, frames, err := audio.WeSpeakerFbankObserved(ctx, pcm, observe.Fbank)
	if err != nil {
		return nil, err
	}
	features, shape, err := m.model.ForwardFramesObserved(ctx, fbank, frames, mode, observe.Trunk)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &EmbeddingPCMFrames{owner: m, values: features, shape: shape, samples: len(pcm), fbankFrames: frames}, nil
}

// EmbedFrames pools/project all mask rows against one shared trunk. masks are
// immutable speaker-major [speakers,maskFrames], using legacy Torch nearest
// resizing. Nil masks require speakers=maskFrames=0 and at least two CNN frames.
// Returns raw unnormalised vectors and support, including projection bias for
// empty masks. Caller must enforce speech-support/overlap policy before using
// these vectors; no admissible-speaker or persistent-identity decision is made.
func (m *ExperimentalEmbedding) EmbedFrames(ctx context.Context, frames *EmbeddingPCMFrames, masks []float32, speakers, maskFrames int, mode WeSpeakerBlockMode) (*WeSpeakerEmbeddingResult, error) {
	return m.EmbedFramesObserved(ctx, frames, masks, speakers, maskFrames, mode, nil)
}
func (m *ExperimentalEmbedding) EmbedFramesObserved(ctx context.Context, frames *EmbeddingPCMFrames, masks []float32, speakers, maskFrames int, mode WeSpeakerBlockMode, observe WeSpeakerEmbeddingObserver) (*WeSpeakerEmbeddingResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.model == nil || frames == nil || frames.owner != m {
		return nil, fmt.Errorf("embedding frames belong to another model or are uninitialised")
	}
	if !validWeSpeakerBlockMode(mode) {
		return nil, fmt.Errorf("invalid embedding CPU mode")
	}
	return m.model.ForwardEmbeddingObserved(ctx, frames.values, frames.shape, masks, speakers, maskFrames, mode, observe)
}
