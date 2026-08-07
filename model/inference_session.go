package model

import "fmt"

// InferenceBackend identifies one execution implementation without prescribing
// device details to callers or schedulers.
type InferenceBackend string

const (
	InferenceBackendScalar InferenceBackend = "scalar"
	InferenceBackendSIMD   InferenceBackend = "simd"
	InferenceBackendNVIDIA InferenceBackend = "nvidia"
)

// FinishReason describes why a resumable inference request stopped.
type FinishReason string

const (
	FinishReasonNone      FinishReason = ""
	FinishReasonLength    FinishReason = "length"
	FinishReasonStopToken FinishReason = "stop_token"
	FinishReasonClosed    FinishReason = "closed"
)

// SessionOptions contains request-local controls. MaxTokens bounds generated
// tokens, not prompt tokens. StopTokenIDs is copied by constructors.
type SessionOptions struct {
	Backend      InferenceBackend
	MaxTokens    int
	StopTokenIDs []int
}

// PrefillResult reports the state after a prompt or prompt chunk is accepted.
type PrefillResult struct {
	ConsumedTokens int
	Position       int
	ReadyToDecode  bool
}

// DecodeResult is one generated token and its request-local state transition.
type DecodeResult struct {
	Token        int
	Logits       []float32
	Position     int
	Generated    int
	Finished     bool
	FinishReason FinishReason
}

// SessionCheckpoint is an opaque, session-owned restore point. Implementations
// must reject checkpoints produced by another session.
type SessionCheckpoint interface {
	inferenceSessionCheckpoint()
}

// InferenceSession is the backend-neutral resumable generation boundary.
// Implementations are request-owned and are not safe for concurrent method
// calls. Returned slices belong to the caller. Close is idempotent.
type InferenceSession interface {
	Backend() InferenceBackend
	PrefillChunk(tokens []int) (PrefillResult, error)
	DecodeStep() (DecodeResult, error)
	Checkpoint() (SessionCheckpoint, error)
	Restore(SessionCheckpoint) error
	Finished() (bool, FinishReason)
	Close() error
}

func validateSessionOptions(opts SessionOptions) error {
	if opts.MaxTokens < 0 {
		return fmt.Errorf("max tokens=%d must be non-negative", opts.MaxTokens)
	}
	switch opts.Backend {
	case "", InferenceBackendScalar, InferenceBackendSIMD, InferenceBackendNVIDIA:
	default:
		return fmt.Errorf("unsupported inference backend %q", opts.Backend)
	}
	for i, tok := range opts.StopTokenIDs {
		if tok < 0 {
			return fmt.Errorf("stop token[%d]=%d must be non-negative", i, tok)
		}
	}
	return nil
}
