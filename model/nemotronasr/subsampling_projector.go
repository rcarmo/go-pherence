package nemotronasr

import "context"

// SubsamplingProjector is an optional per-request projection boundary for the
// unmasked streaming generator. It computes X[rows,4352]*W[1024,4352]^T+B
// and returns owned F32 [rows,1024]. GPU implementations include transfer
// time in each call; the rest of the PCM request stays on CPU.
// A projector must not retain or mutate caller input or weights.
type SubsamplingProjector interface {
	Project(ctx context.Context, input, weight, bias []float32, rows int) ([]float32, error)
}

// ScopedSubsamplingProjector is an optional zero-copy boundary. produce writes
// directly into backend-owned input storage; consume reads projected storage
// before reuse. Neither callback may retain the slice, start asynchronous work,
// or reenter the backend. consume must not mutate its read-only input. Callback
// failure does not publish output or advance the subsampling caches.
// Project still returns owned output for existing callers.
type ScopedSubsamplingProjector interface {
	SubsamplingProjector
	ScopedProjectionEnabled() bool
	ProjectScoped(ctx context.Context, weight, bias []float32, rows int, produce, consume func([]float32) error) error
}
