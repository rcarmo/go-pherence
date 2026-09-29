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
