package pockettts

import (
	"fmt"
)

// TrainingAttentionBackend accelerates only transformer attention backward.
// All forward, linear/norm/activation backward and flow/LSD work can remain on
// CPU. This explicit seam is hybrid execution, not complete GPU training.
// Input tensors are immutable [S,H,D] Q/K/V/dO and dense [H,S,S] probabilities.
// Outputs must be finite owned disjoint [S,H,D] slices, not borrowed scratch.
// Causal/context masks follow the transformer and inverse RoPE stays in caller.
// No backend initialisation or retry/fallback is implicit in this interface.
type TrainingAttentionBackend interface {
	AttentionBackward(q, k, v, probability, dout []float32, rows, heads, dim, contextRows int) (dq, dk, dv []float32, err error)
}

// ForwardBackwardAttention evaluates CPU forward and CPU non-attention backward,
// routing only attention differentiation to the explicitly supplied backend.
// A failure returns no usable gradients; parameters themselves are not updated.
func (m *TransformerCPU) ForwardBackwardAttention(sequence, dOutput []float32, rows int, backend TrainingAttentionBackend) (output []float32, gradients *TransformerGradients, dSequence []float32, err error) {
	if backend == nil {
		return nil, nil, nil, fmt.Errorf("explicit attention backend required")
	}
	if err = validateTrainableTransformer(m, sequence, dOutput, rows); err != nil {
		return nil, nil, nil, err
	}
	tape, output := m.forwardTraining(rows, sequence)
	gradients = newTransformerGradients(m)
	dSequence, err = m.backwardTrainingWithAttention(rows, tape, dOutput, gradients, backend)
	if err != nil {
		return nil, nil, nil, err
	}
	return output, gradients, dSequence, nil
}

// TrainingAttentionBackendIdentity names only the implemented hybrid scope.
// Full-device backend names remain unsupported. Metadata equality binds a
// declared implementation, not proof of device/driver numerical equivalence.
type TrainingAttentionBackendIdentity interface{ TrainingAttentionBackendName() string }

func trainingAttentionBackendName(backend TrainingAttentionBackend) string {
	if backend == nil {
		return "simd"
	}
	if named, ok := backend.(TrainingAttentionBackendIdentity); ok {
		return named.TrainingAttentionBackendName()
	}
	return "unknown"
}
func validateAttentionInputOwnership(dq, dk, dv []float32, inputs ...[]float32) error {
	for _, output := range [][]float32{dq, dk, dv} {
		for _, input := range inputs {
			if f32SlicesOverlap(output, input) {
				return fmt.Errorf("attention backend returned borrowed input storage")
			}
		}
	}
	return nil
}
func validateAttentionDerivatives(dq, dk, dv []float32, elements int) error {
	values := [][]float32{dq, dk, dv}
	for i, a := range values {
		if len(a) != elements || !finiteF32(a) {
			return fmt.Errorf("attention backend derivative%d shape/nonfinite", i)
		}
		for _, b := range values[i+1:] {
			if f32SlicesOverlap(a, b) {
				return fmt.Errorf("attention backend outputs overlap")
			}
		}
	}
	return nil
}
