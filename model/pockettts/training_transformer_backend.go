package pockettts

import "fmt"

// TrainingTransformerBackend evaluates the complete stateless transformer
// forward and input/parameter VJP. Sequence assembly, conditioner/EOS/flow losses
// and optimisers are separate. Parameters must stay immutable between paired
// forward and backward calls. No device initialisation or retry is implicit.
type TrainingTransformerBackend interface {
	EvaluateTransformer(model *TransformerCPU, sequence, dOutput []float32, rows int) (TransformerTrainingResult, error)
}

// TrainingTransformerAccumulator returns boundary values/input VJP only;
// parameter gradients accumulate on device in explicit backward calls.
type TrainingTransformerAccumulator interface {
	EvaluateTransformerAccumulating(*TransformerCPU, []float32, []float32, int, bool) (TransformerTrainingResult, error)
}

func evaluateTrainingTransformer(b TrainingTransformerBackend, m *TransformerCPU, x, seed []float32, rows int, accumulate bool) (TransformerTrainingResult, error) {
	if resident, ok := b.(TrainingTransformerAccumulator); ok {
		return resident.EvaluateTransformerAccumulating(m, x, seed, rows, accumulate)
	}
	return b.EvaluateTransformer(m, x, seed, rows)
}

type TrainingTransformerBackendIdentity interface{ TrainingTransformerBackendName() string }

func (b *TransformerTapeTrainingBackend) TrainingTransformerBackendName() string {
	if b == nil {
		return "unknown"
	}
	if named, ok := b.Executor.(TrainingTransformerBackendIdentity); ok {
		return named.TrainingTransformerBackendName()
	}
	return "unknown"
}
func (e *PTXFlowTapeExecutor) TrainingTransformerBackendName() string {
	if e == nil {
		return "unknown"
	}
	return "ptx-transformer-hybrid-v1"
}
func (e *VulkanFlowTapeExecutor) TrainingTransformerBackendName() string {
	if e == nil {
		return "unknown"
	}
	return "vulkan-transformer-hybrid-v1"
}

type TransformerTapeExecutor interface {
	ExecuteTransformerTape(TransformerTrainingTape) (TransformerTrainingResult, error)
}
type TransformerTapeTrainingBackend struct {
	Executor TransformerTapeExecutor
	MaxBytes int64
}

func (b *TransformerTapeTrainingBackend) EvaluateTransformer(model *TransformerCPU, sequence, dOutput []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil || b.Executor == nil {
		return TransformerTrainingResult{}, fmt.Errorf("explicit transformer executor required")
	}
	tape, err := model.BuildTransformerTrainingTape(sequence, dOutput, rows, b.MaxBytes)
	if err != nil {
		return TransformerTrainingResult{}, err
	}
	return b.Executor.ExecuteTransformerTape(tape)
}
func (b *TransformerTapeTrainingBackend) Close() error {
	if b == nil {
		return nil
	}
	if owner, ok := b.Executor.(interface{ Close() error }); ok {
		return owner.Close()
	}
	return nil
}

// Reuse the bounded flow tape native owners for transformer tapes. They retain
// uncertain resources/terminal state across both graph types. Their flow-only
// replay name stays separate from their transformer-hybrid identity.
func (e *PTXFlowTapeExecutor) ExecuteTransformerTape(t TransformerTrainingTape) (TransformerTrainingResult, error) {
	values, err := e.executeTape(t.Tape)
	if err != nil {
		return TransformerTrainingResult{}, err
	}
	return t.Decode(values)
}
func (e *VulkanFlowTapeExecutor) ExecuteTransformerTape(t TransformerTrainingTape) (TransformerTrainingResult, error) {
	values, err := e.executeTape(t.Tape)
	if err != nil {
		return TransformerTrainingResult{}, err
	}
	return t.Decode(values)
}
func transformerGradientsFromNamed(m *TransformerCPU, named map[string][]float32) (*TransformerGradients, error) {
	g := newTransformerGradients(m)
	bindings := map[string][]float32{}
	addFlowLMGradients(bindings, &FlowLMTrainingGradients{Transformer: g})
	parameters := map[string][]float32{}
	addTransformerParameters(parameters, m)
	for name, p := range parameters {
		if len(p) == 0 {
			delete(parameters, name)
		}
	}
	if len(named) != len(parameters) {
		return nil, fmt.Errorf("transformer backend gradient count")
	}
	for name, p := range parameters {
		src, ok := named[name]
		dst := bindings[name]
		if !ok || len(src) != len(p) || len(dst) != len(p) || !finiteF32(src) {
			return nil, fmt.Errorf("transformer backend gradient %s shape/nonfinite", name)
		}
		copy(dst, src)
	}
	return g, nil
}
func validateTransformerBackendResult(m *TransformerCPU, r TransformerTrainingResult, sequence, dOutput []float32, rows int) error {
	return validateTransformerBackendResultMode(m, r, sequence, dOutput, rows, true)
}
func validateTransformerBackendResultMode(m *TransformerCPU, r TransformerTrainingResult, sequence, dOutput []float32, rows int, parameterGradients bool) error {
	if len(r.Output) != len(sequence) || len(r.DSequence) != len(sequence) || !finiteF32(r.Output) || !finiteF32(r.DSequence) {
		return fmt.Errorf("transformer backend output shape/nonfinite")
	}
	if f32SlicesOverlap(r.Output, r.DSequence) {
		return fmt.Errorf("transformer backend outputs overlap")
	}
	for _, dst := range [][]float32{r.Output, r.DSequence} {
		for _, src := range [][]float32{sequence, dOutput} {
			if f32SlicesOverlap(dst, src) {
				return fmt.Errorf("transformer backend borrowed input")
			}
		}
	}
	if !parameterGradients {
		if len(r.Parameters) != 0 {
			return fmt.Errorf("resident transformer returned host gradients")
		}
		return nil
	}
	parameters := map[string][]float32{}
	addTransformerParameters(parameters, m)
	for name, p := range parameters {
		if len(p) == 0 {
			delete(parameters, name)
		}
	}
	if len(r.Parameters) != len(parameters) {
		return fmt.Errorf("transformer backend gradient count")
	}
	for name, p := range parameters {
		x, ok := r.Parameters[name]
		if !ok || len(x) != len(p) || !finiteF32(x) {
			return fmt.Errorf("transformer backend gradient %s shape/nonfinite", name)
		}
	}
	return nil
}

// ForwardBackwardTrainingBackend keeps the default CPU implementation separate
// and routes all transformer math through the explicitly selected backend.
func (m *TransformerCPU) ForwardBackwardTrainingBackend(sequence, dOutput []float32, rows int, backend TrainingTransformerBackend) ([]float32, *TransformerGradients, []float32, error) {
	if backend == nil {
		return nil, nil, nil, fmt.Errorf("explicit transformer backend required")
	}
	if err := validateTrainableTransformer(m, sequence, dOutput, rows); err != nil {
		return nil, nil, nil, err
	}
	r, err := backend.EvaluateTransformer(m, sequence, dOutput, rows)
	if err != nil {
		return nil, nil, nil, err
	}
	if err := validateTransformerBackendResult(m, r, sequence, dOutput, rows); err != nil {
		return nil, nil, nil, err
	}
	g, err := transformerGradientsFromNamed(m, r.Parameters)
	if err != nil {
		return nil, nil, nil, err
	}
	return r.Output, g, r.DSequence, nil
}
