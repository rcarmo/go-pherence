package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"slices"
)

func (b *ResidentTrainingBackend) deviceHandoffEnabled() bool {
	return b != nil && (b.Scope == "ptx-resident-devicehandoff-hybrid-v1" || b.Scope == "vulkan-resident-devicehandoff-hybrid-v1" || b.sharedWorkspaceEnabled())
}

// externalizeImmediateChild converts a freshly built, private graph immediately
// before BeginChildTape synchronously consumes its constants. Values stay
// borrowed only within that call chain; the lease retains metadata, not arrays.
// Public/deferred tape conversions keep their independent owned-value contract.
func externalizeImmediateChild(t train.Tape, ids []int, maxBytes int64) (train.Tape, error) {
	if err := t.Validate(maxBytes); err != nil {
		return train.Tape{}, err
	}
	if len(ids) == 0 || len(ids) > 4096 {
		return train.Tape{}, fmt.Errorf("external child input count")
	}
	seen := make(map[int]bool, len(ids))
	for _, id := range ids {
		if id < 0 || id >= len(t.Tensors) || t.Tensors[id].Values == nil || t.Tensors[id].External || seen[id] {
			return train.Tape{}, fmt.Errorf("external child input ID/constant/duplicate")
		}
		seen[id] = true
	}
	t.Tensors = append([]train.TensorSpec(nil), t.Tensors...)
	for _, id := range ids {
		t.Tensors[id].Values = nil
		t.Tensors[id].External = true
	}
	if err := t.Validate(maxBytes); err != nil {
		return train.Tape{}, err
	}
	return t, nil
}

// One exact retained row condition, owned by the step coordinator, never a
// caller-supplied lease/root. Forward host downloads and guards are preserved.
func (b *ResidentTrainingBackend) beginDeviceCondition(condition []float32) (diagonalDeviceRoot, error) {
	if b.deviceCondition != nil || b.deviceDiagonal != nil || b.deviceDistill != nil || b.predictionLease != nil {
		return diagonalDeviceRoot{}, fmt.Errorf("device handoff active row")
	}
	if b.handoffLease == nil || b.handoffRow < 0 || b.handoffRow >= b.handoffFrames || len(condition) != b.handoffWidth || len(b.handoffHidden) != b.handoffFrames*b.handoffWidth {
		return diagonalDeviceRoot{}, fmt.Errorf("device handoff condition owner/row")
	}
	row := b.handoffHidden[b.handoffRow*b.handoffWidth : (b.handoffRow+1)*b.handoffWidth]
	if len(row) == 0 || &condition[0] != &row[0] || !finiteF32(condition) {
		return diagonalDeviceRoot{}, fmt.Errorf("device handoff condition provenance")
	}
	g := &flowTapeGraph{}
	source := g.external(len(b.handoffHidden))
	out := g.command("vector", []int{source, g.zero(1)}, []int{b.handoffWidth}, 0, 1, b.handoffRow*b.handoffWidth, 0, 0)
	g.tape.Outputs = []int{out}
	lease, err := b.handoffLease.BeginChildTape(b.ctx, g.tape, nil, g.tape.Outputs, []train.ResidentLeaseEdge{{Source: b.handoffLease, Output: b.handoffRoot, Input: source}})
	if err != nil {
		return diagonalDeviceRoot{}, err
	}
	b.deviceCondition = lease
	return diagonalDeviceRoot{lease, out}, nil
}
func (b *ResidentTrainingBackend) rememberDeviceValues(kind string, values [][]float32) {
	b.deviceKind = kind
	b.deviceRow = b.handoffRow
	b.deviceValues = values
	b.deviceCopies = make([][]float32, len(values))
	for i, v := range values {
		b.deviceCopies[i] = append([]float32(nil), v...)
	}
}
func (b *ResidentTrainingBackend) validateDeviceValues(kind string, condition []float32, values [][]float32) error {
	if b.deviceCondition == nil || b.deviceKind != kind || b.deviceRow != b.handoffRow || len(values) != len(b.deviceValues) || len(b.deviceCopies) != len(values) || b.handoffRow < 0 || b.handoffRow >= b.handoffFrames || b.handoffWidth <= 0 || len(b.handoffHidden) != b.handoffFrames*b.handoffWidth {
		return fmt.Errorf("device handoff backward row/kind")
	}
	row := b.handoffHidden[b.handoffRow*b.handoffWidth : (b.handoffRow+1)*b.handoffWidth]
	if len(condition) != len(row) || len(row) == 0 || &condition[0] != &row[0] {
		return fmt.Errorf("device handoff backward condition")
	}
	for i, v := range values {
		saved := b.deviceValues[i]
		if len(v) != len(saved) || len(v) == 0 || &v[0] != &saved[0] || !slices.Equal(v, b.deviceCopies[i]) {
			return fmt.Errorf("device handoff backward borrowed value provenance")
		}
	}
	return nil
}
func (b *ResidentTrainingBackend) finishDeviceHandoff() error {
	if b.deviceCondition == nil {
		return fmt.Errorf("device handoff missing row condition")
	}
	if err := b.deviceCondition.Close(); err != nil {
		return b.fail(err)
	}
	b.deviceCondition = nil
	b.deviceDiagonal = nil
	b.deviceDistill = nil
	b.deviceValues = nil
	b.deviceCopies = nil
	b.deviceKind = ""
	b.deviceRow = -1
	return nil
}
func (b *ResidentTrainingBackend) evaluateDeviceDiagonalForward(m *FlowHeadCPU, in DiagonalForwardInputs, logvar float32) (result DiagonalForwardResult, err error) {
	condition, err := b.beginDeviceCondition(in.Condition)
	if err != nil {
		return result, b.fail(err)
	}
	sources, err := beginDiagonalDeviceSources(b.ctx, b.handoffLease, m, b.handoffRow, condition, in, b.MaxTapeBytes)
	if err != nil {
		return result, b.fail(err)
	}
	b.deviceDiagonal = &sources
	ids := []int{sources.Input.Output, sources.Desired.Output, sources.Prediction.Output}
	values, err := sources.Input.Owner.DownloadOutputs(b.ctx, ids)
	if err != nil {
		return result, b.fail(err)
	}
	if len(values) != 3 {
		return DiagonalForwardResult{}, b.fail(fmt.Errorf("device diagonal forward count"))
	}
	result = DiagonalForwardResult{values[0], values[1], values[2]}
	if err = validateDiagonalForwardResult(result, in); err != nil {
		return DiagonalForwardResult{}, b.fail(err)
	}
	if err = admitLSDLossTraining(result.Prediction, result.Desired, logvar, b.MaxTapeBytes); err != nil {
		return DiagonalForwardResult{}, b.fail(err)
	}
	b.rememberDeviceValues("diagonal", values)
	return result, nil
}
func (b *ResidentTrainingBackend) evaluateDeviceDiagonal(m *FlowHeadCPU, condition []float32, time float32, input, prediction, desired []float32, logvar, scale float32) (result DiagonalTrainingResult, err error) {
	if b.deviceDiagonal == nil {
		return result, b.fail(fmt.Errorf("device diagonal missing forward"))
	}
	if err = b.validateDeviceValues("diagonal", condition, [][]float32{input, desired, prediction}); err != nil {
		return result, b.fail(err)
	}
	result, err = runDiagonalDeviceTraining(b.ctx, m, *b.deviceDiagonal, b.handoffRow, len(condition), len(input), time, logvar, scale, b.MaxTapeBytes)
	if err != nil {
		return DiagonalTrainingResult{}, b.fail(err)
	}
	if err = validateDiagonalResult(result, condition, input, prediction, desired); err != nil {
		return DiagonalTrainingResult{}, b.fail(err)
	}
	if err = b.finishDeviceHandoff(); err != nil {
		return DiagonalTrainingResult{}, err
	}
	return result, nil
}
func (b *ResidentTrainingBackend) evaluateDeviceDistillForward(m *FlowHeadCPU, in DistillForwardInputs, logvar float32) (result DistillForwardResult, err error) {
	condition, err := b.beginDeviceCondition(in.Condition)
	if err != nil {
		return result, b.fail(err)
	}
	sources, err := beginDistillDeviceSources(b.ctx, b.handoffLease, m, b.handoffRow, condition, in, b.MaxTapeBytes)
	if err != nil {
		return result, b.fail(err)
	}
	b.deviceDistill = &sources
	ids := make([]int, 6)
	for i, r := range sources.forward {
		ids[i] = r.Output
	}
	values, err := sources.forward[0].Owner.DownloadOutputs(b.ctx, ids)
	if err != nil {
		return result, b.fail(err)
	}
	result, err = decodeDistillForwardBoundary(values, in)
	if err != nil {
		return DistillForwardResult{}, b.fail(err)
	}
	if err = admitLSDLossTraining(result.Prediction, result.Desired, logvar, b.MaxTapeBytes); err != nil {
		return DistillForwardResult{}, b.fail(err)
	}
	b.rememberDeviceValues("distill", values)
	return result, nil
}
func (b *ResidentTrainingBackend) evaluateDeviceDistill(m *FlowHeadCPU, in DistillTrainingInputs, parameterScale float32) (result stepDistillResult, err error) {
	if b.deviceDistill == nil {
		return result, b.fail(fmt.Errorf("device distill missing forward"))
	}
	if err = b.validateDeviceValues("distill", in.Condition, [][]float32{in.PrimaryInput, in.Velocity, in.Tangent, in.EndpointInput, in.Desired, in.Prediction}); err != nil {
		return result, b.fail(err)
	}
	p := b.deviceDistill.provenance
	if p == nil || p.s != in.S || p.t != in.T || !slices.Equal(p.target, in.Target) || !slices.Equal(p.noise, in.Noise) {
		return result, b.fail(fmt.Errorf("device distill times/target/noise provenance"))
	}
	r, e := runDistillDeviceTraining(b.ctx, m, *b.deviceDistill, b.handoffRow, in.LogVariance, parameterScale, b.MaxTapeBytes)
	if e != nil {
		return result, b.fail(e)
	}
	if e = validateDistillResult(r, in); e != nil {
		return result, b.fail(e)
	}
	if err = b.finishDeviceHandoff(); err != nil {
		return stepDistillResult{}, err
	}
	return stepDistillResult{DCondition: r.DCondition, DTarget: r.DTarget, RawSquare: r.RawSquare, Loss: r.Loss, DLogVariance: r.DLogVariance}, nil
}
