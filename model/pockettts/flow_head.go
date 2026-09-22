package pockettts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type LinearF32 struct {
	Weight     []float32
	WeightBF16 []uint16
	Bias       []float32
	In, Out    int
}

func (l LinearF32) Forward(dst, input []float32) error {
	if len(dst) != l.Out || len(input) != l.In || l.In <= 0 || l.Out <= 0 {
		return fmt.Errorf("invalid Pocket TTS linear shape")
	}
	if len(l.WeightBF16) == l.In*l.Out {
		if !simd.GemvRowsBF16(dst, input, l.WeightBF16, l.Out, l.In) {
			return fmt.Errorf("Pocket TTS BF16 GEMV failed")
		}
		if l.Bias != nil && !simd.VecAddTo(dst, dst, l.Bias) {
			return fmt.Errorf("Pocket TTS BF16 bias failed")
		}
		return nil
	}
	return AffineSIMD(dst, input, l.Weight, l.Bias, l.In, l.Out)
}

func (l LinearF32) ForwardRows(dst, input []float32, rows int) error {
	if rows <= 0 || len(dst) < rows*l.Out || len(input) < rows*l.In || len(l.Weight) != l.In*l.Out {
		return fmt.Errorf("invalid Pocket TTS linear rows")
	}
	clear(dst[:rows*l.Out])
	ok := simd.DenseNTTo(dst, input, l.Weight, rows, l.Out, l.In, 1, l.In, l.In, l.Out)
	if !ok {
		return fmt.Errorf("Pocket TTS batched linear failed")
	}
	if l.Bias != nil && !simd.AddBiasRowsTo(dst, l.Bias, rows, l.Out) {
		return fmt.Errorf("Pocket TTS batched bias failed")
	}
	return nil
}

type TimestepMLP struct {
	Frequencies []float32
	FC1, FC2    LinearF32
	RMSWeight   []float32
	RMSEpsilon  float32
}
type AdaLNResidual struct {
	NormWeight, NormBias []float32
	FC1, FC2             LinearF32
	Modulation           LinearF32
	Epsilon              float32
}
type AdaLNFinal struct {
	Linear, Modulation LinearF32
	Epsilon            float32
}
type FlowHeadCPU struct {
	Input, Condition LinearF32
	Time             []TimestepMLP
	Blocks           []AdaLNResidual
	Final            AdaLNFinal
}

type FlowHeadScratch struct {
	Hidden, Condition, TimeEmbedding                     []float32
	TimeInput, CosAngles, PiHalf, TimeHidden             []float32
	Mod, CondAct, Norm, Ones, Scale, BlockHidden, Update []float32
	Centered                                             []float32
}

func (m *FlowHeadCPU) NewScratch() (*FlowHeadScratch, error) {
	if m == nil || len(m.Blocks) == 0 {
		return nil, fmt.Errorf("invalid Pocket TTS flow head")
	}
	d := m.Input.Out
	half := len(m.Time[0].Frequencies)
	s := &FlowHeadScratch{Hidden: make([]float32, d), Condition: make([]float32, d), TimeEmbedding: make([]float32, d), TimeInput: make([]float32, 2*half), CosAngles: make([]float32, half), PiHalf: make([]float32, half), TimeHidden: make([]float32, d), Mod: make([]float32, 3*d), CondAct: make([]float32, d), Norm: make([]float32, d), Ones: make([]float32, d), Scale: make([]float32, d), BlockHidden: make([]float32, d), Update: make([]float32, d), Centered: make([]float32, d)}
	for i := range s.Ones {
		s.Ones[i] = 1
	}
	for i := range s.PiHalf {
		s.PiHalf[i] = math.Pi / 2
	}
	return s, nil
}

func (m *FlowHeadCPU) Forward(dst, condition, times, input []float32) error {
	s, err := m.NewScratch()
	if err != nil {
		return err
	}
	return m.ForwardInto(dst, condition, times, input, s)
}
func (m *FlowHeadCPU) ForwardInto(dst, condition, times, input []float32, s *FlowHeadScratch) error {
	if m == nil || s == nil || len(times) != len(m.Time) || len(dst) != m.Final.Linear.Out || len(input) != m.Input.In || len(condition) != m.Condition.In || len(m.Blocks) == 0 {
		return fmt.Errorf("invalid Pocket TTS flow head input")
	}
	if err := m.Input.Forward(s.Hidden, input); err != nil {
		return err
	}
	if err := m.Condition.Forward(s.Condition, condition); err != nil {
		return err
	}
	for i := range m.Time {
		if err := m.Time[i].ForwardInto(s.TimeEmbedding, times[i], s); err != nil {
			return err
		}
		if !simd.VecScaleAddTo(s.Condition, s.Condition, s.TimeEmbedding, 1/float32(len(m.Time))) {
			return fmt.Errorf("Pocket TTS time conditioning SIMD add failed")
		}
	}
	for i := range m.Blocks {
		if err := m.Blocks[i].ForwardInPlaceScratch(s.Hidden, s.Condition, s); err != nil {
			return fmt.Errorf("Pocket TTS flow block %d: %w", i, err)
		}
	}
	return m.Final.ForwardScratch(dst, s.Hidden, s.Condition, s)
}
func (m TimestepMLP) Forward(dst []float32, time float32) error {
	d := m.FC2.Out
	s := &FlowHeadScratch{TimeInput: make([]float32, 2*len(m.Frequencies)), CosAngles: make([]float32, len(m.Frequencies)), PiHalf: make([]float32, len(m.Frequencies)), TimeHidden: make([]float32, d), Centered: make([]float32, d), Ones: make([]float32, d)}
	for i := range s.PiHalf {
		s.PiHalf[i] = math.Pi / 2
	}
	for i := range s.Ones {
		s.Ones[i] = 1
	}
	return m.ForwardInto(dst, time, s)
}
func (m TimestepMLP) ForwardInto(dst []float32, time float32, s *FlowHeadScratch) error {
	half := len(m.Frequencies)
	if half == 0 || s == nil || len(dst) != m.FC2.Out || m.FC1.In != 2*half || m.FC1.Out != m.FC2.In || len(m.RMSWeight) != m.FC2.Out || m.RMSEpsilon <= 0 || !isFinite(time) || len(s.TimeInput) < 2*half || len(s.TimeHidden) < m.FC1.Out {
		return fmt.Errorf("invalid Pocket TTS time embedding")
	}
	input := s.TimeInput[:2*half]
	angles := input[:half]
	if !simd.VecScaleTo(angles, m.Frequencies, time) || !simd.SinF32To(input[half:], angles) {
		return fmt.Errorf("Pocket TTS time sin SIMD failed")
	}
	if !simd.VecAddTo(s.CosAngles[:half], angles, s.PiHalf[:half]) || !simd.SinF32To(input[:half], s.CosAngles[:half]) {
		return fmt.Errorf("Pocket TTS time cos SIMD failed")
	}
	hidden := s.TimeHidden[:m.FC1.Out]
	if err := m.FC1.Forward(hidden, input); err != nil {
		return err
	}
	simd.SiLU(hidden, hidden)
	if err := m.FC2.Forward(dst, hidden); err != nil {
		return err
	}
	return pocketVarianceNormSIMDScratch(dst, m.RMSWeight, m.RMSEpsilon, s.Centered[:len(dst)], s.Ones[:len(dst)])
}
func (b AdaLNResidual) ForwardInPlace(x, condition []float32) error {
	s := &FlowHeadScratch{Mod: make([]float32, 3*len(x)), CondAct: make([]float32, len(x)), Norm: make([]float32, len(x)), Ones: make([]float32, len(x)), Scale: make([]float32, len(x)), BlockHidden: make([]float32, b.FC1.Out), Update: make([]float32, len(x))}
	for i := range s.Ones {
		s.Ones[i] = 1
	}
	return b.ForwardInPlaceScratch(x, condition, s)
}
func (b AdaLNResidual) ForwardInPlaceScratch(x, condition []float32, s *FlowHeadScratch) error {
	channels := len(x)
	if channels == 0 || s == nil || len(condition) != b.Modulation.In || b.Modulation.Out != 3*channels || len(b.NormWeight) != channels || len(b.NormBias) != channels || b.Epsilon <= 0 {
		return fmt.Errorf("invalid Pocket TTS AdaLN residual")
	}
	mod := s.Mod[:3*channels]
	condAct := s.CondAct[:channels]
	copy(condAct, condition)
	simd.SiLU(condAct, condAct)
	if err := b.Modulation.Forward(mod, condAct); err != nil {
		return err
	}
	norm := s.Norm[:channels]
	if !simd.LayerNormLastAxisTo(norm, x, 1, channels, b.NormWeight, b.NormBias, b.Epsilon) {
		return fmt.Errorf("Pocket TTS residual LayerNorm failed")
	}
	scale := s.Scale[:channels]
	if !simd.VecAddTo(scale, mod[channels:2*channels], s.Ones[:channels]) || !simd.VecMulTo(norm, norm, scale) || !simd.VecAddTo(norm, norm, mod[:channels]) {
		return fmt.Errorf("Pocket TTS residual AdaLN SIMD modulation failed")
	}
	hidden := s.BlockHidden[:b.FC1.Out]
	if err := b.FC1.Forward(hidden, norm); err != nil {
		return err
	}
	simd.SiLU(hidden, hidden)
	update := s.Update[:channels]
	if err := b.FC2.Forward(update, hidden); err != nil {
		return err
	}
	if !simd.VecMulTo(update, update, mod[2*channels:]) || !simd.VecAddTo(x, x, update) {
		return fmt.Errorf("Pocket TTS residual SIMD update failed")
	}
	return nil
}
func (f AdaLNFinal) Forward(dst, x, condition []float32) error {
	s := &FlowHeadScratch{Mod: make([]float32, 2*len(x)), CondAct: make([]float32, len(x)), Norm: make([]float32, len(x)), Ones: make([]float32, len(x)), Scale: make([]float32, len(x))}
	for i := range s.Ones {
		s.Ones[i] = 1
	}
	return f.ForwardScratch(dst, x, condition, s)
}
func (f AdaLNFinal) ForwardScratch(dst, x, condition []float32, s *FlowHeadScratch) error {
	channels := len(x)
	if channels == 0 || s == nil || len(condition) != f.Modulation.In || f.Modulation.Out != 2*channels || f.Epsilon <= 0 {
		return fmt.Errorf("invalid Pocket TTS final AdaLN")
	}
	condAct := s.CondAct[:channels]
	copy(condAct, condition)
	simd.SiLU(condAct, condAct)
	mod := s.Mod[:2*channels]
	if err := f.Modulation.Forward(mod, condAct); err != nil {
		return err
	}
	norm := s.Norm[:channels]
	if !simd.LayerNormLastAxisTo(norm, x, 1, channels, nil, nil, f.Epsilon) {
		return fmt.Errorf("Pocket TTS final LayerNorm failed")
	}
	scale := s.Scale[:channels]
	if !simd.VecAddTo(scale, mod[channels:], s.Ones[:channels]) || !simd.VecMulTo(norm, norm, scale) || !simd.VecAddTo(norm, norm, mod[:channels]) {
		return fmt.Errorf("Pocket TTS final AdaLN SIMD modulation failed")
	}
	return f.Linear.Forward(dst, norm)
}
func pocketVarianceNormSIMD(x, alpha []float32, eps float32) error {
	centered, ones := make([]float32, len(x)), make([]float32, len(x))
	for i := range ones {
		ones[i] = 1
	}
	return pocketVarianceNormSIMDScratch(x, alpha, eps, centered, ones)
}
func pocketVarianceNormSIMDScratch(x, alpha []float32, eps float32, centered, ones []float32) error {
	if len(x) < 2 || len(alpha) != len(x) || eps <= 0 || len(centered) < len(x) || len(ones) < len(x) {
		return fmt.Errorf("invalid Pocket TTS variance norm")
	}
	mean := simd.Sdot(x, ones[:len(x)]) / float32(len(x))
	if !simd.VecScaleAddTo(centered[:len(x)], x, ones[:len(x)], -mean) {
		return fmt.Errorf("Pocket TTS variance centring SIMD failed")
	}
	variance := simd.Sdot(centered[:len(x)], centered[:len(x)]) / float32(len(x)-1)
	scale := float32(1 / math.Sqrt(float64(variance+eps)))
	if !simd.VecMulTo(x, x, alpha) || !simd.VecScaleTo(x, x, scale) {
		return fmt.Errorf("Pocket TTS variance norm SIMD failed")
	}
	return nil
}
func isFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }
