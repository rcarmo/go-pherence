package pockettts

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type seanetResidualCPU struct{ Conv1, Conv2 CausalConv1D }
type seanetStageCPU struct {
	Up       TransposedConv1D
	Residual seanetResidualCPU
}
type MimiDecoderCPU struct {
	Quantizer   LinearF32
	Upsample    TransposedConv1D
	Transformer *TransformerCPU
	Initial     CausalConv1D
	Stages      []seanetStageCPU
	Final       CausalConv1D
}

type seanetResidualState struct{ Conv1, Conv2 StreamingConvState }
type seanetStageState struct {
	Up       StreamingTransposedState
	Residual seanetResidualState
}
type MimiDecoderState struct {
	Upsample     StreamingTransposedState
	Transformer  *TransformerState
	Initial      StreamingConvState
	Stages       []seanetStageState
	Final        StreamingConvState
	Quantized    []float32
	TimeA, TimeB []float32
	WorkA, WorkB []float32
	ConvScratch  *Scratch
	ChunkScratch *TransformerChunkScratch
}

func loadCausalConv(src *safetensors.File, prefix string, in, out, kernel, stride, dilation int) (CausalConv1D, error) {
	return loadCausalConvWithBias(src, prefix, in, out, kernel, stride, dilation, true)
}

func loadCausalConvWithBias(src *safetensors.File, prefix string, in, out, kernel, stride, dilation int, bias bool) (CausalConv1D, error) {
	w, shape, err := src.GetFloat32(prefix + ".weight")
	if err != nil {
		return CausalConv1D{}, err
	}
	inPer := in
	if len(shape) == 3 && shape[0] == out && shape[2] == kernel {
		inPer = shape[1]
	}
	if !equalShape(shape, []int{out, inPer, kernel}) || !finiteF32(w) {
		return CausalConv1D{}, fmt.Errorf("invalid Pocket TTS conv %s shape=%v", prefix, shape)
	}
	var b []float32
	if bias {
		b, err = loadVectorF32(src, prefix+".bias", out)
		if err != nil {
			return CausalConv1D{}, err
		}
	}
	return CausalConv1D{Weight: w, Bias: b, In: in, Out: out, Kernel: kernel, Stride: stride, Dilation: dilation}, nil
}
func loadTransposedConv(src *safetensors.File, prefix string, in, out, kernel, stride, groups int, bias bool) (TransposedConv1D, error) {
	w, shape, err := src.GetFloat32(prefix + ".weight")
	if err != nil {
		return TransposedConv1D{}, err
	}
	if !equalShape(shape, []int{in, out / groups, kernel}) || !finiteF32(w) {
		return TransposedConv1D{}, fmt.Errorf("invalid Pocket TTS transposed conv %s shape=%v", prefix, shape)
	}
	var b []float32
	if bias {
		b, err = loadVectorF32(src, prefix+".bias", out)
		if err != nil {
			return TransposedConv1D{}, err
		}
	}
	conv := TransposedConv1D{Weight: w, Bias: b, In: in, Out: out, Kernel: kernel, Stride: stride, Groups: groups}
	conv.packPhases()
	return conv, nil
}

func LoadMimiDecoderCPU(src *safetensors.File, cfg Config) (*MimiDecoderCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Pocket TTS tensor source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	m := &MimiDecoderCPU{}
	var err error
	w, shape, err := src.GetBF16("mimi.quantizer.output_proj.weight")
	if err != nil {
		return nil, err
	}
	if !equalShape(shape, []int{cfg.Mimi.OuterDim, cfg.Mimi.InnerDim, 1}) || len(w) != cfg.Mimi.OuterDim*cfg.Mimi.InnerDim {
		return nil, fmt.Errorf("invalid Pocket TTS quantizer shape=%v", shape)
	}
	m.Quantizer = LinearF32{WeightBF16: w, In: cfg.Mimi.InnerDim, Out: cfg.Mimi.OuterDim}
	if m.Upsample, err = loadTransposedConv(src, "mimi.upsample.convtr.convtr", cfg.Mimi.OuterDim, cfg.Mimi.SEANet.Dimension, 32, 16, cfg.Mimi.SEANet.Dimension, false); err != nil {
		return nil, err
	}
	if m.Transformer, err = LoadTransformerCPU(src, "mimi.decoder_transformer.transformer", cfg.Mimi.Transformer, ""); err != nil {
		return nil, err
	}
	for i := range m.Transformer.Layers {
		p := fmt.Sprintf("mimi.decoder_transformer.transformer.layers.%d", i)
		l := &m.Transformer.Layers[i]
		for _, item := range []struct {
			suffix string
			linear *LinearF32
		}{{"self_attn.in_proj", &l.InProjection}, {"self_attn.out_proj", &l.OutProjection}, {"linear1", &l.FC1}, {"linear2", &l.FC2}} {
			if err := materializeLinearF32(src, p+"."+item.suffix, item.linear); err != nil {
				return nil, err
			}
		}
	}
	if m.Initial, err = loadCausalConv(src, "mimi.decoder.model.0.conv", 512, 512, 7, 1, 1); err != nil {
		return nil, err
	}
	channels := 512
	m.Stages = make([]seanetStageCPU, len(cfg.Mimi.SEANet.Ratios))
	transIndices, resIndices := []int{2, 5, 8}, []int{3, 6, 9}
	for i, ratio := range cfg.Mimi.SEANet.Ratios {
		out := channels / 2
		if m.Stages[i].Up, err = loadTransposedConv(src, fmt.Sprintf("mimi.decoder.model.%d.convtr", transIndices[i]), channels, out, 2*ratio, ratio, 1, true); err != nil {
			return nil, err
		}
		r := &m.Stages[i].Residual
		if r.Conv1, err = loadCausalConv(src, fmt.Sprintf("mimi.decoder.model.%d.block.1.conv", resIndices[i]), out, out/2, 3, 1, 1); err != nil {
			return nil, err
		}
		if r.Conv2, err = loadCausalConv(src, fmt.Sprintf("mimi.decoder.model.%d.block.3.conv", resIndices[i]), out/2, out, 1, 1, 1); err != nil {
			return nil, err
		}
		channels = out
	}
	if m.Final, err = loadCausalConv(src, "mimi.decoder.model.11.conv", channels, 1, 3, 1, 1); err != nil {
		return nil, err
	}
	return m, nil
}
func (m *MimiDecoderCPU) NewState() *MimiDecoderState { return m.NewStateForFrames(250 / 16) }
func (m *MimiDecoderCPU) NewStateForFrames(frames int) *MimiDecoderState {
	capacity := frames * 16
	if capacity < 16 {
		capacity = 16
	}
	transformer, _ := m.Transformer.NewState(capacity)
	convNeed := m.Upsample.scratchFloats(1)
	length := 16
	convNeed = max(convNeed, m.Initial.scratchFloats(length))
	for i := range m.Stages {
		convNeed = max(convNeed, m.Stages[i].Up.scratchFloats(length))
		length *= m.Stages[i].Up.Stride
		convNeed = max(convNeed, m.Stages[i].Residual.Conv1.scratchFloats(length), m.Stages[i].Residual.Conv2.scratchFloats(length))
	}
	convNeed = max(convNeed, m.Final.scratchFloats(length))
	convScratch, _ := NewScratch(convNeed)
	chunkScratch, _ := m.Transformer.NewChunkScratch(16)
	s := &MimiDecoderState{Upsample: m.Upsample.NewState(), Transformer: transformer, Initial: m.Initial.NewState(), Stages: make([]seanetStageState, len(m.Stages)), Final: m.Final.NewState(), Quantized: make([]float32, 512), TimeA: make([]float32, 128*SamplesPerFrame), TimeB: make([]float32, 128*SamplesPerFrame), WorkA: make([]float32, 64*SamplesPerFrame), WorkB: make([]float32, 64*SamplesPerFrame), ConvScratch: convScratch, ChunkScratch: chunkScratch}
	for i := range m.Stages {
		s.Stages[i] = seanetStageState{Up: m.Stages[i].Up.NewState(), Residual: seanetResidualState{Conv1: m.Stages[i].Residual.Conv1.NewState(), Conv2: m.Stages[i].Residual.Conv2.NewState()}}
	}
	return s
}

func (m *MimiDecoderCPU) Decode(latents []float32, frames int, state *MimiDecoderState) ([]float32, error) {
	if frames <= 0 {
		return nil, fmt.Errorf("invalid Pocket TTS Mimi frames")
	}
	if state == nil {
		state = m.NewStateForFrames(frames)
	}
	out := make([]float32, frames*SamplesPerFrame)
	for frame := 0; frame < frames; frame++ {
		if err := m.DecodeFrameInto(out[frame*SamplesPerFrame:(frame+1)*SamplesPerFrame], latents[frame*m.Quantizer.In:(frame+1)*m.Quantizer.In], state); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (m *MimiDecoderCPU) DecodeFrameInto(out, latent []float32, state *MimiDecoderState) error {
	if m == nil || state == nil || len(out) != SamplesPerFrame || len(latent) != m.Quantizer.In || state.ConvScratch == nil {
		return fmt.Errorf("invalid Pocket TTS Mimi frame input")
	}
	if err := m.Quantizer.Forward(state.Quantized, latent); err != nil {
		return err
	}
	hidden := state.TimeA[:m.Quantizer.Out*16]
	if err := m.Upsample.ForwardInto(hidden, state.Quantized, 1, &state.Upsample, state.ConvScratch); err != nil {
		return err
	}
	length := 16
	if state.Transformer == nil || state.Transformer.Position+length > state.Transformer.Layers[0].Capacity {
		return fmt.Errorf("Pocket TTS Mimi transformer state exhausted")
	}
	timeMajor := state.TimeB[:len(hidden)]
	channelToTimePocketInto(timeMajor, hidden, m.Quantizer.Out, length)
	streamed := state.TimeA[:len(hidden)]
	if err := m.Transformer.ForwardChunkInto(streamed, timeMajor, length, state.Transformer, state.ChunkScratch); err != nil {
		return err
	}
	hidden = state.WorkA[:m.Quantizer.Out*length]
	timeToChannelPocketInto(hidden, streamed, length, m.Quantizer.Out)
	initial := state.WorkB[:m.Initial.Out*length]
	if err := m.Initial.ForwardInto(initial, hidden, length, &state.Initial, state.ConvScratch); err != nil {
		return err
	}
	hidden = initial
	for i := range m.Stages {
		if !simd.ELUF32To(hidden, hidden) {
			return fmt.Errorf("Pocket TTS Mimi ELU failed")
		}
		stageOutLen := length * m.Stages[i].Up.Stride
		stageOut := state.WorkA[:m.Stages[i].Up.Out*stageOutLen]
		if err := m.Stages[i].Up.ForwardInto(stageOut, hidden, length, &state.Stages[i].Up, state.ConvScratch); err != nil {
			return err
		}
		hidden, length = stageOut, stageOutLen
		residual := state.WorkB[:len(hidden)]
		copy(residual, hidden)
		if !simd.ELUF32To(hidden, hidden) {
			return fmt.Errorf("Pocket TTS Mimi residual ELU failed")
		}
		conv1 := state.TimeA[:m.Stages[i].Residual.Conv1.Out*length]
		if err := m.Stages[i].Residual.Conv1.ForwardInto(conv1, hidden, length, &state.Stages[i].Residual.Conv1, state.ConvScratch); err != nil {
			return err
		}
		hidden = conv1
		if !simd.ELUF32To(hidden, hidden) {
			return fmt.Errorf("Pocket TTS Mimi residual ELU2 failed")
		}
		conv2 := state.WorkA[:m.Stages[i].Residual.Conv2.Out*length]
		if err := m.Stages[i].Residual.Conv2.ForwardInto(conv2, hidden, length, &state.Stages[i].Residual.Conv2, state.ConvScratch); err != nil {
			return err
		}
		hidden = conv2
		if !simd.VecAddTo(hidden, hidden, residual) {
			return fmt.Errorf("Pocket TTS Mimi residual add failed")
		}
	}
	if !simd.ELUF32To(hidden, hidden) {
		return fmt.Errorf("Pocket TTS Mimi final ELU failed")
	}
	if err := m.Final.ForwardInto(out, hidden, length, &state.Final, state.ConvScratch); err != nil {
		return err
	}
	if length != SamplesPerFrame {
		return fmt.Errorf("Pocket TTS Mimi samples=%d want=%d", length, SamplesPerFrame)
	}
	return nil
}
func channelToTimePocket(x []float32, channels, length int) []float32 {
	out := make([]float32, len(x))
	for c := 0; c < channels; c++ {
		for t := 0; t < length; t++ {
			out[t*channels+c] = x[c*length+t]
		}
	}
	return out
}
func timeToChannelPocketInto(out, x []float32, length, channels int) {
	for t := 0; t < length; t++ {
		for c := 0; c < channels; c++ {
			out[c*length+t] = x[t*channels+c]
		}
	}
}
func timeToChannelPocket(x []float32, length, channels int) []float32 {
	out := make([]float32, len(x))
	for t := 0; t < length; t++ {
		for c := 0; c < channels; c++ {
			out[c*length+t] = x[t*channels+c]
		}
	}
	return out
}
