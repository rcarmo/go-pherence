package pockettts

import (
	"fmt"
	"unsafe"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type seanetEncoderStageCPU struct {
	Residual seanetResidualCPU
	Down     CausalConv1D
}

// MimiEncoderCPU is the frozen raw-audio encode path used by training and
// voice cloning: SEANet encoder, projected causal transformer, and 16x
// replicate-padded convolutional downsampling into 32-D latents.
type MimiEncoderCPU struct {
	Initial     CausalConv1D
	Stages      []seanetEncoderStageCPU
	Final       CausalConv1D
	Transformer *TransformerCPU
	Downsample  CausalConv1D
	FrameSize   int
}

func LoadMimiEncoderCPU(src *safetensors.File, cfg Config) (*MimiEncoderCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Pocket TTS Mimi encoder source")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	dimension, transformer := cfg.Mimi.SEANet.Dimension, cfg.Mimi.Transformer
	if transformer.DModel != dimension || transformer.InputDimension != dimension || len(transformer.OutputDimensions) != 1 || transformer.OutputDimensions[0] != dimension {
		return nil, fmt.Errorf("unsupported Pocket TTS Mimi encoder projection geometry")
	}
	m := &MimiEncoderCPU{Stages: make([]seanetEncoderStageCPU, len(cfg.Mimi.SEANet.Ratios)), FrameSize: SamplesPerFrame}
	var err error
	channels := cfg.Mimi.SEANet.NFilters
	if m.Initial, err = loadCausalConv(src, "mimi.encoder.model.0.conv", cfg.Mimi.Channels, channels, cfg.Mimi.SEANet.KernelSize, 1, 1); err != nil {
		return nil, err
	}
	// Upstream encoder reverses ratios: [6,5,4] -> strides [4,5,6].
	resIndices, downIndices := []int{1, 4, 7}, []int{3, 6, 9}
	if len(m.Stages) != 3 {
		return nil, fmt.Errorf("unsupported Pocket TTS Mimi encoder stage count %d", len(m.Stages))
	}
	for i := range m.Stages {
		ratio := cfg.Mimi.SEANet.Ratios[len(cfg.Mimi.SEANet.Ratios)-1-i]
		stage := &m.Stages[i]
		if stage.Residual.Conv1, err = loadCausalConv(src, fmt.Sprintf("mimi.encoder.model.%d.block.1.conv", resIndices[i]), channels, channels/cfg.Mimi.SEANet.Compress, cfg.Mimi.SEANet.ResidualKernelSize, 1, 1); err != nil {
			return nil, err
		}
		if stage.Residual.Conv2, err = loadCausalConv(src, fmt.Sprintf("mimi.encoder.model.%d.block.3.conv", resIndices[i]), channels/cfg.Mimi.SEANet.Compress, channels, 1, 1, 1); err != nil {
			return nil, err
		}
		if stage.Down, err = loadCausalConv(src, fmt.Sprintf("mimi.encoder.model.%d.conv", downIndices[i]), channels, 2*channels, 2*ratio, ratio, 1); err != nil {
			return nil, err
		}
		channels *= 2
	}
	if m.Final, err = loadCausalConv(src, "mimi.encoder.model.11.conv", channels, cfg.Mimi.SEANet.Dimension, cfg.Mimi.SEANet.LastKernelSize, 1, 1); err != nil {
		return nil, err
	}
	if m.Transformer, err = LoadTransformerCPU(src, "mimi.encoder_transformer.transformer", cfg.Mimi.Transformer, ""); err != nil {
		return nil, err
	}
	for i := range m.Transformer.Layers {
		p := fmt.Sprintf("mimi.encoder_transformer.transformer.layers.%d", i)
		l := &m.Transformer.Layers[i]
		for _, item := range []struct {
			suffix string
			linear *LinearF32
		}{{"self_attn.in_proj", &l.InProjection}, {"self_attn.out_proj", &l.OutProjection}, {"linear1", &l.FC1}, {"linear2", &l.FC2}} {
			if err = materializeLinearF32(src, p+"."+item.suffix, item.linear); err != nil {
				return nil, err
			}
		}
	}
	if m.Downsample, err = loadCausalConvWithBias(src, "mimi.downsample.conv.conv", cfg.Mimi.SEANet.Dimension, cfg.Mimi.InnerDim, 32, 16, 1, false); err != nil {
		return nil, err
	}
	return m, nil
}

type mimiEncoderStageState struct {
	Residual seanetResidualState
	Down     StreamingConvState
}
type MimiEncoderWorkspace struct {
	encoder                                                  *MimiEncoderCPU
	MaxFrames, ChunkFrames                                   int
	Initial                                                  StreamingConvState
	Stages                                                   []mimiEncoderStageState
	Final, Downsample                                        StreamingConvState
	Transformer                                              *TransformerState
	ChunkScratch                                             *TransformerChunkScratch
	ConvScratch                                              *Scratch
	Audio, A, B, Residual, Time, Transformed, LatentChannels []float32
}

func (m *MimiEncoderCPU) NewWorkspace(maxFrames int) (*MimiEncoderWorkspace, error) {
	if m == nil || m.Transformer == nil || maxFrames <= 0 || m.FrameSize != SamplesPerFrame || len(m.Stages) != 3 || m.Transformer.Width != m.Final.Out || m.Downsample.In != m.Transformer.Width {
		return nil, fmt.Errorf("invalid Pocket TTS Mimi encoder workspace")
	}
	rows, ok := checked.MulInt(maxFrames, 16)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
	}
	chunkFrames := min(maxFrames, 16)
	chunkRows, ok := checked.MulInt(chunkFrames, 16)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
	}
	chunkSamples, ok := checked.MulInt(chunkFrames, m.FrameSize)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
	}
	transformer, err := m.Transformer.NewState(rows)
	if err != nil {
		return nil, err
	}
	chunk, err := m.Transformer.NewChunkScratch(chunkRows)
	if err != nil {
		return nil, err
	}
	maxElements := 0
	grow := func(channels, length int) bool {
		elements, valid := checked.MulInt(channels, length)
		if valid {
			maxElements = max(maxElements, elements)
		}
		return valid
	}
	if !grow(m.Initial.Out, chunkSamples) {
		return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
	}
	convNeed := m.Initial.scratchFloats(chunkSamples)
	length := chunkSamples
	for i := range m.Stages {
		stage := &m.Stages[i]
		if stage.Down.Stride <= 0 || length%stage.Down.Stride != 0 || !grow(stage.Residual.Conv1.Out, length) || !grow(stage.Residual.Conv2.Out, length) {
			return nil, fmt.Errorf("invalid Pocket TTS Mimi encoder stage geometry")
		}
		convNeed = max(convNeed, stage.Residual.Conv1.scratchFloats(length), stage.Residual.Conv2.scratchFloats(length), stage.Down.scratchFloats(length))
		length /= stage.Down.Stride
		if !grow(stage.Down.Out, length) {
			return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
		}
	}
	if length != chunkRows || !grow(m.Final.Out, length) || !grow(m.Transformer.Width, length) || !grow(m.Downsample.Out, chunkFrames) {
		return nil, fmt.Errorf("invalid Pocket TTS Mimi encoder output geometry")
	}
	convNeed = max(convNeed, m.Final.scratchFloats(length), m.Downsample.scratchFloats(chunkRows))
	if convNeed <= 0 {
		return nil, fmt.Errorf("invalid Pocket TTS Mimi encoder scratch geometry")
	}
	timeElements, ok := checked.MulInt(m.Transformer.Width, chunkRows)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
	}
	latentElements, ok := checked.MulInt(m.Downsample.Out, chunkFrames)
	if !ok {
		return nil, fmt.Errorf("Pocket TTS Mimi encoder capacity overflows")
	}
	scratch, err := NewScratch(convNeed)
	if err != nil {
		return nil, err
	}
	w := &MimiEncoderWorkspace{encoder: m, MaxFrames: maxFrames, ChunkFrames: chunkFrames, Initial: m.Initial.NewState(), Stages: make([]mimiEncoderStageState, len(m.Stages)), Final: m.Final.NewState(), Downsample: m.Downsample.NewState(), Transformer: transformer, ChunkScratch: chunk, ConvScratch: scratch, Audio: make([]float32, chunkSamples), A: make([]float32, maxElements), B: make([]float32, maxElements), Residual: make([]float32, maxElements), Time: make([]float32, timeElements), Transformed: make([]float32, timeElements), LatentChannels: make([]float32, latentElements)}
	for i := range m.Stages {
		w.Stages[i] = mimiEncoderStageState{Residual: seanetResidualState{Conv1: m.Stages[i].Residual.Conv1.NewState(), Conv2: m.Stages[i].Residual.Conv2.NewState()}, Down: m.Stages[i].Down.NewState()}
	}
	return w, nil
}

func resetConvState(s *StreamingConvState) { clear(s.Previous); s.First = true }
func (w *MimiEncoderWorkspace) reset() {
	resetConvState(&w.Initial)
	for i := range w.Stages {
		resetConvState(&w.Stages[i].Residual.Conv1)
		resetConvState(&w.Stages[i].Residual.Conv2)
		resetConvState(&w.Stages[i].Down)
	}
	resetConvState(&w.Final)
	resetConvState(&w.Downsample)
	w.Transformer.Position = 0
	for i := range w.Transformer.Layers {
		w.Transformer.Layers[i].Length = 0
	}
	w.ConvScratch.Reset()
}

// Encode converts mono waveform samples to row-major [frames,latent]. Input is
// zero-padded to an exact 1,920-sample frame multiple, matching upstream.
func (m *MimiEncoderCPU) Encode(audio []float32) ([]float32, int, error) {
	if m == nil || len(audio) == 0 {
		return nil, 0, fmt.Errorf("invalid Pocket TTS Mimi encoder input")
	}
	frames := 1 + (len(audio)-1)/m.FrameSize
	outElements, ok := checked.MulInt(frames, m.Downsample.Out)
	if !ok {
		return nil, 0, fmt.Errorf("Pocket TTS Mimi encoder output overflows")
	}
	workspace, err := m.NewWorkspace(frames)
	if err != nil {
		return nil, 0, err
	}
	out := make([]float32, outElements)
	if err = m.EncodeInto(out, audio, workspace); err != nil {
		return nil, 0, err
	}
	return out, frames, nil
}

func f32SlicesOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	a0, b0 := uintptr(unsafe.Pointer(unsafe.SliceData(a))), uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	a1, b1 := a0+uintptr(len(a))*unsafe.Sizeof(a[0]), b0+uintptr(len(b))*unsafe.Sizeof(b[0])
	return a0 < b1 && b0 < a1
}

func (w *MimiEncoderWorkspace) overlapsWorkspace(x []float32) bool {
	if w == nil {
		return false
	}
	if f32SlicesOverlap(x, w.Audio) || f32SlicesOverlap(x, w.A) || f32SlicesOverlap(x, w.B) || f32SlicesOverlap(x, w.Residual) || f32SlicesOverlap(x, w.Time) || f32SlicesOverlap(x, w.Transformed) || f32SlicesOverlap(x, w.LatentChannels) || f32SlicesOverlap(x, w.Initial.Previous) || f32SlicesOverlap(x, w.Final.Previous) || f32SlicesOverlap(x, w.Downsample.Previous) {
		return true
	}
	for i := range w.Stages {
		stage := &w.Stages[i]
		if f32SlicesOverlap(x, stage.Residual.Conv1.Previous) || f32SlicesOverlap(x, stage.Residual.Conv2.Previous) || f32SlicesOverlap(x, stage.Down.Previous) {
			return true
		}
	}
	if w.ConvScratch != nil && f32SlicesOverlap(x, w.ConvScratch.values) {
		return true
	}
	if t := w.Transformer; t != nil {
		if f32SlicesOverlap(x, t.Rope) || f32SlicesOverlap(x, t.RopeCos) || f32SlicesOverlap(x, t.RopeSin) || f32SlicesOverlap(x, t.RopeReal) || f32SlicesOverlap(x, t.RopeImag) || f32SlicesOverlap(x, t.RopeRC) || f32SlicesOverlap(x, t.RopeIS) || f32SlicesOverlap(x, t.RopeIC) || f32SlicesOverlap(x, t.RopeRS) || f32SlicesOverlap(x, t.HiddenA) || f32SlicesOverlap(x, t.HiddenB) {
			return true
		}
		for i := range t.Layers {
			l := &t.Layers[i]
			if f32SlicesOverlap(x, l.Keys) || f32SlicesOverlap(x, l.Values) || f32SlicesOverlap(x, l.Norm) || f32SlicesOverlap(x, l.QKV) || f32SlicesOverlap(x, l.Attention) || f32SlicesOverlap(x, l.Scores) || f32SlicesOverlap(x, l.Projected) || f32SlicesOverlap(x, l.Output) || f32SlicesOverlap(x, l.FFNorm) || f32SlicesOverlap(x, l.Wide) || f32SlicesOverlap(x, l.Down) {
				return true
			}
		}
	}
	if s := w.ChunkScratch; s != nil {
		return f32SlicesOverlap(x, s.Norm) || f32SlicesOverlap(x, s.QKV) || f32SlicesOverlap(x, s.Attention) || f32SlicesOverlap(x, s.Projected) || f32SlicesOverlap(x, s.Output) || f32SlicesOverlap(x, s.FFNorm) || f32SlicesOverlap(x, s.Wide) || f32SlicesOverlap(x, s.Down) || f32SlicesOverlap(x, s.HiddenA) || f32SlicesOverlap(x, s.HiddenB)
	}
	return false
}

// EncodeInto reuses request-owned state and bounded 16-frame chunk buffers.
// Results are written row-major [frames,latent]. The non-concurrent workspace,
// audio, and written output region must have disjoint backing storage.
func (m *MimiEncoderCPU) EncodeInto(out, audio []float32, w *MimiEncoderWorkspace) error {
	if m == nil || w == nil || w.encoder != m || len(audio) == 0 || m.FrameSize <= 0 || !finiteF32(audio) {
		return fmt.Errorf("invalid Pocket TTS Mimi encoder input")
	}
	frames := 1 + (len(audio)-1)/m.FrameSize
	outElements, ok := checked.MulInt(frames, m.Downsample.Out)
	if !ok || frames > w.MaxFrames || len(out) < outElements {
		return fmt.Errorf("Pocket TTS Mimi encoder workspace capacity exceeded")
	}
	written := out[:outElements]
	if f32SlicesOverlap(written, audio) || w.overlapsWorkspace(audio) || w.overlapsWorkspace(written) {
		return fmt.Errorf("Pocket TTS Mimi encoder buffers overlap")
	}
	w.reset()
	for frame := 0; frame < frames; frame += w.ChunkFrames {
		chunkFrames := min(w.ChunkFrames, frames-frame)
		clear(w.Audio)
		start := frame * m.FrameSize
		chunkSamples := chunkFrames * m.FrameSize
		end := min(len(audio), start+chunkSamples)
		copy(w.Audio[:chunkSamples], audio[start:end])
		length := chunkSamples
		hidden := w.A[:m.Initial.Out*length]
		if err := m.Initial.ForwardInto(hidden, w.Audio[:length], length, &w.Initial, w.ConvScratch); err != nil {
			return err
		}
		for i := range m.Stages {
			copy(w.Residual[:len(hidden)], hidden)
			if !simd.ELUF32To(hidden, hidden) {
				return fmt.Errorf("Pocket TTS Mimi encoder residual ELU failed")
			}
			branch := w.B[:m.Stages[i].Residual.Conv1.Out*length]
			if err := m.Stages[i].Residual.Conv1.ForwardInto(branch, hidden, length, &w.Stages[i].Residual.Conv1, w.ConvScratch); err != nil {
				return err
			}
			if !simd.ELUF32To(branch, branch) {
				return fmt.Errorf("Pocket TTS Mimi encoder residual ELU2 failed")
			}
			updated := w.A[:m.Stages[i].Residual.Conv2.Out*length]
			if err := m.Stages[i].Residual.Conv2.ForwardInto(updated, branch, length, &w.Stages[i].Residual.Conv2, w.ConvScratch); err != nil {
				return err
			}
			if !simd.VecAddTo(updated, updated, w.Residual[:len(updated)]) || !simd.ELUF32To(updated, updated) {
				return fmt.Errorf("Pocket TTS Mimi encoder residual update failed")
			}
			nextLength := length / m.Stages[i].Down.Stride
			down := w.B[:m.Stages[i].Down.Out*nextLength]
			if err := m.Stages[i].Down.ForwardInto(down, updated, length, &w.Stages[i].Down, w.ConvScratch); err != nil {
				return err
			}
			hidden, length = down, nextLength
		}
		if !simd.ELUF32To(hidden, hidden) {
			return fmt.Errorf("Pocket TTS Mimi encoder final ELU failed")
		}
		final := w.A[:m.Final.Out*length]
		if err := m.Final.ForwardInto(final, hidden, length, &w.Final, w.ConvScratch); err != nil {
			return err
		}
		channelToTimePocketInto(w.Time[:len(final)], final, m.Final.Out, length)
		if err := m.Transformer.ForwardChunkInto(w.Transformed[:len(final)], w.Time[:len(final)], length, w.Transformer, w.ChunkScratch); err != nil {
			return err
		}
		channelMajor := w.A[:len(final)]
		timeToChannelPocketInto(channelMajor, w.Transformed[:len(final)], length, m.Transformer.Width)
		if frame == 0 {
			prevPer := len(w.Downsample.Previous) / m.Downsample.In
			for c := 0; c < m.Downsample.In; c++ {
				first := channelMajor[c*length]
				for i := 0; i < prevPer; i++ {
					w.Downsample.Previous[c*prevPer+i] = first
				}
			}
			w.Downsample.First = false
		}
		latentChannels := w.LatentChannels[:m.Downsample.Out*chunkFrames]
		if err := m.Downsample.ForwardInto(latentChannels, channelMajor, length, &w.Downsample, w.ConvScratch); err != nil {
			return err
		}
		for c := 0; c < m.Downsample.Out; c++ {
			for f := 0; f < chunkFrames; f++ {
				out[(frame+f)*m.Downsample.Out+c] = latentChannels[c*chunkFrames+f]
			}
		}
	}
	return nil
}

func coldConvForward(conv CausalConv1D, input []float32, length int) ([]float32, int, error) {
	state := conv.NewState()
	return conv.Forward(input, length, &state)
}
