package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// FlowLMTrainingCPU is the F32-owned stateless training boundary around the
// FlowLM conditioner, shifted audio input, causal transformer and EOS head.
// FlowHead training is composed separately after this boundary produces Z.
type FlowLMTrainingCPU struct {
	Embedding                     []float32
	Vocabulary, Hidden, LatentDim int
	BOS                           []float32
	BOSBeforeVoice                []float32
	LatentMean, LatentStd         []float32
	SpeakerProjection             LinearF32
	Input                         LinearF32
	Transformer                   *TransformerCPU
	EOS                           LinearF32
}

// FlowLMTrainingBatch is one unpadded training row. NormalizedLatents contains
// [T,C], VoiceLatents [Tv,C], and TextTokens selects embedding rows.
type FlowLMTrainingBatch struct {
	Frames, VoiceFrames int
	NormalizedLatents   []float32
	VoiceLatents        []float32
	TextTokens          []uint32
}

// FlowLMTrainingOutput contains one transformer condition and EOS logit per
// audio frame. SequenceRows records the assembled prefix+audio length.
type FlowLMTrainingOutput struct {
	Z            []float32
	EOS          []float32
	Sequence     []float32
	PrefixRows   int
	SequenceRows int
}

// FlowLMTrainingGradients mirrors every trainable parameter owned by the
// conditioning/backbone/EOS boundary.
type FlowLMTrainingGradients struct {
	Embedding         []float32
	BOS               []float32
	BOSBeforeVoice    []float32
	SpeakerProjection LinearF32Gradient
	Input             LinearF32Gradient
	Transformer       *TransformerGradients
	EOS               LinearF32Gradient
}

// FlowLMTrainingInputGradients returns gradients for caller-owned normalized
// audio and voice latents. Text IDs are discrete; their embedding rows are in
// FlowLMTrainingGradients.Embedding.
type FlowLMTrainingInputGradients struct {
	NormalizedLatents []float32
	VoiceLatents      []float32
}

// ForwardBackward evaluates the exact upstream one-row sequence layout:
// [bos_before_voice, voice, text, input_linear(bos, audio[:-1])]. dZ and dEOS
// are independent seeds over [T,H] and [T].
func (m *FlowLMTrainingCPU) ForwardBackward(batch FlowLMTrainingBatch, dZ, dEOS []float32) (FlowLMTrainingOutput, *FlowLMTrainingGradients, FlowLMTrainingInputGradients, error) {
	tape, err := m.forwardTrainingTape(batch, dZ, dEOS)
	if err != nil {
		return FlowLMTrainingOutput{}, nil, FlowLMTrainingInputGradients{}, err
	}
	gradients, inputGradients, err := m.backwardTrainingTape(tape, batch, dZ, dEOS)
	return tape.output, gradients, inputGradients, err
}

type flowLMTrainingTape struct {
	rows, prefix, textOffset int
	sequence, audioInput     []float32
	transformer              *transformerTape
	output                   FlowLMTrainingOutput
}

func (m *FlowLMTrainingCPU) forwardTrainingTape(batch FlowLMTrainingBatch, dZ, dEOS []float32) (*flowLMTrainingTape, error) {
	rows, prefix, err := m.validate(batch, dZ, dEOS)
	if err != nil {
		return nil, err
	}
	h, c := m.Hidden, m.LatentDim
	sequenceElements, _ := checked.MulInt(rows, h)
	voiceHiddenElements, _ := checked.MulInt(batch.VoiceFrames, h)
	audioLatentElements, _ := checked.MulInt(batch.Frames, c)
	audioHiddenElements, _ := checked.MulInt(batch.Frames, h)
	sequence := make([]float32, sequenceElements)
	copy(sequence[:h], m.BOSBeforeVoice)
	voiceEmbeddings := make([]float32, voiceHiddenElements)
	for row := 0; row < batch.VoiceFrames; row++ {
		embedding := linearForwardTraining(m.SpeakerProjection, batch.VoiceLatents[row*c:(row+1)*c])
		copy(voiceEmbeddings[row*h:(row+1)*h], embedding)
		copy(sequence[(1+row)*h:(2+row)*h], embedding)
	}
	textOffset := 1 + batch.VoiceFrames
	for row, token := range batch.TextTokens {
		copy(sequence[(textOffset+row)*h:(textOffset+row+1)*h], m.Embedding[int(token)*h:(int(token)+1)*h])
	}
	audioInput := make([]float32, audioLatentElements)
	copy(audioInput[:c], m.BOS)
	if batch.Frames > 1 {
		copy(audioInput[c:], batch.NormalizedLatents[:(batch.Frames-1)*c])
	}
	audioEmbeddings := make([]float32, audioHiddenElements)
	for row := 0; row < batch.Frames; row++ {
		embedding := linearForwardTraining(m.Input, audioInput[row*c:(row+1)*c])
		copy(audioEmbeddings[row*h:(row+1)*h], embedding)
		copy(sequence[(prefix+row)*h:(prefix+row+1)*h], embedding)
	}
	zeroSeed := make([]float32, len(sequence))
	if err = validateTrainableTransformer(m.Transformer, sequence, zeroSeed, rows); err != nil {
		return nil, err
	}
	transformerTape, transformed := m.Transformer.forwardTraining(rows, sequence)
	output := FlowLMTrainingOutput{Z: make([]float32, audioHiddenElements), EOS: make([]float32, batch.Frames), Sequence: append([]float32(nil), sequence...), PrefixRows: prefix, SequenceRows: rows}
	for row := 0; row < batch.Frames; row++ {
		z := transformed[(prefix+row)*h : (prefix+row+1)*h]
		copy(output.Z[row*h:(row+1)*h], z)
		output.EOS[row] = linearForwardTraining(m.EOS, z)[0]
	}
	return &flowLMTrainingTape{rows: rows, prefix: prefix, textOffset: textOffset, sequence: sequence, audioInput: audioInput, transformer: transformerTape, output: output}, nil
}

func (m *FlowLMTrainingCPU) backwardTrainingTape(tape *flowLMTrainingTape, batch FlowLMTrainingBatch, dZ, dEOS []float32) (*FlowLMTrainingGradients, FlowLMTrainingInputGradients, error) {
	if tape == nil {
		return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("nil Pocket TTS FlowLM training tape")
	}
	h, c := m.Hidden, m.LatentDim
	gradients := &FlowLMTrainingGradients{Embedding: make([]float32, len(m.Embedding)), BOS: make([]float32, len(m.BOS)), BOSBeforeVoice: make([]float32, len(m.BOSBeforeVoice)), SpeakerProjection: newLinearGradient(m.SpeakerProjection), Input: newLinearGradient(m.Input), EOS: newLinearGradient(m.EOS)}
	inputGradients := FlowLMTrainingInputGradients{NormalizedLatents: make([]float32, len(batch.NormalizedLatents)), VoiceLatents: make([]float32, len(batch.VoiceLatents))}
	dTransformed := make([]float32, tape.rows*h)
	for row := 0; row < batch.Frames; row++ {
		z := tape.output.Z[row*h : (row+1)*h]
		dEOSInput := linearBackwardTraining(m.EOS, z, []float32{dEOS[row]}, &gradients.EOS)
		for i := 0; i < h; i++ {
			dTransformed[(tape.prefix+row)*h+i] = dZ[row*h+i] + dEOSInput[i]
		}
	}
	transformerGradients := newTransformerGradients(m.Transformer)
	dSequence := m.Transformer.backwardTraining(tape.rows, tape.transformer, dTransformed, transformerGradients)
	gradients.Transformer = transformerGradients
	copy(gradients.BOSBeforeVoice, dSequence[:h])
	for row := 0; row < batch.VoiceFrames; row++ {
		dVoice := linearBackwardTraining(m.SpeakerProjection, batch.VoiceLatents[row*c:(row+1)*c], dSequence[(1+row)*h:(2+row)*h], &gradients.SpeakerProjection)
		copy(inputGradients.VoiceLatents[row*c:(row+1)*c], dVoice)
	}
	for row, token := range batch.TextTokens {
		base := int(token) * h
		for i := 0; i < h; i++ {
			gradients.Embedding[base+i] += dSequence[(tape.textOffset+row)*h+i]
		}
	}
	for row := 0; row < batch.Frames; row++ {
		dAudio := linearBackwardTraining(m.Input, tape.audioInput[row*c:(row+1)*c], dSequence[(tape.prefix+row)*h:(tape.prefix+row+1)*h], &gradients.Input)
		if row == 0 {
			addInPlace(gradients.BOS, dAudio)
		} else {
			copy(inputGradients.NormalizedLatents[(row-1)*c:row*c], dAudio)
		}
	}
	return gradients, inputGradients, nil
}

func (m *FlowLMTrainingCPU) validate(batch FlowLMTrainingBatch, dZ, dEOS []float32) (rows, prefix int, err error) {
	if m == nil || m.Hidden <= 0 || m.LatentDim <= 0 || m.Vocabulary <= 0 || batch.Frames <= 0 || batch.VoiceFrames < 0 || len(batch.TextTokens) == 0 || m.Transformer == nil {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM training boundary")
	}
	embeddingElements, ok := checked.MulInt(m.Vocabulary, m.Hidden)
	if !ok || len(m.Embedding) != embeddingElements || len(m.BOS) != m.LatentDim || len(m.BOSBeforeVoice) != m.Hidden || m.SpeakerProjection.In != m.LatentDim || m.SpeakerProjection.Out != m.Hidden || m.Input.In != m.LatentDim || m.Input.Out != m.Hidden || m.EOS.In != m.Hidden || m.EOS.Out != 1 {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM training topology")
	}
	if err := validateOwnedTrainingLinear(m.SpeakerProjection, false); err != nil {
		return 0, 0, err
	}
	if err := validateOwnedTrainingLinear(m.Input, false); err != nil {
		return 0, 0, err
	}
	if err := validateOwnedTrainingLinear(m.EOS, true); err != nil {
		return 0, 0, err
	}
	latentElements, ok := checked.MulInt(batch.Frames, m.LatentDim)
	if !ok || len(batch.NormalizedLatents) != latentElements {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM audio shape")
	}
	voiceElements, ok := checked.MulInt(batch.VoiceFrames, m.LatentDim)
	if !ok || len(batch.VoiceLatents) != voiceElements {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM voice shape")
	}
	zElements, ok := checked.MulInt(batch.Frames, m.Hidden)
	if !ok || len(dZ) != zElements || len(dEOS) != batch.Frames {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM gradient seed shape")
	}
	prefix, ok = checked.AddInt(1, batch.VoiceFrames)
	if !ok {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM prefix shape")
	}
	prefix, ok = checked.AddInt(prefix, len(batch.TextTokens))
	if !ok {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM prefix shape")
	}
	rows, ok = checked.AddInt(prefix, batch.Frames)
	if !ok {
		return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM sequence shape")
	}
	for _, shape := range [][2]int{{rows, m.Hidden}, {batch.VoiceFrames, m.Hidden}, {batch.Frames, m.LatentDim}, {batch.Frames, m.Hidden}} {
		if _, ok = checked.MulInt(shape[0], shape[1]); !ok {
			return 0, 0, fmt.Errorf("invalid Pocket TTS FlowLM allocation shape")
		}
	}
	if m.Transformer.Width != m.Hidden {
		return 0, 0, fmt.Errorf("Pocket TTS FlowLM transformer width mismatch")
	}
	for _, token := range batch.TextTokens {
		if uint64(token) >= uint64(m.Vocabulary) {
			return 0, 0, fmt.Errorf("Pocket TTS training token %d out of range", token)
		}
	}
	for _, values := range [][]float32{m.Embedding, m.BOS, m.BOSBeforeVoice, batch.NormalizedLatents, batch.VoiceLatents, dZ, dEOS} {
		for _, value := range values {
			if !isFinite(value) {
				return 0, 0, fmt.Errorf("Pocket TTS FlowLM training value is non-finite")
			}
		}
	}
	return rows, prefix, nil
}

func validateOwnedTrainingLinear(linear LinearF32, requireBias bool) error {
	weightElements, ok := checked.MulInt(linear.In, linear.Out)
	if linear.In <= 0 || linear.Out <= 0 || !ok || len(linear.WeightBF16) != 0 || len(linear.Weight) != weightElements || (requireBias && len(linear.Bias) != linear.Out) || (!requireBias && linear.Bias != nil) {
		return fmt.Errorf("Pocket TTS FlowLM training requires owned F32 linear parameters")
	}
	return nil
}
