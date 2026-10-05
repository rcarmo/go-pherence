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
	// Optional explicit hybrid training backend; nil keeps CPU arithmetic.
	// No automatic device initialisation or fallback on backend error.
	AttentionBackwardBackend TrainingAttentionBackend
	// Complete transformer tape offload; cannot combine with attention-only
	// offload. Runner replay uses a separately named transformer-hybrid scope.
	TransformerBackend TrainingTransformerBackend
	ProjectionBackend  TrainingProjectionBackend
	SequenceBackend    TrainingSequenceBackend
	EOSBackend         TrainingEOSBackend
	EOS                LinearF32
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
	transformerBackend       TrainingTransformerBackend
	projectionBackend        TrainingProjectionBackend
	sequenceBackend          TrainingSequenceBackend
	output                   FlowLMTrainingOutput
	fusedEOS                 bool
	audioFusion              TrainingAudioProjectionBackend
	voiceFusion              TrainingVoiceSequenceBackend
	conditioner              TrainingConditionerBackend
	backbone                 TrainingBackboneBackend
	fusedEOSInput            []float32
	eosBackbone              TrainingEOSBackboneBackend
	eosMask                  []bool
	eosWeight                float32
}

func (m *FlowLMTrainingCPU) forwardTrainingTape(batch FlowLMTrainingBatch, dZ, dEOS []float32) (*flowLMTrainingTape, error) {
	return m.forwardTrainingTapeMode(batch, dZ, dEOS, false)
}

func (m *FlowLMTrainingCPU) forwardTrainingTapeMode(batch FlowLMTrainingBatch, dZ, dEOS []float32, allowEmptyText bool) (*flowLMTrainingTape, error) {
	return m.forwardTrainingTapeEOSMode(batch, dZ, dEOS, allowEmptyText, false)
}
func (m *FlowLMTrainingCPU) forwardTrainingTapeEOSMode(batch FlowLMTrainingBatch, dZ, dEOS []float32, allowEmptyText, skipEOS bool) (*flowLMTrainingTape, error) {
	rows, prefix, err := m.validateMode(batch, dZ, dEOS, allowEmptyText)
	if err != nil {
		return nil, err
	}
	if m.SequenceBackend != nil && m.ProjectionBackend == nil {
		return nil, fmt.Errorf("sequence backend requires resident projections")
	}
	h, c := m.Hidden, m.LatentDim
	sequenceElements, _ := checked.MulInt(rows, h)
	voiceHiddenElements, _ := checked.MulInt(batch.VoiceFrames, h)
	audioLatentElements, _ := checked.MulInt(batch.Frames, c)
	audioHiddenElements, _ := checked.MulInt(batch.Frames, h)
	sequence := make([]float32, sequenceElements)
	assembly := m.SequenceBackend
	if assembly == nil {
		copy(sequence[:h], m.BOSBeforeVoice)
	}
	projection := m.ProjectionBackend
	conditioner, combined := assembly.(TrainingConditionerBackend)
	combined = combined && conditioner.ConditionerEnabled()
	if !combined {
		conditioner = nil
	}
	var audioInput []float32
	var audioFusion TrainingAudioProjectionBackend
	var voiceFusion TrainingVoiceSequenceBackend
	textOffset := 1 + batch.VoiceFrames
	backbone, fusedBackbone := assembly.(TrainingBackboneBackend)
	fusedBackbone = fusedBackbone && backbone.BackboneEnabled()
	if !fusedBackbone {
		backbone = nil
	}
	var transformed []float32
	if fusedBackbone {
		if m.AttentionBackwardBackend != nil || m.TransformerBackend == nil {
			return nil, fmt.Errorf("backbone requires exclusive transformer owner")
		}
		r, e := backbone.EvaluateBackbone(m, batch, make([]float32, len(sequence)), false)
		if e != nil {
			return nil, e
		}
		if e = validateBackboneResult(r, m, batch, make([]float32, len(sequence))); e != nil {
			return nil, e
		}
		copy(sequence, r.Sequence)
		transformed = r.Output
	} else if combined {
		r, e := conditioner.EvaluateConditioner(m, batch, make([]float32, len(sequence)), false)
		if e != nil {
			return nil, e
		}
		if e = validateSequenceResult(r, len(sequence), batch.VoiceFrames*c, batch.Frames*c, batch.VoiceLatents, batch.NormalizedLatents); e != nil {
			return nil, e
		}
		copy(sequence, r.Sequence)
	} else {
		var fusedVoice bool
		voiceFusion, fusedVoice = assembly.(TrainingVoiceSequenceBackend)
		fusedVoice = fusedVoice && voiceFusion.VoiceSequenceEnabled()
		if !fusedVoice {
			voiceFusion = nil
		}
		var voiceEmbeddings []float32
		if !fusedVoice {
			voiceEmbeddings = make([]float32, voiceHiddenElements)
			if projection != nil && batch.VoiceFrames > 0 {
				r, e := projection.EvaluateProjection(m, "speaker", batch.VoiceLatents, make([]float32, voiceHiddenElements), batch.VoiceFrames, false)
				if e != nil {
					return nil, e
				}
				if e := validateProjectionResult(r, batch.VoiceFrames, c, h, batch.VoiceLatents); e != nil {
					return nil, e
				}
				copy(voiceEmbeddings, r.Output)
			} else {
				for row := 0; row < batch.VoiceFrames; row++ {
					copy(voiceEmbeddings[row*h:(row+1)*h], linearForwardTraining(m.SpeakerProjection, batch.VoiceLatents[row*c:(row+1)*c]))
				}
			}
		}
		if assembly == nil {
			copy(sequence[h:(1+batch.VoiceFrames)*h], voiceEmbeddings)
		}
		textOffset = 1 + batch.VoiceFrames
		if assembly == nil {
			for row, token := range batch.TextTokens {
				copy(sequence[(textOffset+row)*h:(textOffset+row+1)*h], m.Embedding[int(token)*h:(int(token)+1)*h])
			}
		}
		var fusedAudio bool
		audioFusion, fusedAudio = assembly.(TrainingAudioProjectionBackend)
		fusedAudio = fusedAudio && audioFusion.AudioProjectionEnabled()
		if !fusedAudio {
			audioFusion = nil
		}
		if !fusedAudio {
			audioInput = make([]float32, audioLatentElements)
			if assembly != nil {
				r, e := assembly.EvaluateAudioAssembly(m, batch, make([]float32, audioLatentElements), false)
				if e != nil {
					return nil, e
				}
				if e := validateAudioAssemblyResult(r, audioLatentElements, batch.NormalizedLatents); e != nil {
					return nil, e
				}
				copy(audioInput, r.AudioInput)
			} else {
				copy(audioInput[:c], m.BOS)
				if batch.Frames > 1 {
					copy(audioInput[c:], batch.NormalizedLatents[:(batch.Frames-1)*c])
				}
			}
		}
		audioEmbeddings := make([]float32, audioHiddenElements)
		if fusedAudio {
			r, e := audioFusion.EvaluateAudioProjection(m, batch, make([]float32, audioHiddenElements), false)
			if e != nil {
				return nil, e
			}
			if e = validateAudioProjectionBoundary(r, batch.Frames, h, c, batch.NormalizedLatents); e != nil {
				return nil, e
			}
			copy(audioEmbeddings, r.Output)
		} else if projection != nil {
			r, e := projection.EvaluateProjection(m, "input", audioInput, make([]float32, audioHiddenElements), batch.Frames, false)
			if e != nil {
				return nil, e
			}
			if e := validateProjectionResult(r, batch.Frames, c, h, audioInput); e != nil {
				return nil, e
			}
			copy(audioEmbeddings, r.Output)
		} else {
			for row := 0; row < batch.Frames; row++ {
				copy(audioEmbeddings[row*h:(row+1)*h], linearForwardTraining(m.Input, audioInput[row*c:(row+1)*c]))
			}
		}
		if assembly == nil {
			copy(sequence[prefix*h:], audioEmbeddings)
		}
		zeroSeed := make([]float32, len(sequence))
		if fusedVoice {
			r, e := voiceFusion.EvaluateVoiceSequence(m, batch, audioEmbeddings, zeroSeed, false)
			if e != nil {
				return nil, e
			}
			if e = validateSequenceResult(r, len(sequence), batch.VoiceFrames*c, audioHiddenElements, batch.VoiceLatents, audioEmbeddings, zeroSeed); e != nil {
				return nil, e
			}
			copy(sequence, r.Sequence)
		} else if assembly != nil {
			r, e := assembly.EvaluateSequence(m, batch, voiceEmbeddings, audioEmbeddings, zeroSeed, false)
			if e != nil {
				return nil, e
			}
			if e := validateSequenceResult(r, len(sequence), voiceHiddenElements, audioHiddenElements, voiceEmbeddings, audioEmbeddings, zeroSeed); e != nil {
				return nil, e
			}
			copy(sequence, r.Sequence)
		}
	}
	zeroSeed := make([]float32, len(sequence))
	if err = validateTrainableTransformer(m.Transformer, sequence, zeroSeed, rows); err != nil {
		return nil, err
	}
	var transformerTape *transformerTape
	backend := m.TransformerBackend
	if fusedBackbone {
		// The composed tape already evaluated the transformer without a host edge.
	} else if backend != nil {
		if m.AttentionBackwardBackend != nil {
			return nil, fmt.Errorf("combined transformer/attention backends are not admitted")
		}
		result, err := evaluateTrainingTransformer(backend, m.Transformer, sequence, zeroSeed, rows, false)
		if err != nil {
			return nil, err
		}
		_, resident := backend.(TrainingTransformerAccumulator)
		if err := validateTransformerBackendResultMode(m.Transformer, result, sequence, zeroSeed, rows, !resident); err != nil {
			return nil, err
		}
		transformed = result.Output
	} else {
		transformerTape, transformed = m.Transformer.forwardTraining(rows, sequence)
	}
	output := FlowLMTrainingOutput{Z: make([]float32, audioHiddenElements), EOS: make([]float32, batch.Frames), Sequence: append([]float32(nil), sequence...), PrefixRows: prefix, SequenceRows: rows}
	for row := 0; row < batch.Frames; row++ {
		z := transformed[(prefix+row)*h : (prefix+row+1)*h]
		copy(output.Z[row*h:(row+1)*h], z)
		if !skipEOS && projection == nil {
			output.EOS[row] = linearForwardTraining(m.EOS, z)[0]
		}
	}
	if !skipEOS && projection != nil {
		r, e := projection.EvaluateProjection(m, "eos", output.Z, make([]float32, batch.Frames), batch.Frames, false)
		if e != nil {
			return nil, e
		}
		if e := validateProjectionResult(r, batch.Frames, h, 1, output.Z); e != nil {
			return nil, e
		}
		copy(output.EOS, r.Output)
	}
	return &flowLMTrainingTape{rows: rows, prefix: prefix, textOffset: textOffset, sequence: sequence, audioInput: audioInput, transformer: transformerTape, transformerBackend: backend, projectionBackend: projection, sequenceBackend: assembly, output: output, fusedEOS: skipEOS, audioFusion: audioFusion, voiceFusion: voiceFusion, conditioner: conditioner, backbone: backbone}, nil
}

func newFlowLMTrainingGradients(m *FlowLMTrainingCPU) *FlowLMTrainingGradients {
	return newFlowLMGradientStorage(m, func(n int) []float32 { return make([]float32, n) })
}
func newFlowLMGradientStorage(m *FlowLMTrainingCPU, storage func(int) []float32) *FlowLMTrainingGradients {
	if m == nil {
		return nil
	}
	linear := func(l LinearF32) LinearF32Gradient { return newLinearGradientStorage(l, storage) }
	return &FlowLMTrainingGradients{Embedding: storage(len(m.Embedding)), BOS: storage(len(m.BOS)), BOSBeforeVoice: storage(len(m.BOSBeforeVoice)), SpeakerProjection: linear(m.SpeakerProjection), Input: linear(m.Input), Transformer: newTransformerGradientStorage(m.Transformer, storage), EOS: linear(m.EOS)}
}

func clearTransformerGradients(g *TransformerGradients) {
	if g == nil {
		return
	}
	for i := range g.Layers {
		layer := &g.Layers[i]
		clear(layer.Norm1Weight)
		clear(layer.Norm1Bias)
		clear(layer.Norm2Weight)
		clear(layer.Norm2Bias)
		clearLinearGradient(&layer.InProjection)
		clearLinearGradient(&layer.OutProjection)
		clearLinearGradient(&layer.FC1)
		clearLinearGradient(&layer.FC2)
		clear(layer.LayerScale1)
		clear(layer.LayerScale2)
	}
	clear(g.FinalWeight)
	clear(g.FinalBias)
}

func clearFlowLMTrainingGradients(g *FlowLMTrainingGradients) {
	if g == nil {
		return
	}
	clear(g.Embedding)
	clear(g.BOS)
	clear(g.BOSBeforeVoice)
	clearLinearGradient(&g.SpeakerProjection)
	clearLinearGradient(&g.Input)
	clearLinearGradient(&g.EOS)
	clearTransformerGradients(g.Transformer)
}

func (m *FlowLMTrainingCPU) backwardTrainingTape(tape *flowLMTrainingTape, batch FlowLMTrainingBatch, dZ, dEOS []float32) (*FlowLMTrainingGradients, FlowLMTrainingInputGradients, error) {
	gradients := newFlowLMTrainingGradients(m)
	inputGradients := FlowLMTrainingInputGradients{NormalizedLatents: make([]float32, len(batch.NormalizedLatents)), VoiceLatents: make([]float32, len(batch.VoiceLatents))}
	return m.backwardTrainingTapeInto(tape, batch, dZ, dEOS, gradients, inputGradients)
}

func (m *FlowLMTrainingCPU) backwardTrainingTapeInto(tape *flowLMTrainingTape, batch FlowLMTrainingBatch, dZ, dEOS []float32, gradients *FlowLMTrainingGradients, inputGradients FlowLMTrainingInputGradients) (*FlowLMTrainingGradients, FlowLMTrainingInputGradients, error) {
	if tape == nil || gradients == nil || gradients.Transformer == nil || len(gradients.Embedding) != len(m.Embedding) || len(inputGradients.NormalizedLatents) != len(batch.NormalizedLatents) || len(inputGradients.VoiceLatents) != len(batch.VoiceLatents) {
		return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("invalid Pocket TTS FlowLM backward workspace")
	}
	h, c := m.Hidden, m.LatentDim
	assembly := tape.sequenceBackend
	if (assembly == nil) != (m.SequenceBackend == nil) {
		return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("sequence backend changed after forward")
	}
	projection := tape.projectionBackend
	if (projection == nil) != (m.ProjectionBackend == nil) {
		return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("projection backend changed after forward")
	}
	dTransformed := make([]float32, tape.rows*h)
	if tape.eosBackbone != nil {
		current, ok := m.EOSBackend.(TrainingEOSBackboneBackend)
		if !ok || !current.EOSBackboneEnabled() || tape.backbone == nil || !tape.backbone.BackboneEnabled() || !tape.fusedEOS || !zeroTrainingSeed(dEOS) || len(tape.eosMask) != batch.Frames {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("EOS backbone changed/invalid seed after forward")
		}
		for row := 0; row < batch.Frames; row++ {
			copy(dTransformed[(tape.prefix+row)*h:(tape.prefix+row+1)*h], dZ[row*h:(row+1)*h])
		}
		r, e := tape.eosBackbone.EvaluateEOSBackbone(m, batch, dTransformed, tape.output.Z, tape.eosMask, tape.eosWeight)
		if e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		if e = validateBackboneResult(r, m, batch, dTransformed); e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		copy(inputGradients.VoiceLatents, r.DVoice)
		copy(inputGradients.NormalizedLatents, r.DNormalized)
		return gradients, inputGradients, nil
	}
	var eosInput []float32
	if tape.fusedEOS {
		fused, ok := m.EOSBackend.(TrainingEOSProjectionBackend)
		if !ok || !fused.EOSProjectionEnabled() || projection == nil {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("fused EOS backend changed after forward")
		}
		if len(tape.fusedEOSInput) != batch.Frames*h || !finiteF32(tape.fusedEOSInput) || !zeroTrainingSeed(dEOS) {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("invalid fused EOS input VJP/seed")
		}
		eosInput = tape.fusedEOSInput
	} else if projection != nil {
		r, e := projection.EvaluateProjection(m, "eos", tape.output.Z, dEOS, batch.Frames, true)
		if e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		if e := validateProjectionResult(r, batch.Frames, h, 1, tape.output.Z, dEOS); e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		eosInput = r.DInput
	}
	for row := 0; row < batch.Frames; row++ {
		z := tape.output.Z[row*h : (row+1)*h]
		var dEOSInput []float32
		if tape.fusedEOS || projection != nil {
			dEOSInput = eosInput[row*h : (row+1)*h]
		} else {
			dEOSInput = linearBackwardTraining(m.EOS, z, []float32{dEOS[row]}, &gradients.EOS)
		}
		for i := 0; i < h; i++ {
			dTransformed[(tape.prefix+row)*h+i] = dZ[row*h+i] + dEOSInput[i]
		}
	}
	var dSequence []float32
	if tape.backbone != nil {
		current, ok := m.SequenceBackend.(TrainingBackboneBackend)
		if !ok || !current.BackboneEnabled() || m.TransformerBackend == nil || m.AttentionBackwardBackend != nil {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("backbone changed after forward")
		}
		r, e := tape.backbone.EvaluateBackbone(m, batch, dTransformed, true)
		if e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		if e = validateBackboneResult(r, m, batch, dTransformed); e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		copy(inputGradients.VoiceLatents, r.DVoice)
		copy(inputGradients.NormalizedLatents, r.DNormalized)
		return gradients, inputGradients, nil
	}
	if tape.transformerBackend != nil {
		if m.TransformerBackend == nil {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("transformer backend removed after forward")
		}
		if m.AttentionBackwardBackend != nil {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("combined transformer/attention backends are not admitted")
		}
		// Re-evaluate the immutable transformer graph with the now-known loss
		// seed. Transfer-inclusive baseline; no CPU forward/backward fallback.
		r, err := evaluateTrainingTransformer(tape.transformerBackend, m.Transformer, tape.sequence, dTransformed, tape.rows, true)
		if err != nil {
			return nil, FlowLMTrainingInputGradients{}, err
		}
		_, resident := tape.transformerBackend.(TrainingTransformerAccumulator)
		if err := validateTransformerBackendResultMode(m.Transformer, r, tape.sequence, dTransformed, tape.rows, !resident); err != nil {
			return nil, FlowLMTrainingInputGradients{}, err
		}
		if !resident {
			g, err := transformerGradientsFromNamed(m.Transformer, r.Parameters)
			if err != nil {
				return nil, FlowLMTrainingInputGradients{}, err
			}
			gradients.Transformer = g
		}
		dSequence = r.DSequence
	} else {
		if m.TransformerBackend != nil {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("transformer backend changed after forward")
		}
		var err error
		dSequence, err = m.Transformer.backwardTrainingWithAttention(tape.rows, tape.transformer, dTransformed, gradients.Transformer, m.AttentionBackwardBackend)
		if err != nil {
			return nil, FlowLMTrainingInputGradients{}, err
		}
	}
	if tape.conditioner != nil {
		current, ok := m.SequenceBackend.(TrainingConditionerBackend)
		if !ok || !current.ConditionerEnabled() {
			return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("conditioner changed after forward")
		}
		result, e := tape.conditioner.EvaluateConditioner(m, batch, dSequence, true)
		if e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		if e = validateSequenceResult(result, tape.rows*h, batch.VoiceFrames*c, batch.Frames*c, batch.VoiceLatents, batch.NormalizedLatents, dSequence); e != nil {
			return nil, FlowLMTrainingInputGradients{}, e
		}
		copy(inputGradients.VoiceLatents, result.DVoice)
		copy(inputGradients.NormalizedLatents, result.DAudio)
	} else {
		voiceSeed, audioSeed := dSequence[h:(1+batch.VoiceFrames)*h], dSequence[tape.prefix*h:]
		if tape.voiceFusion != nil {
			current, ok := m.SequenceBackend.(TrainingVoiceSequenceBackend)
			if !ok || !current.VoiceSequenceEnabled() {
				return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("voice fusion changed after forward")
			}
			r, e := tape.voiceFusion.EvaluateVoiceSequence(m, batch, tape.sequence[tape.prefix*h:], dSequence, true)
			if e != nil {
				return nil, FlowLMTrainingInputGradients{}, e
			}
			if e = validateSequenceResult(r, tape.rows*h, batch.VoiceFrames*c, batch.Frames*h, batch.VoiceLatents, dSequence); e != nil {
				return nil, FlowLMTrainingInputGradients{}, e
			}
			copy(inputGradients.VoiceLatents, r.DVoice)
			audioSeed = r.DAudio
		} else if assembly != nil {
			r, e := assembly.EvaluateSequence(m, batch, tape.sequence[h:(1+batch.VoiceFrames)*h], tape.sequence[tape.prefix*h:], dSequence, true)
			if e != nil {
				return nil, FlowLMTrainingInputGradients{}, e
			}
			if e := validateSequenceResult(r, len(dSequence), len(voiceSeed), len(audioSeed), dSequence, tape.sequence); e != nil {
				return nil, FlowLMTrainingInputGradients{}, e
			}
			voiceSeed, audioSeed = r.DVoice, r.DAudio
		} else {
			copy(gradients.BOSBeforeVoice, dSequence[:h])
		}
		if tape.voiceFusion != nil {
			// Speaker VJP/parameter accumulation already executed in sequence tape.
		} else if projection != nil {
			if batch.VoiceFrames > 0 {
				seed := voiceSeed
				r, e := projection.EvaluateProjection(m, "speaker", batch.VoiceLatents, seed, batch.VoiceFrames, true)
				if e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
				if e := validateProjectionResult(r, batch.VoiceFrames, c, h, batch.VoiceLatents, seed); e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
				copy(inputGradients.VoiceLatents, r.DInput)
			} else {
				if _, e := projection.EvaluateProjection(m, "speaker-empty", nil, nil, 0, true); e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
			}
		} else {
			for row := 0; row < batch.VoiceFrames; row++ {
				dVoice := linearBackwardTraining(m.SpeakerProjection, batch.VoiceLatents[row*c:(row+1)*c], voiceSeed[row*h:(row+1)*h], &gradients.SpeakerProjection)
				copy(inputGradients.VoiceLatents[row*c:(row+1)*c], dVoice)
			}
		}
		if assembly == nil {
			for row, token := range batch.TextTokens {
				base := int(token) * h
				for i := 0; i < h; i++ {
					gradients.Embedding[base+i] += dSequence[(tape.textOffset+row)*h+i]
				}
			}
		}
		if tape.audioFusion != nil {
			current, ok := m.SequenceBackend.(TrainingAudioProjectionBackend)
			if !ok || !current.AudioProjectionEnabled() {
				return nil, FlowLMTrainingInputGradients{}, fmt.Errorf("fused audio backend changed after forward")
			}
			r, e := tape.audioFusion.EvaluateAudioProjection(m, batch, audioSeed, true)
			if e != nil {
				return nil, FlowLMTrainingInputGradients{}, e
			}
			if e = validateAudioProjectionBoundary(r, batch.Frames, h, c, batch.NormalizedLatents, audioSeed); e != nil {
				return nil, FlowLMTrainingInputGradients{}, e
			}
			copy(inputGradients.NormalizedLatents, r.DNormalized)
		} else {
			var audioInputGradients []float32
			if projection != nil {
				seed := audioSeed
				r, e := projection.EvaluateProjection(m, "input", tape.audioInput, seed, batch.Frames, true)
				if e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
				if e := validateProjectionResult(r, batch.Frames, c, h, tape.audioInput, seed); e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
				audioInputGradients = r.DInput
			}
			if assembly == nil {
				for row := 0; row < batch.Frames; row++ {
					var dAudio []float32
					if projection != nil {
						dAudio = audioInputGradients[row*c : (row+1)*c]
					} else {
						dAudio = linearBackwardTraining(m.Input, tape.audioInput[row*c:(row+1)*c], audioSeed[row*h:(row+1)*h], &gradients.Input)
					}
					if row == 0 {
						addInPlace(gradients.BOS, dAudio)
					} else {
						copy(inputGradients.NormalizedLatents[(row-1)*c:row*c], dAudio)
					}
				}
			} else {
				r, e := assembly.EvaluateAudioAssembly(m, batch, audioInputGradients, true)
				if e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
				if e := validateAudioAssemblyResult(r, len(audioInputGradients), audioInputGradients, batch.NormalizedLatents); e != nil {
					return nil, FlowLMTrainingInputGradients{}, e
				}
				copy(inputGradients.NormalizedLatents, r.DNormalized)
			}
		}
	}
	return gradients, inputGradients, nil
}

func (m *FlowLMTrainingCPU) validate(batch FlowLMTrainingBatch, dZ, dEOS []float32) (rows, prefix int, err error) {
	return m.validateMode(batch, dZ, dEOS, false)
}

func (m *FlowLMTrainingCPU) validateMode(batch FlowLMTrainingBatch, dZ, dEOS []float32, allowEmptyText bool) (rows, prefix int, err error) {
	if m == nil || m.Hidden <= 0 || m.LatentDim <= 0 || m.Vocabulary <= 0 || batch.Frames <= 0 || batch.VoiceFrames < 0 || (!allowEmptyText && len(batch.TextTokens) == 0) || m.Transformer == nil {
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
