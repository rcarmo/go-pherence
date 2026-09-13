package omnivoice

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

// CodecEncoder is the native non-HuBERT HiggsAudioV2 encoder path: DAC
// acoustic encoder, semantic encoder, fusion projection and RVQ code search.
// Prepare/EncodeFeaturesInto keep a bounded per-instance workspace so repeated
// calls can reuse activations and convolution packing buffers without changing
// the allocating convenience API. Instances are single-caller.
type CodecEncoder struct {
	ops                   map[string]codecOperator
	weights               *loader.CodecWeights
	pad                   int
	semanticStem          codecOperator
	semanticBlocks        [2]codecEncoderSemanticBlock
	acousticStem          codecOperator
	acousticBlocks        [5]codecEncoderAcousticBlock
	acousticTailSnake     []float32
	acousticTail          codecOperator
	fusion                codecOperator
	quantizers            [8]codecEncoderQuantizer
	preparedWaveSamples   int
	preparedSemanticFrame int
	preparedAlignedFrames int
	preparedSignalCap     int
	preparedPackedCap     int
	preparedResultCap     int
	scratch               *codecEncoderScratch
}

type codecEncoderSemanticResidual struct {
	conv1 codecOperator
	conv2 codecOperator
}

type codecEncoderSemanticBlock struct {
	conv      codecOperator
	residuals [2]codecEncoderSemanticResidual
}

type codecEncoderAcousticResidual struct {
	snake1   []float32
	snake2   []float32
	conv1    codecOperator
	conv2    codecOperator
	dilation int
}

type codecEncoderAcousticBlock struct {
	snake1    []float32
	conv1     codecOperator
	stride    int
	residuals [3]codecEncoderAcousticResidual
}

type codecEncoderQuantizer struct {
	projectIn  codecOperator
	projectOut codecOperator
	codebook   []float32
	shape      []int
	norms      []float32
}

type codecEncoderScratch struct {
	slots  [8][]float32
	used   [8]bool
	packed []float32
	result []float32
}

func NewCodecEncoder(w *loader.CodecWeights) (*CodecEncoder, error) {
	if w == nil || len(w.Tensors["fc.weight"]) == 0 || len(w.Tensors["acoustic_encoder.conv2.weight"]) == 0 || len(w.Tensors["encoder_semantic.conv.weight"]) == 0 {
		return nil, fmt.Errorf("omnivoice: missing codec encoder tensors")
	}
	if err := validateCodecEncoderWeights(w); err != nil {
		return nil, err
	}
	e := &CodecEncoder{weights: w, ops: map[string]codecOperator{}, pad: 480}
	for name, shape := range w.Shapes {
		if len(shape) >= 2 && len(shape) <= 3 {
			if weight := w.Tensors[name]; len(weight) > 0 && nameHasWeight(name) {
				base := name[:len(name)-7]
				e.ops[base] = codecOperator{weight: weight, bias: w.Tensors[base+".bias"], shape: shape}
			}
		}
	}
	opFor := func(name string) (codecOperator, error) {
		op, ok := e.ops[name]
		if !ok {
			return codecOperator{}, fmt.Errorf("omnivoice: missing encoder op %s", name)
		}
		return op, nil
	}
	var err error
	e.semanticStem, err = opFor("encoder_semantic.conv")
	if err != nil {
		return nil, err
	}
	for i := range e.semanticBlocks {
		prefix := fmt.Sprintf("encoder_semantic.conv_blocks.%d.", i)
		block := &e.semanticBlocks[i]
		if block.conv, err = opFor(prefix + "conv"); err != nil {
			return nil, err
		}
		for j := range block.residuals {
			residualPrefix := fmt.Sprintf("%sres_units.%d.", prefix, j)
			residual := &block.residuals[j]
			if residual.conv1, err = opFor(residualPrefix + "conv1"); err != nil {
				return nil, err
			}
			if residual.conv2, err = opFor(residualPrefix + "conv2"); err != nil {
				return nil, err
			}
		}
	}
	e.acousticStem, err = opFor("acoustic_encoder.conv1")
	if err != nil {
		return nil, err
	}
	for i := range e.acousticBlocks {
		prefix := fmt.Sprintf("acoustic_encoder.block.%d.", i)
		block := &e.acousticBlocks[i]
		block.snake1 = w.Tensors[prefix+"snake1.alpha"]
		block.stride = e.weights.Rates[i]
		if block.conv1, err = opFor(prefix + "conv1"); err != nil {
			return nil, err
		}
		for j, dilation := range [...]int{1, 3, 9} {
			residualPrefix := fmt.Sprintf("%sres_unit%d.", prefix, j+1)
			residual := &block.residuals[j]
			residual.snake1 = w.Tensors[residualPrefix+"snake1.alpha"]
			residual.snake2 = w.Tensors[residualPrefix+"snake2.alpha"]
			residual.dilation = dilation
			if residual.conv1, err = opFor(residualPrefix + "conv1"); err != nil {
				return nil, err
			}
			if residual.conv2, err = opFor(residualPrefix + "conv2"); err != nil {
				return nil, err
			}
		}
	}
	e.acousticTailSnake = w.Tensors["acoustic_encoder.snake1.alpha"]
	e.acousticTail, err = opFor("acoustic_encoder.conv2")
	if err != nil {
		return nil, err
	}
	e.fusion, err = opFor("fc")
	if err != nil {
		return nil, err
	}
	for i := range e.quantizers {
		prefix := fmt.Sprintf("quantizer.quantizers.%d.", i)
		quantizer := &e.quantizers[i]
		if quantizer.projectIn, err = opFor(prefix + "project_in"); err != nil {
			return nil, err
		}
		if quantizer.projectOut, err = opFor(prefix + "project_out"); err != nil {
			return nil, err
		}
		name := prefix + "codebook.embed"
		quantizer.codebook = w.Tensors[name]
		quantizer.shape = w.Shapes[name]
		norms := make([]float32, quantizer.shape[0])
		for code := 0; code < quantizer.shape[0]; code++ {
			base := code * quantizer.shape[1]
			sum := float32(0)
			for j := 0; j < quantizer.shape[1]; j++ {
				v := quantizer.codebook[base+j]
				sum += v * v
			}
			norms[code] = sum
		}
		quantizer.norms = norms
	}
	return e, nil
}

func nameHasWeight(name string) bool {
	return len(name) > len(".weight") && name[len(name)-7:] == ".weight"
}

func (e *CodecEncoder) Prepare(waveSamples, semanticFrames int) error {
	if e == nil || e.weights == nil {
		return fmt.Errorf("omnivoice: nil codec encoder")
	}
	if waveSamples < 1 || waveSamples > 480000 || semanticFrames < 1 || semanticFrames > 500 {
		return fmt.Errorf("omnivoice: invalid codec encoder input")
	}
	alignedFrames, err := e.alignedWaveLengthFor(waveSamples, semanticFrames)
	if err != nil {
		return err
	}
	if e.scratch != nil && waveSamples == e.preparedWaveSamples && semanticFrames == e.preparedSemanticFrame && alignedFrames == e.preparedAlignedFrames {
		return nil
	}
	signalCap, packedCap, resultCap, err := e.workspaceCaps(alignedFrames, semanticFrames)
	if err != nil {
		return err
	}
	if e.scratch != nil && signalCap <= len(e.scratch.slots[0]) && packedCap <= len(e.scratch.packed) && resultCap <= len(e.scratch.result) {
		e.preparedWaveSamples = waveSamples
		e.preparedSemanticFrame = semanticFrames
		e.preparedAlignedFrames = alignedFrames
		e.preparedSignalCap = signalCap
		e.preparedPackedCap = packedCap
		e.preparedResultCap = resultCap
		return nil
	}
	s := &codecEncoderScratch{packed: make([]float32, packedCap), result: make([]float32, resultCap)}
	for i := range s.slots {
		s.slots[i] = make([]float32, signalCap)
	}
	e.scratch = s
	e.preparedWaveSamples = waveSamples
	e.preparedSemanticFrame = semanticFrames
	e.preparedAlignedFrames = alignedFrames
	e.preparedSignalCap = signalCap
	e.preparedPackedCap = packedCap
	e.preparedResultCap = resultCap
	return nil
}

func (e *CodecEncoder) EncodeFeatures(ctx context.Context, wave []float32, semantic []float32, semanticFrames int) ([]int, error) {
	if e == nil || e.weights == nil || ctx == nil || semanticFrames < 1 || semanticFrames > 500 || len(semantic) != semanticFrames*768 {
		return nil, fmt.Errorf("omnivoice: invalid encoder input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := e.Prepare(len(wave), semanticFrames); err != nil {
		return nil, err
	}
	codes := make([]int, e.weights.Quantizers*semanticFrames)
	if err := e.EncodeFeaturesInto(ctx, codes, wave, semantic, semanticFrames); err != nil {
		return nil, err
	}
	return codes, nil
}

func (e *CodecEncoder) EncodeFeaturesInto(ctx context.Context, dst []int, wave []float32, semantic []float32, semanticFrames int) error {
	if e == nil || e.weights == nil || ctx == nil || semanticFrames < 1 || semanticFrames > 500 || len(wave) < 1 || len(wave) > 480000 || len(semantic) != semanticFrames*768 || len(dst) != e.weights.Quantizers*semanticFrames {
		return fmt.Errorf("omnivoice: invalid codec encoder input")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := e.Prepare(len(wave), semanticFrames); err != nil {
		return err
	}
	e.scratch.used = [8]bool{}

	semanticInput := signal{data: e.buffer(768 * semanticFrames), channels: 768, frames: semanticFrames}
	for t := 0; t < semanticFrames; t++ {
		row := semantic[t*768 : (t+1)*768]
		for c, v := range row {
			semanticInput.data[c*semanticFrames+t] = v
		}
	}
	eSemantic, err := e.encodeSemantic(ctx, semanticInput)
	e.release(semanticInput)
	if err != nil {
		e.scratch.used = [8]bool{}
		return err
	}
	if eSemantic.frames != semanticFrames {
		e.release(eSemantic)
		e.scratch.used = [8]bool{}
		return fmt.Errorf("omnivoice: semantic encoder length mismatch")
	}
	waveInput, err := e.alignWaveLength(wave, semanticFrames)
	if err != nil {
		e.release(eSemantic)
		e.scratch.used = [8]bool{}
		return err
	}
	eAcoustic, err := e.encodeAcoustic(ctx, waveInput)
	e.release(waveInput)
	if err != nil {
		e.release(eSemantic)
		e.scratch.used = [8]bool{}
		return err
	}
	if eAcoustic.frames != eSemantic.frames {
		e.release(eAcoustic)
		e.release(eSemantic)
		e.scratch.used = [8]bool{}
		return fmt.Errorf("omnivoice: acoustic/semantic frame mismatch")
	}
	embeddings := signal{data: e.buffer((eAcoustic.channels + eSemantic.channels) * semanticFrames), channels: eAcoustic.channels + eSemantic.channels, frames: semanticFrames}
	copy(embeddings.data[:len(eAcoustic.data)], eAcoustic.data)
	copy(embeddings.data[len(eAcoustic.data):], eSemantic.data)
	e.release(eAcoustic)
	e.release(eSemantic)
	prev := embeddings
	embeddings, err = e.conv(embeddings, e.fusion, 1, 0, 1)
	e.release(prev)
	if err != nil {
		e.scratch.used = [8]bool{}
		return err
	}
	if err := e.quantizeInto(ctx, dst, embeddings); err != nil {
		e.release(embeddings)
		e.scratch.used = [8]bool{}
		return err
	}
	e.release(embeddings)
	e.scratch.used = [8]bool{}
	return nil
}

func (e *CodecEncoder) buffer(n int) []float32 {
	if e.scratch == nil {
		return make([]float32, n)
	}
	for i := range e.scratch.slots {
		if !e.scratch.used[i] && n <= len(e.scratch.slots[i]) {
			e.scratch.used[i] = true
			out := e.scratch.slots[i][:n]
			clear(out)
			return out
		}
	}
	panic("omnivoice: internal codec encoder workspace exhausted")
}

func (e *CodecEncoder) release(x signal) {
	if e.scratch == nil || len(x.data) == 0 {
		return
	}
	for i := range e.scratch.slots {
		if len(e.scratch.slots[i]) >= len(x.data) && &e.scratch.slots[i][0] == &x.data[0] {
			e.scratch.used[i] = false
			return
		}
	}
}

func (e *CodecEncoder) workspaceCaps(alignedWaveFrames, semanticFrames int) (signalCap, packedCap, resultCap int, err error) {
	const (
		convTile      = 64
		packedPanelNR = 16
	)
	signalCap = max(768*semanticFrames, alignedWaveFrames)
	trackConv := func(op codecOperator, in signal, stride, padding, dilation int) (signal, error) {
		shape := op.shape
		if len(shape) < 2 || len(shape) > 3 || shape[1] != in.channels {
			return signal{}, fmt.Errorf("omnivoice: invalid encoder conv")
		}
		kernel := 1
		if len(shape) == 3 {
			kernel = shape[2]
		}
		length := convOutputLength(in.frames, kernel, stride, padding, dilation)
		if length <= 0 {
			return signal{}, fmt.Errorf("omnivoice: short encoder conv input")
		}
		out := signal{channels: shape[0], frames: length}
		k := in.channels * kernel
		signalCap = max(signalCap, out.channels*out.frames)
		packedCap = max(packedCap, k*convTile)
		resultCap = max(resultCap, k*packedPanelNR)
		return out, nil
	}

	semantic := signal{channels: 768, frames: semanticFrames}
	semantic, err = trackConv(e.semanticStem, semantic, 1, 1, 1)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := range e.semanticBlocks {
		block := &e.semanticBlocks[i]
		for j := range block.residuals {
			residual := &block.residuals[j]
			semanticResidual, convErr := trackConv(residual.conv1, semantic, 1, 1, 1)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			semanticResidual, convErr = trackConv(residual.conv2, semanticResidual, 1, 0, 1)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			signalCap = max(signalCap, semanticResidual.channels*semanticResidual.frames)
		}
		semantic, err = trackConv(block.conv, semantic, 1, 1, 1)
		if err != nil {
			return 0, 0, 0, err
		}
	}

	acoustic := signal{channels: 1, frames: alignedWaveFrames}
	acoustic, err = trackConv(e.acousticStem, acoustic, 1, 3, 1)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := range e.acousticBlocks {
		block := &e.acousticBlocks[i]
		for j := range block.residuals {
			residual := &block.residuals[j]
			acousticResidual, convErr := trackConv(residual.conv1, acoustic, 1, 3*residual.dilation, residual.dilation)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			acousticResidual, convErr = trackConv(residual.conv2, acousticResidual, 1, 0, 1)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			signalCap = max(signalCap, acousticResidual.channels*acousticResidual.frames)
		}
		acoustic, err = trackConv(block.conv1, acoustic, block.stride, (block.stride+1)/2, 1)
		if err != nil {
			return 0, 0, 0, err
		}
	}
	acoustic, err = trackConv(e.acousticTail, acoustic, 1, 1, 1)
	if err != nil {
		return 0, 0, 0, err
	}

	embeddings := signal{channels: acoustic.channels + semantic.channels, frames: semanticFrames}
	signalCap = max(signalCap, embeddings.channels*embeddings.frames)
	embeddings, err = trackConv(e.fusion, embeddings, 1, 0, 1)
	if err != nil {
		return 0, 0, 0, err
	}
	for book := 0; book < e.weights.Quantizers; book++ {
		quantizer := &e.quantizers[book]
		projected, convErr := trackConv(quantizer.projectIn, embeddings, 1, 0, 1)
		if convErr != nil {
			return 0, 0, 0, convErr
		}
		if len(quantizer.shape) != 2 || quantizer.shape[1] != projected.channels {
			return 0, 0, 0, fmt.Errorf("omnivoice: invalid encoder codebook")
		}
		signalCap = max(signalCap, quantizer.shape[1]*semanticFrames)
		decoded, convErr := trackConv(quantizer.projectOut, signal{channels: quantizer.shape[1], frames: semanticFrames}, 1, 0, 1)
		if convErr != nil {
			return 0, 0, 0, convErr
		}
		signalCap = max(signalCap, decoded.channels*decoded.frames)
	}
	return signalCap, packedCap, resultCap, nil
}

func (e *CodecEncoder) alignedWaveLengthFor(waveSamples, semanticFrames int) (int, error) {
	if got := e.acousticOutputFrames(waveSamples); got == semanticFrames {
		return waveSamples, nil
	}
	paddedSamples := waveSamples + 2*e.pad
	paddedFrames := e.acousticOutputFrames(paddedSamples)
	if paddedFrames != semanticFrames {
		return 0, fmt.Errorf("omnivoice: semantic frames %d do not match acoustic encoder lengths %d or %d", semanticFrames, e.acousticOutputFrames(waveSamples), paddedFrames)
	}
	return paddedSamples, nil
}

func (e *CodecEncoder) alignWaveLength(wave []float32, semanticFrames int) (signal, error) {
	alignedFrames, err := e.alignedWaveLengthFor(len(wave), semanticFrames)
	if err != nil {
		return signal{}, err
	}
	if alignedFrames == len(wave) {
		return signal{data: wave, channels: 1, frames: len(wave)}, nil
	}
	padded := e.buffer(alignedFrames)
	copy(padded[e.pad:], wave)
	return signal{data: padded, channels: 1, frames: len(padded)}, nil
}

func (e *CodecEncoder) acousticOutputFrames(samples int) int {
	length := convOutputLength(samples, 7, 1, 3, 1)
	for _, stride := range e.weights.Rates {
		length = convOutputLength(length, 2*stride, stride, (stride+1)/2, 1)
	}
	return convOutputLength(length, 3, 1, 1, 1)
}

func convOutputLength(length, kernel, stride, padding, dilation int) int {
	return (length+2*padding-dilation*(kernel-1)-1)/stride + 1
}

func (e *CodecEncoder) encodeSemantic(ctx context.Context, x signal) (signal, error) {
	if err := ctx.Err(); err != nil {
		return signal{}, err
	}
	out, err := e.conv(x, e.semanticStem, 1, 1, 1)
	if err != nil {
		return signal{}, err
	}
	for i := range e.semanticBlocks {
		block := &e.semanticBlocks[i]
		for j := range block.residuals {
			if err = ctx.Err(); err != nil {
				e.release(out)
				return signal{}, err
			}
			prev := out
			out, err = e.semanticResidual(out, &block.residuals[j])
			e.release(prev)
			if err != nil {
				return signal{}, err
			}
		}
		prev := out
		out, err = e.conv(out, block.conv, 1, 1, 1)
		e.release(prev)
		if err != nil {
			return signal{}, err
		}
	}
	return out, nil
}

func (e *CodecEncoder) semanticResidual(x signal, residual *codecEncoderSemanticResidual) (signal, error) {
	work := signal{data: e.buffer(len(x.data)), channels: x.channels, frames: x.frames}
	copy(work.data, x.data)
	e.elu(work)
	out, err := e.conv(work, residual.conv1, 1, 1, 1)
	e.release(work)
	if err != nil {
		return signal{}, err
	}
	e.elu(out)
	prev := out
	out, err = e.conv(out, residual.conv2, 1, 0, 1)
	e.release(prev)
	if err != nil {
		return signal{}, err
	}
	if len(out.data) != len(x.data) {
		e.release(out)
		return signal{}, fmt.Errorf("omnivoice: semantic residual dimensions differ")
	}
	simd.VecAdd(out.data, out.data, x.data)
	return out, nil
}

func (e *CodecEncoder) elu(x signal) {
	for i, v := range x.data {
		if v < 0 {
			x.data[i] = float32(math.Exp(float64(v)) - 1)
		}
	}
}

func (e *CodecEncoder) encodeAcoustic(ctx context.Context, x signal) (signal, error) {
	if err := ctx.Err(); err != nil {
		return signal{}, err
	}
	out, err := e.conv(x, e.acousticStem, 1, 3, 1)
	if err != nil {
		return signal{}, err
	}
	for i := range e.acousticBlocks {
		block := &e.acousticBlocks[i]
		for j := range block.residuals {
			if err = ctx.Err(); err != nil {
				e.release(out)
				return signal{}, err
			}
			prev := out
			out, err = e.acousticResidual(out, &block.residuals[j])
			e.release(prev)
			if err != nil {
				return signal{}, err
			}
		}
		if err = e.snake(out, block.snake1); err != nil {
			e.release(out)
			return signal{}, err
		}
		prev := out
		out, err = e.conv(out, block.conv1, block.stride, (block.stride+1)/2, 1)
		e.release(prev)
		if err != nil {
			return signal{}, err
		}
	}
	if err = e.snake(out, e.acousticTailSnake); err != nil {
		e.release(out)
		return signal{}, err
	}
	prev := out
	out, err = e.conv(out, e.acousticTail, 1, 1, 1)
	e.release(prev)
	if err != nil {
		return signal{}, err
	}
	return out, nil
}

func (e *CodecEncoder) acousticResidual(x signal, residual *codecEncoderAcousticResidual) (signal, error) {
	work := signal{data: e.buffer(len(x.data)), channels: x.channels, frames: x.frames}
	copy(work.data, x.data)
	if err := e.snake(work, residual.snake1); err != nil {
		e.release(work)
		return signal{}, err
	}
	out, err := e.conv(work, residual.conv1, 1, 3*residual.dilation, residual.dilation)
	e.release(work)
	if err != nil {
		return signal{}, err
	}
	if err = e.snake(out, residual.snake2); err != nil {
		e.release(out)
		return signal{}, err
	}
	prev := out
	out, err = e.conv(out, residual.conv2, 1, 0, 1)
	e.release(prev)
	if err != nil {
		return signal{}, err
	}
	if len(out.data) != len(x.data) {
		e.release(out)
		return signal{}, fmt.Errorf("omnivoice: acoustic residual dimensions differ")
	}
	simd.VecAdd(out.data, out.data, x.data)
	return out, nil
}

func (e *CodecEncoder) snake(x signal, alpha []float32) error {
	if len(alpha) != x.channels {
		return fmt.Errorf("omnivoice: invalid snake")
	}
	for c, a := range alpha {
		for t := 0; t < x.frames; t++ {
			i := c*x.frames + t
			phase := a * x.data[i]
			s := float32(math.Sin(float64(phase)))
			x.data[i] += (float32(1) / (a + 1e-9)) * (s * s)
		}
	}
	return nil
}

func (e *CodecEncoder) conv(x signal, op codecOperator, stride, padding, dilation int) (signal, error) {
	weight, shape, bias := op.weight, op.shape, op.bias
	if len(shape) < 2 || len(shape) > 3 || shape[1] != x.channels {
		return signal{}, fmt.Errorf("omnivoice: invalid encoder conv")
	}
	outChannels, kernel := shape[0], 1
	if len(shape) == 3 {
		kernel = shape[2]
	}
	if len(bias) > 0 && len(bias) != outChannels {
		return signal{}, fmt.Errorf("omnivoice: invalid encoder bias")
	}
	length := convOutputLength(x.frames, kernel, stride, padding, dilation)
	if length <= 0 {
		return signal{}, fmt.Errorf("omnivoice: short encoder conv input")
	}
	y := signal{data: e.buffer(outChannels * length), channels: outChannels, frames: length}
	const (
		tile          = 64
		packedPanelNR = 16
	)
	k := x.channels * kernel
	scratchNeed := k * packedPanelNR
	var packed, scratch []float32
	if e.scratch != nil && k*tile <= len(e.scratch.packed) && scratchNeed <= len(e.scratch.result) {
		packed = e.scratch.packed[:k*tile]
		scratch = e.scratch.result[:scratchNeed]
	} else {
		packed = make([]float32, k*tile)
		scratch = make([]float32, scratchNeed)
	}
	for start := 0; start < length; start += tile {
		n := min(tile, length-start)
		p := packed[:k*n]
		clear(p)
		for t := 0; t < n; t++ {
			row := p[t*k : (t+1)*k]
			sourceBase := (start+t)*stride - padding
			fan := 0
			for c := 0; c < x.channels; c++ {
				channelBase := c * x.frames
				for j := 0; j < kernel; j++ {
					source := sourceBase + j*dilation
					if source >= 0 && source < x.frames {
						row[fan] = x.data[channelBase+source]
					}
					fan++
				}
			}
		}
		if !simd.SgemmNTPackedTo(y.data[start:], weight, p, scratch, outChannels, n, k, 1, k, k, length) {
			e.release(y)
			return signal{}, fmt.Errorf("omnivoice: encoder conv GEMM shape")
		}
		if len(bias) == 0 {
			continue
		}
		for c, b := range bias {
			row := y.data[c*length+start : c*length+start+n]
			for t := range row {
				row[t] += b
			}
		}
	}
	return y, nil
}

func (e *CodecEncoder) quantizeInto(ctx context.Context, dst []int, embeddings signal) error {
	frames := embeddings.frames
	if len(dst) != e.weights.Quantizers*frames {
		return fmt.Errorf("omnivoice: invalid codec encoder output")
	}
	residual := embeddings
	for book := 0; book < e.weights.Quantizers; book++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		quantizer := &e.quantizers[book]
		projected, err := e.conv(residual, quantizer.projectIn, 1, 0, 1)
		if err != nil {
			return err
		}
		codes := dst[book*frames : (book+1)*frames]
		if err := e.nearestCodesInto(ctx, quantizer, projected, codes); err != nil {
			e.release(projected)
			return err
		}
		quantized, err := e.lookupCodebook(quantizer, codes)
		if err != nil {
			e.release(projected)
			return err
		}
		decoded, err := e.conv(quantized, quantizer.projectOut, 1, 0, 1)
		e.release(quantized)
		e.release(projected)
		if err != nil {
			return err
		}
		simd.VecScaleAdd(residual.data, residual.data, decoded.data, -1)
		e.release(decoded)
	}
	return nil
}

func (e *CodecEncoder) nearestCodesInto(ctx context.Context, quantizer *codecEncoderQuantizer, projected signal, dst []int) error {
	embed := quantizer.codebook
	shape := quantizer.shape
	if len(shape) != 2 || shape[1] != projected.channels || len(dst) != projected.frames {
		return fmt.Errorf("omnivoice: invalid encoder codebook")
	}
	norms := quantizer.norms
	for t := 0; t < projected.frames; t++ {
		if t&31 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		bestIndex := 0
		bestScore := float32(-math.MaxFloat32)
		for code := 0; code < shape[0]; code++ {
			base := code * shape[1]
			dot := float32(0)
			for c := 0; c < shape[1]; c++ {
				dot += projected.data[c*projected.frames+t] * embed[base+c]
			}
			score := 2*dot - norms[code]
			if code == 0 || score > bestScore {
				bestScore = score
				bestIndex = code
			}
		}
		dst[t] = bestIndex
	}
	return nil
}

func (e *CodecEncoder) lookupCodebook(quantizer *codecEncoderQuantizer, codes []int) (signal, error) {
	embed := quantizer.codebook
	shape := quantizer.shape
	if len(shape) != 2 {
		return signal{}, fmt.Errorf("omnivoice: invalid encoder codebook")
	}
	out := signal{data: e.buffer(shape[1] * len(codes)), channels: shape[1], frames: len(codes)}
	for t, code := range codes {
		base := code * shape[1]
		for c := 0; c < shape[1]; c++ {
			out.data[c*out.frames+t] = embed[base+c]
		}
	}
	return out, nil
}
