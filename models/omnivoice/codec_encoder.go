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
	ops               map[string]codecOperator
	weights           *loader.CodecWeights
	pad               int
	codebookNorms     [8][]float32
	quantizerNames    [8]string
	acousticBlocks    [5]string
	acousticResiduals [5][3]string
	semanticBlocks    [2]string
	semanticResiduals [2][2]string
	scratch           *codecEncoderScratch
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
	for i := range e.quantizerNames {
		e.quantizerNames[i] = fmt.Sprintf("quantizer.quantizers.%d.", i)
		name := e.quantizerNames[i] + "codebook.embed"
		shape := w.Shapes[name]
		norms := make([]float32, shape[0])
		embed := w.Tensors[name]
		for code := 0; code < shape[0]; code++ {
			base := code * shape[1]
			sum := float32(0)
			for j := 0; j < shape[1]; j++ {
				v := embed[base+j]
				sum += v * v
			}
			norms[code] = sum
		}
		e.codebookNorms[i] = norms
	}
	for i := range e.acousticBlocks {
		e.acousticBlocks[i] = fmt.Sprintf("acoustic_encoder.block.%d.", i)
		for j := range e.acousticResiduals[i] {
			e.acousticResiduals[i][j] = fmt.Sprintf("%sres_unit%d.", e.acousticBlocks[i], j+1)
		}
	}
	for i := range e.semanticBlocks {
		e.semanticBlocks[i] = fmt.Sprintf("encoder_semantic.conv_blocks.%d.", i)
		for j := range e.semanticResiduals[i] {
			e.semanticResiduals[i][j] = fmt.Sprintf("%sres_units.%d.", e.semanticBlocks[i], j)
		}
	}
	for name, shape := range w.Shapes {
		if len(shape) >= 2 && len(shape) <= 3 {
			if weight := w.Tensors[name]; len(weight) > 0 && nameHasWeight(name) {
				base := name[:len(name)-7]
				e.ops[base] = codecOperator{weight: weight, bias: w.Tensors[base+".bias"], shape: shape}
			}
		}
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
	if waveSamples < 1 || semanticFrames < 1 {
		return fmt.Errorf("omnivoice: invalid codec encoder input")
	}
	alignedFrames, err := e.alignedWaveLengthFor(waveSamples, semanticFrames)
	if err != nil {
		return err
	}
	signalCap, packedCap, resultCap, err := e.workspaceCaps(alignedFrames, semanticFrames)
	if err != nil {
		return err
	}
	if e.scratch != nil && signalCap <= len(e.scratch.slots[0]) && packedCap <= len(e.scratch.packed) && resultCap <= len(e.scratch.result) {
		return nil
	}
	s := &codecEncoderScratch{packed: make([]float32, packedCap), result: make([]float32, resultCap)}
	for i := range s.slots {
		s.slots[i] = make([]float32, signalCap)
	}
	e.scratch = s
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
	if ctx == nil || semanticFrames < 1 || len(wave) < 1 || len(semantic) != semanticFrames*768 || len(dst) != e.weights.Quantizers*semanticFrames {
		return fmt.Errorf("omnivoice: invalid codec encoder input")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := e.Prepare(len(wave), semanticFrames); err != nil {
		return err
	}
	e.scratch.used = [8]bool{}
	defer func() { e.scratch.used = [8]bool{} }()

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
		return err
	}
	if eSemantic.frames != semanticFrames {
		return fmt.Errorf("omnivoice: semantic encoder length mismatch")
	}
	waveInput, err := e.alignWaveLength(wave, semanticFrames)
	if err != nil {
		e.release(eSemantic)
		return err
	}
	eAcoustic, err := e.encodeAcoustic(ctx, waveInput)
	e.release(waveInput)
	if err != nil {
		e.release(eSemantic)
		return err
	}
	if eAcoustic.frames != eSemantic.frames {
		e.release(eAcoustic)
		e.release(eSemantic)
		return fmt.Errorf("omnivoice: acoustic/semantic frame mismatch")
	}
	embeddings := signal{data: e.buffer((eAcoustic.channels + eSemantic.channels) * semanticFrames), channels: eAcoustic.channels + eSemantic.channels, frames: semanticFrames}
	copy(embeddings.data[:len(eAcoustic.data)], eAcoustic.data)
	copy(embeddings.data[len(eAcoustic.data):], eSemantic.data)
	e.release(eAcoustic)
	e.release(eSemantic)
	prev := embeddings
	embeddings, err = e.conv(embeddings, "fc", 1, 0, 1)
	e.release(prev)
	if err != nil {
		return err
	}
	if err := e.quantizeInto(ctx, dst, embeddings); err != nil {
		e.release(embeddings)
		return err
	}
	e.release(embeddings)
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
	signalCap = max(768*semanticFrames, alignedWaveFrames)
	trackConv := func(name string, in signal, stride, padding, dilation int) (signal, error) {
		op, ok := e.ops[name]
		if !ok {
			return signal{}, fmt.Errorf("omnivoice: missing encoder op %s", name)
		}
		shape := op.shape
		if len(shape) < 2 || len(shape) > 3 || shape[1] != in.channels {
			return signal{}, fmt.Errorf("omnivoice: invalid encoder conv %s", name)
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
		signalCap = max(signalCap, out.channels*out.frames)
		packedCap = max(packedCap, in.channels*kernel*64)
		resultCap = max(resultCap, out.channels*64)
		return out, nil
	}

	semantic := signal{channels: 768, frames: semanticFrames}
	semantic, err = trackConv("encoder_semantic.conv", semantic, 1, 1, 1)
	if err != nil {
		return 0, 0, 0, err
	}
	for i := range e.semanticBlocks {
		for j := range e.semanticResiduals[i] {
			semanticResidual, convErr := trackConv(e.semanticResiduals[i][j]+"conv1", semantic, 1, 1, 1)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			semanticResidual, convErr = trackConv(e.semanticResiduals[i][j]+"conv2", semanticResidual, 1, 0, 1)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			signalCap = max(signalCap, semanticResidual.channels*semanticResidual.frames)
		}
		semantic, err = trackConv(e.semanticBlocks[i]+"conv", semantic, 1, 1, 1)
		if err != nil {
			return 0, 0, 0, err
		}
	}

	acoustic := signal{channels: 1, frames: alignedWaveFrames}
	acoustic, err = trackConv("acoustic_encoder.conv1", acoustic, 1, 3, 1)
	if err != nil {
		return 0, 0, 0, err
	}
	for i, stride := range e.weights.Rates {
		for j, dilation := range []int{1, 3, 9} {
			acousticResidual, convErr := trackConv(e.acousticResiduals[i][j]+"conv1", acoustic, 1, 3*dilation, dilation)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			acousticResidual, convErr = trackConv(e.acousticResiduals[i][j]+"conv2", acousticResidual, 1, 0, 1)
			if convErr != nil {
				return 0, 0, 0, convErr
			}
			signalCap = max(signalCap, acousticResidual.channels*acousticResidual.frames)
		}
		acoustic, err = trackConv(e.acousticBlocks[i]+"conv1", acoustic, stride, (stride+1)/2, 1)
		if err != nil {
			return 0, 0, 0, err
		}
	}
	acoustic, err = trackConv("acoustic_encoder.conv2", acoustic, 1, 1, 1)
	if err != nil {
		return 0, 0, 0, err
	}

	embeddings := signal{channels: acoustic.channels + semantic.channels, frames: semanticFrames}
	signalCap = max(signalCap, embeddings.channels*embeddings.frames)
	embeddings, err = trackConv("fc", embeddings, 1, 0, 1)
	if err != nil {
		return 0, 0, 0, err
	}
	for book := 0; book < e.weights.Quantizers; book++ {
		prefix := e.quantizerNames[book]
		projected, convErr := trackConv(prefix+"project_in", embeddings, 1, 0, 1)
		if convErr != nil {
			return 0, 0, 0, convErr
		}
		shape := e.weights.Shapes[prefix+"codebook.embed"]
		if len(shape) != 2 || shape[1] != projected.channels {
			return 0, 0, 0, fmt.Errorf("omnivoice: invalid encoder codebook")
		}
		signalCap = max(signalCap, shape[1]*semanticFrames)
		decoded, convErr := trackConv(prefix+"project_out", signal{channels: shape[1], frames: semanticFrames}, 1, 0, 1)
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
	out, err := e.conv(x, "encoder_semantic.conv", 1, 1, 1)
	if err != nil {
		return signal{}, err
	}
	for i := range e.semanticBlocks {
		for j := range e.semanticResiduals[i] {
			if err = ctx.Err(); err != nil {
				e.release(out)
				return signal{}, err
			}
			prev := out
			out, err = e.semanticResidual(out, e.semanticResiduals[i][j])
			e.release(prev)
			if err != nil {
				return signal{}, err
			}
		}
		prev := out
		out, err = e.conv(out, e.semanticBlocks[i]+"conv", 1, 1, 1)
		e.release(prev)
		if err != nil {
			return signal{}, err
		}
	}
	return out, nil
}

func (e *CodecEncoder) semanticResidual(x signal, prefix string) (signal, error) {
	work := signal{data: e.buffer(len(x.data)), channels: x.channels, frames: x.frames}
	copy(work.data, x.data)
	e.elu(work)
	out, err := e.conv(work, prefix+"conv1", 1, 1, 1)
	e.release(work)
	if err != nil {
		return signal{}, err
	}
	e.elu(out)
	prev := out
	out, err = e.conv(out, prefix+"conv2", 1, 0, 1)
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
	out, err := e.conv(x, "acoustic_encoder.conv1", 1, 3, 1)
	if err != nil {
		return signal{}, err
	}
	for i, stride := range e.weights.Rates {
		for j, dilation := range []int{1, 3, 9} {
			if err = ctx.Err(); err != nil {
				e.release(out)
				return signal{}, err
			}
			prev := out
			out, err = e.acousticResidual(out, e.acousticResiduals[i][j], dilation)
			e.release(prev)
			if err != nil {
				return signal{}, err
			}
		}
		if err = e.snake(out, e.acousticBlocks[i]+"snake1.alpha"); err != nil {
			e.release(out)
			return signal{}, err
		}
		prev := out
		out, err = e.conv(out, e.acousticBlocks[i]+"conv1", stride, (stride+1)/2, 1)
		e.release(prev)
		if err != nil {
			return signal{}, err
		}
	}
	if err = e.snake(out, "acoustic_encoder.snake1.alpha"); err != nil {
		e.release(out)
		return signal{}, err
	}
	prev := out
	out, err = e.conv(out, "acoustic_encoder.conv2", 1, 1, 1)
	e.release(prev)
	if err != nil {
		return signal{}, err
	}
	return out, nil
}

func (e *CodecEncoder) acousticResidual(x signal, prefix string, dilation int) (signal, error) {
	work := signal{data: e.buffer(len(x.data)), channels: x.channels, frames: x.frames}
	copy(work.data, x.data)
	if err := e.snake(work, prefix+"snake1.alpha"); err != nil {
		e.release(work)
		return signal{}, err
	}
	out, err := e.conv(work, prefix+"conv1", 1, 3*dilation, dilation)
	e.release(work)
	if err != nil {
		return signal{}, err
	}
	if err = e.snake(out, prefix+"snake2.alpha"); err != nil {
		e.release(out)
		return signal{}, err
	}
	prev := out
	out, err = e.conv(out, prefix+"conv2", 1, 0, 1)
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

func (e *CodecEncoder) snake(x signal, name string) error {
	alpha := e.weights.Tensors[name]
	if len(alpha) != x.channels {
		return fmt.Errorf("omnivoice: invalid snake %s", name)
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

func (e *CodecEncoder) conv(x signal, name string, stride, padding, dilation int) (signal, error) {
	op, ok := e.ops[name]
	if !ok {
		op = codecOperator{weight: e.weights.Tensors[name+".weight"], bias: e.weights.Tensors[name+".bias"], shape: e.weights.Shapes[name+".weight"]}
	}
	weight, shape, bias := op.weight, op.shape, op.bias
	if len(shape) < 2 || len(shape) > 3 || shape[1] != x.channels {
		return signal{}, fmt.Errorf("omnivoice: invalid encoder conv %s", name)
	}
	outChannels, kernel := shape[0], 1
	if len(shape) == 3 {
		kernel = shape[2]
	}
	length := convOutputLength(x.frames, kernel, stride, padding, dilation)
	if length <= 0 {
		return signal{}, fmt.Errorf("omnivoice: short encoder conv input")
	}
	y := signal{data: e.buffer(outChannels * length), channels: outChannels, frames: length}
	const tile = 64
	k := x.channels * kernel
	var packed, result []float32
	if e.scratch != nil && k*tile <= len(e.scratch.packed) && outChannels*tile <= len(e.scratch.result) {
		packed = e.scratch.packed[:k*tile]
		result = e.scratch.result[:outChannels*tile]
	} else {
		packed = make([]float32, k*tile)
		result = make([]float32, outChannels*tile)
	}
	for start := 0; start < length; start += tile {
		n := min(tile, length-start)
		p := packed[:k*n]
		clear(p)
		for c := 0; c < x.channels; c++ {
			for j := 0; j < kernel; j++ {
				for t := 0; t < n; t++ {
					source := (start+t)*stride - padding + j*dilation
					if source >= 0 && source < x.frames {
						p[(c*kernel+j)*n+t] = x.data[c*x.frames+source]
					}
				}
			}
		}
		r := result[:outChannels*n]
		clear(r)
		if !simd.SgemmNNTo(r, weight, p, outChannels, n, k, 1, k, n, n) {
			e.release(y)
			return signal{}, fmt.Errorf("omnivoice: encoder conv GEMM shape")
		}
		for c := 0; c < outChannels; c++ {
			b := float32(0)
			if len(bias) > 0 {
				if len(bias) != outChannels {
					e.release(y)
					return signal{}, fmt.Errorf("omnivoice: invalid encoder bias")
				}
				b = bias[c]
			}
			for t := 0; t < n; t++ {
				y.data[c*length+start+t] = r[c*n+t] + b
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
		prefix := e.quantizerNames[book]
		projected, err := e.conv(residual, prefix+"project_in", 1, 0, 1)
		if err != nil {
			return err
		}
		codes := dst[book*frames : (book+1)*frames]
		if err := e.nearestCodesInto(ctx, book, projected, codes); err != nil {
			e.release(projected)
			return err
		}
		quantized, err := e.lookupCodebook(book, codes)
		if err != nil {
			e.release(projected)
			return err
		}
		decoded, err := e.conv(quantized, prefix+"project_out", 1, 0, 1)
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

func (e *CodecEncoder) nearestCodesInto(ctx context.Context, book int, projected signal, dst []int) error {
	name := e.quantizerNames[book] + "codebook.embed"
	embed := e.weights.Tensors[name]
	shape := e.weights.Shapes[name]
	if len(shape) != 2 || shape[1] != projected.channels || len(dst) != projected.frames {
		return fmt.Errorf("omnivoice: invalid encoder codebook")
	}
	norms := e.codebookNorms[book]
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

func (e *CodecEncoder) lookupCodebook(book int, codes []int) (signal, error) {
	name := e.quantizerNames[book] + "codebook.embed"
	embed := e.weights.Tensors[name]
	shape := e.weights.Shapes[name]
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
