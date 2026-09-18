package omnivoice

import (
	"context"
	"fmt"
	"math"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

var hubertFeatureNames = [7]string{"semantic_model.feature_extractor.conv_layers.0.conv", "semantic_model.feature_extractor.conv_layers.1.conv", "semantic_model.feature_extractor.conv_layers.2.conv", "semantic_model.feature_extractor.conv_layers.3.conv", "semantic_model.feature_extractor.conv_layers.4.conv", "semantic_model.feature_extractor.conv_layers.5.conv", "semantic_model.feature_extractor.conv_layers.6.conv"}

const (
	hubertHidden     = 768
	hubertConvWidth  = 512
	hubertHeads      = 12
	hubertHeadDim    = 64
	hubertFFN        = 3072
	hubertLayers     = 12
	hubertLayerNormE = 1e-5
)

// Hubert owns mutable scratch and is single-caller.
type Hubert struct {
	weights               *loader.HubertWeights
	ops                   map[string]codecOperator
	posWeight             []float32
	posBias               []float32
	featureProjNormWeight []float32
	featureProjNormBias   []float32
	featureProjection     codecOperator
	encoderNormWeight     []float32
	encoderNormBias       []float32
	layers                [hubertLayers]hubertLayer
	preparedSamples       int
	preparedFrames        int
	preparedSignalCap     int
	preparedPackedCap     int
	preparedResultCap     int
	workspace             *hubertWorkspace
}

type hubertLayer struct {
	qProj, kProj, vProj codecOperator
	outProj             codecOperator
	layerNormWeight     []float32
	layerNormBias       []float32
	ffIntermediate      codecOperator
	ffOutput            codecOperator
	finalNormWeight     []float32
	finalNormBias       []float32
}

type hubertWorkspace struct {
	slots   [2][]float32
	used    [2]bool
	packed  []float32
	result  []float32
	rows    []float32
	current []float32
	next    []float32
	scratch hubertScratch
}

type hubertScratch struct {
	q, k, v        []float32
	attended       []float32
	norm           []float32
	ff             []float32
	scores         []float32
	qhead, khead   []float32
	vhead, headout []float32
}

func NewHubert(w *loader.HubertWeights) (*Hubert, error) {
	if w == nil || len(w.Tensors["semantic_model.encoder.layer_norm.weight"]) == 0 {
		return nil, fmt.Errorf("omnivoice: missing hubert tensors")
	}
	if err := validateHubertWeights(w); err != nil {
		return nil, err
	}
	h := &Hubert{
		weights:               w,
		ops:                   map[string]codecOperator{},
		posBias:               w.Tensors["semantic_model.encoder.pos_conv_embed.conv.bias"],
		featureProjNormWeight: w.Tensors["semantic_model.feature_projection.layer_norm.weight"],
		featureProjNormBias:   w.Tensors["semantic_model.feature_projection.layer_norm.bias"],
		encoderNormWeight:     w.Tensors["semantic_model.encoder.layer_norm.weight"],
		encoderNormBias:       w.Tensors["semantic_model.encoder.layer_norm.bias"],
	}
	for name, shape := range w.Shapes {
		if strings.HasSuffix(name, ".weight") {
			base := strings.TrimSuffix(name, ".weight")
			h.ops[base] = codecOperator{weight: w.Tensors[name], bias: w.Tensors[base+".bias"], shape: shape}
		}
	}
	opFor := func(name string) (codecOperator, error) {
		op, ok := h.ops[name]
		if !ok {
			return codecOperator{}, fmt.Errorf("omnivoice: missing hubert op %s", name)
		}
		return op, nil
	}
	var err error
	h.featureProjection, err = opFor("semantic_model.feature_projection.projection")
	if err != nil {
		return nil, err
	}
	for i := range h.layers {
		prefix := fmt.Sprintf("semantic_model.encoder.layers.%d.", i)
		layer := &h.layers[i]
		if layer.qProj, err = opFor(prefix + "attention.q_proj"); err != nil {
			return nil, err
		}
		if layer.kProj, err = opFor(prefix + "attention.k_proj"); err != nil {
			return nil, err
		}
		if layer.vProj, err = opFor(prefix + "attention.v_proj"); err != nil {
			return nil, err
		}
		if layer.outProj, err = opFor(prefix + "attention.out_proj"); err != nil {
			return nil, err
		}
		layer.layerNormWeight = w.Tensors[prefix+"layer_norm.weight"]
		layer.layerNormBias = w.Tensors[prefix+"layer_norm.bias"]
		if layer.ffIntermediate, err = opFor(prefix + "feed_forward.intermediate_dense"); err != nil {
			return nil, err
		}
		if layer.ffOutput, err = opFor(prefix + "feed_forward.output_dense"); err != nil {
			return nil, err
		}
		layer.finalNormWeight = w.Tensors[prefix+"final_layer_norm.weight"]
		layer.finalNormBias = w.Tensors[prefix+"final_layer_norm.bias"]
	}
	pos, err := buildHubertPosWeight(w)
	if err != nil {
		return nil, err
	}
	h.posWeight = pos
	return h, nil
}

func buildHubertPosWeight(w *loader.HubertWeights) ([]float32, error) {
	g := w.Tensors["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"]
	v := w.Tensors["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
	shape := w.Shapes["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
	if len(g) != 128 || len(shape) != 3 || shape[0] != hubertHidden || shape[1] != 48 || shape[2] != 128 {
		return nil, fmt.Errorf("omnivoice: invalid hubert positional conv")
	}
	out := make([]float32, len(v))
	const eps = 1e-12
	stride := shape[1] * shape[2]
	for k := 0; k < shape[2]; k++ {
		sum := 0.0
		for oc := 0; oc < shape[0]; oc++ {
			base := oc * stride
			for ic := 0; ic < shape[1]; ic++ {
				x := v[base+ic*shape[2]+k]
				sum += float64(x) * float64(x)
			}
		}
		norm := math.Sqrt(sum)
		if norm < eps {
			norm = eps
		}
		scale := float32(float64(g[k]) / norm)
		for oc := 0; oc < shape[0]; oc++ {
			base := oc * stride
			for ic := 0; ic < shape[1]; ic++ {
				idx := base + ic*shape[2] + k
				out[idx] = v[idx] * scale
			}
		}
	}
	return out, nil
}

func hubertOutputFrames(samples int) int {
	length := convOutputLength(samples, 10, 5, 0, 1)
	for i := 0; i < 6; i++ {
		length = convOutputLength(length, 3, 2, 0, 1)
	}
	return length
}

func newHubertScratch(frames int) hubertScratch {
	return hubertScratch{
		q:        make([]float32, frames*hubertHidden),
		k:        make([]float32, frames*hubertHidden),
		v:        make([]float32, frames*hubertHidden),
		attended: make([]float32, frames*hubertHidden),
		norm:     make([]float32, frames*hubertHidden),
		ff:       make([]float32, frames*hubertFFN),
		scores:   make([]float32, frames*frames),
		qhead:    make([]float32, frames*hubertHeadDim),
		khead:    make([]float32, frames*hubertHeadDim),
		vhead:    make([]float32, frames*hubertHeadDim),
		headout:  make([]float32, frames*hubertHeadDim),
	}
}

func (h *Hubert) Prepare(samples int) (int, error) {
	if h == nil || h.weights == nil {
		return 0, fmt.Errorf("omnivoice: nil Hubert")
	}
	if samples < 400 || samples > 20*16000+320 {
		return 0, fmt.Errorf("omnivoice: invalid hubert input")
	}
	frames := hubertOutputFrames(samples)
	if h.workspace != nil && samples == h.preparedSamples {
		return h.preparedFrames, nil
	}
	frames, signalCap, packedCap, resultCap, err := h.workspaceCaps(samples)
	if err != nil {
		return 0, err
	}
	hiddenCap := frames * hubertHidden
	rowsCap := frames * hubertConvWidth
	ffCap := frames * hubertFFN
	scoresCap := frames * frames
	headCap := frames * hubertHeadDim
	if h.workspace != nil && signalCap <= len(h.workspace.slots[0]) && packedCap <= len(h.workspace.packed) && resultCap <= len(h.workspace.result) && rowsCap <= len(h.workspace.rows) && hiddenCap <= len(h.workspace.current) && hiddenCap <= len(h.workspace.next) && hiddenCap <= len(h.workspace.scratch.q) && ffCap <= len(h.workspace.scratch.ff) && scoresCap <= len(h.workspace.scratch.scores) && headCap <= len(h.workspace.scratch.qhead) {
		h.preparedSamples = samples
		h.preparedFrames = frames
		h.preparedSignalCap = signalCap
		h.preparedPackedCap = packedCap
		h.preparedResultCap = resultCap
		return frames, nil
	}
	w := &hubertWorkspace{
		packed:  make([]float32, packedCap),
		result:  make([]float32, resultCap),
		rows:    make([]float32, rowsCap),
		current: make([]float32, hiddenCap),
		next:    make([]float32, hiddenCap),
		scratch: newHubertScratch(frames),
	}
	for i := range w.slots {
		w.slots[i] = make([]float32, signalCap)
	}
	h.workspace = w
	h.preparedSamples = samples
	h.preparedFrames = frames
	h.preparedSignalCap = signalCap
	h.preparedPackedCap = packedCap
	h.preparedResultCap = resultCap
	return frames, nil
}

// Extract runs the native HuBERT semantic path on already-16k, already-padded
// mono audio and returns the mean of the 13 hidden states as time-major
// [frames,768].
func (h *Hubert) Extract(ctx context.Context, input16k []float32) ([]float32, int, error) {
	if h == nil || h.weights == nil || ctx == nil {
		return nil, 0, fmt.Errorf("omnivoice: nil Hubert/context")
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	frames, err := h.Prepare(len(input16k))
	if err != nil {
		return nil, 0, err
	}
	out := make([]float32, frames*hubertHidden)
	if err := h.ExtractInto(ctx, out, input16k); err != nil {
		return nil, 0, err
	}
	return out, frames, nil
}

func (h *Hubert) ExtractInto(ctx context.Context, dst []float32, input16k []float32) error {
	return h.extractInto(ctx, dst, input16k, true)
}

func (h *Hubert) extractInto(ctx context.Context, dst []float32, input16k []float32, usePackedLinear bool) error {
	if h == nil || h.weights == nil || ctx == nil || len(input16k) == 0 {
		return fmt.Errorf("omnivoice: invalid hubert input")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	frames, err := h.Prepare(len(input16k))
	if err != nil {
		return err
	}
	if len(dst) != frames*hubertHidden {
		return fmt.Errorf("omnivoice: invalid hubert output")
	}
	h.workspace.used = [2]bool{}

	features, err := h.featureExtractor(signal{data: input16k, channels: 1, frames: len(input16k)})
	if err != nil {
		h.workspace.used = [2]bool{}
		return err
	}
	current := h.workspace.current[:len(dst)]
	if err := h.featureProjectionInto(current, features, usePackedLinear); err != nil {
		h.release(features)
		h.workspace.used = [2]bool{}
		return err
	}
	h.release(features)
	position := h.workspace.next[:len(dst)]
	if err := h.positionalConvInto(position, current, frames); err != nil {
		h.workspace.used = [2]bool{}
		return err
	}
	simd.VecAdd(current, current, position)
	layerNormRows(current, current, h.encoderNormWeight, h.encoderNormBias, frames, hubertHidden, hubertLayerNormE)
	copy(dst, current)
	next := h.workspace.next[:len(dst)]
	scratch := &h.workspace.scratch
	for i := range h.layers {
		if i&1 == 0 {
			if err = ctx.Err(); err != nil {
				h.workspace.used = [2]bool{}
				return err
			}
		}
		if err = h.layerInto(ctx, next, current, &h.layers[i], frames, scratch, usePackedLinear); err != nil {
			h.workspace.used = [2]bool{}
			return err
		}
		simd.VecAdd(dst, dst, next)
		current, next = next, current
	}
	simd.VecScale(dst, dst, float32(1.0/13.0))
	h.workspace.used = [2]bool{}
	return nil
}

func (h *Hubert) buffer(n int) []float32 {
	if h.workspace == nil {
		return make([]float32, n)
	}
	for i := range h.workspace.slots {
		if !h.workspace.used[i] && n <= len(h.workspace.slots[i]) {
			h.workspace.used[i] = true
			out := h.workspace.slots[i][:n]
			clear(out)
			return out
		}
	}
	panic("omnivoice: internal hubert workspace exhausted")
}

func (h *Hubert) release(x signal) {
	if h.workspace == nil || len(x.data) == 0 {
		return
	}
	for i := range h.workspace.slots {
		if len(h.workspace.slots[i]) >= len(x.data) && &h.workspace.slots[i][0] == &x.data[0] {
			h.workspace.used[i] = false
			return
		}
	}
}

func (h *Hubert) workspaceCaps(samples int) (frames, signalCap, packedCap, resultCap int, err error) {
	signalCap = samples
	trackConv := func(name string, in signal, stride, padding, dilation, groups int) (signal, error) {
		op, ok := h.ops[name]
		if !ok {
			return signal{}, fmt.Errorf("omnivoice: missing hubert conv %s", name)
		}
		shape := op.shape
		if len(shape) != 3 || groups < 1 || shape[1]*groups != in.channels || shape[0]%groups != 0 {
			return signal{}, fmt.Errorf("omnivoice: invalid hubert conv %s", name)
		}
		outChannels, inPerGroup, kernel := shape[0], shape[1], shape[2]
		length := convOutputLength(in.frames, kernel, stride, padding, dilation)
		if length <= 0 {
			return signal{}, fmt.Errorf("omnivoice: short hubert conv input")
		}
		out := signal{channels: outChannels, frames: length}
		signalCap = max(signalCap, out.channels*out.frames)
		packedCap = max(packedCap, inPerGroup*kernel*32)
		resultCap = max(resultCap, (outChannels/groups)*32)
		return out, nil
	}
	x := signal{channels: 1, frames: samples}
	x, err = trackConv("semantic_model.feature_extractor.conv_layers.0.conv", x, 5, 0, 1, 1)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	for i := 1; i <= 6; i++ {
		x, err = trackConv(hubertFeatureNames[i], x, 2, 0, 1, 1)
		if err != nil {
			return 0, 0, 0, 0, err
		}
	}
	frames = x.frames
	const (
		posGroups = 16
		posKernel = 128
		posTile   = 16
	)
	packedCap = max(packedCap, (hubertHidden/posGroups)*posKernel*posTile)
	packedCap = max(packedCap, hubertFFN*16)
	resultCap = max(resultCap, (hubertHidden/posGroups)*posTile)
	return frames, signalCap, packedCap, resultCap, nil
}

func (h *Hubert) featureExtractor(x signal) (signal, error) {
	out, err := h.convSignal(x, "semantic_model.feature_extractor.conv_layers.0.conv", 5, 0, 1, 1)
	if err != nil {
		return signal{}, err
	}
	groupNormSignal(out, h.weights.Tensors["semantic_model.feature_extractor.conv_layers.0.layer_norm.weight"], h.weights.Tensors["semantic_model.feature_extractor.conv_layers.0.layer_norm.bias"], hubertLayerNormE)
	geluSlice(out.data)
	for i := 0; i < 6; i++ {
		prev := out
		out, err = h.convSignal(out, hubertFeatureNames[i+1], 2, 0, 1, 1)
		h.release(prev)
		if err != nil {
			return signal{}, err
		}
		geluSlice(out.data)
	}
	return out, nil
}

func (h *Hubert) featureProjectionInto(dst []float32, x signal, usePackedLinear bool) error {
	if x.channels != hubertConvWidth || len(dst) != x.frames*hubertHidden {
		return fmt.Errorf("omnivoice: invalid hubert feature channels")
	}
	var rows []float32
	if h.workspace != nil && x.frames*x.channels <= len(h.workspace.rows) {
		rows = h.workspace.rows[:x.frames*x.channels]
	} else {
		rows = make([]float32, x.frames*x.channels)
	}
	for t := 0; t < x.frames; t++ {
		for c := 0; c < x.channels; c++ {
			rows[t*x.channels+c] = x.data[c*x.frames+t]
		}
	}
	layerNormRows(rows, rows, h.featureProjNormWeight, h.featureProjNormBias, x.frames, hubertConvWidth, hubertLayerNormE)
	if err := h.linearRows(dst, rows, h.featureProjection.weight, h.featureProjection.bias, x.frames, hubertConvWidth, hubertHidden, usePackedLinear); err != nil {
		return err
	}
	return nil
}

func (h *Hubert) positionalConvInto(dst, x []float32, frames int) error {
	const (
		groups = 16
		kernel = 128
		pad    = 64
	)
	if len(x) != frames*hubertHidden || len(dst) != len(x) {
		return fmt.Errorf("omnivoice: invalid hubert positional input")
	}
	outPerGroup, inPerGroup := hubertHidden/groups, hubertHidden/groups
	const tile = 16
	k := inPerGroup * kernel
	var packed, result []float32
	if h.workspace != nil && k*tile <= len(h.workspace.packed) && outPerGroup*tile <= len(h.workspace.result) {
		packed = h.workspace.packed[:k*tile]
		result = h.workspace.result[:outPerGroup*tile]
	} else {
		packed = make([]float32, k*tile)
		result = make([]float32, outPerGroup*tile)
	}
	for start := 0; start < frames; start += tile {
		n := min(tile, frames-start)
		for g := 0; g < groups; g++ {
			p := packed[:k*n]
			clear(p)
			for c := 0; c < inPerGroup; c++ {
				channel := g*inPerGroup + c
				for j := 0; j < kernel; j++ {
					for t := 0; t < n; t++ {
						source := start + t - pad + j
						if source >= 0 && source < frames {
							p[(c*kernel+j)*n+t] = x[source*hubertHidden+channel]
						}
					}
				}
			}
			r := result[:outPerGroup*n]
			clear(r)
			weight := h.posWeight[g*outPerGroup*k : (g+1)*outPerGroup*k]
			if !simd.SgemmNNTo(r, weight, p, outPerGroup, n, k, 1, k, n, n) {
				return fmt.Errorf("omnivoice: hubert positional conv GEMM shape")
			}
			for oc := 0; oc < outPerGroup; oc++ {
				bias := h.posBias[g*outPerGroup+oc]
				channel := g*outPerGroup + oc
				for t := 0; t < n; t++ {
					dst[(start+t)*hubertHidden+channel] = r[oc*n+t] + bias
				}
			}
		}
	}
	geluSlice(dst)
	return nil
}

func (h *Hubert) layerInto(ctx context.Context, dst, src []float32, layer *hubertLayer, frames int, s *hubertScratch, usePackedLinear bool) error {
	if err := h.linearRows(s.q, src, layer.qProj.weight, layer.qProj.bias, frames, hubertHidden, hubertHidden, usePackedLinear); err != nil {
		return err
	}
	if err := h.linearRows(s.k, src, layer.kProj.weight, layer.kProj.bias, frames, hubertHidden, hubertHidden, usePackedLinear); err != nil {
		return err
	}
	if err := h.linearRows(s.v, src, layer.vProj.weight, layer.vProj.bias, frames, hubertHidden, hubertHidden, usePackedLinear); err != nil {
		return err
	}
	if err := h.attentionInto(ctx, s.attended, s.q, s.k, s.v, frames, s); err != nil {
		return err
	}
	if err := h.linearRows(dst, s.attended, layer.outProj.weight, layer.outProj.bias, frames, hubertHidden, hubertHidden, usePackedLinear); err != nil {
		return err
	}
	simd.VecAdd(dst, dst, src)
	layerNormRows(s.norm, dst, layer.layerNormWeight, layer.layerNormBias, frames, hubertHidden, hubertLayerNormE)
	if err := h.linearRows(s.ff, s.norm, layer.ffIntermediate.weight, layer.ffIntermediate.bias, frames, hubertHidden, hubertFFN, usePackedLinear); err != nil {
		return err
	}
	geluSlice(s.ff)
	if err := h.linearRows(dst, s.ff, layer.ffOutput.weight, layer.ffOutput.bias, frames, hubertFFN, hubertHidden, usePackedLinear); err != nil {
		return err
	}
	simd.VecAdd(dst, dst, s.norm)
	layerNormRows(dst, dst, layer.finalNormWeight, layer.finalNormBias, frames, hubertHidden, hubertLayerNormE)
	return nil
}

func (h *Hubert) attentionInto(ctx context.Context, dst, q, k, v []float32, frames int, s *hubertScratch) error {
	const scale = float32(1.0 / 8.0)
	for head := 0; head < hubertHeads; head++ {
		if head&1 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for t := 0; t < frames; t++ {
			copy(s.qhead[t*hubertHeadDim:(t+1)*hubertHeadDim], q[t*hubertHidden+head*hubertHeadDim:t*hubertHidden+(head+1)*hubertHeadDim])
			copy(s.khead[t*hubertHeadDim:(t+1)*hubertHeadDim], k[t*hubertHidden+head*hubertHeadDim:t*hubertHidden+(head+1)*hubertHeadDim])
			copy(s.vhead[t*hubertHeadDim:(t+1)*hubertHeadDim], v[t*hubertHidden+head*hubertHeadDim:t*hubertHidden+(head+1)*hubertHeadDim])
		}
		clear(s.scores)
		if !simd.SgemmNTTo(s.scores, s.qhead, s.khead, frames, frames, hubertHeadDim, scale, hubertHeadDim, hubertHeadDim, frames) {
			return fmt.Errorf("omnivoice: hubert attention score GEMM shape")
		}
		for t := 0; t < frames; t++ {
			if !simd.SoftmaxInPlace(s.scores[t*frames : (t+1)*frames]) {
				return fmt.Errorf("omnivoice: hubert attention softmax failed")
			}
		}
		clear(s.headout)
		if !simd.SgemmNNTo(s.headout, s.scores, s.vhead, frames, hubertHeadDim, frames, 1, frames, hubertHeadDim, hubertHeadDim) {
			return fmt.Errorf("omnivoice: hubert attention value GEMM shape")
		}
		for t := 0; t < frames; t++ {
			copy(dst[t*hubertHidden+head*hubertHeadDim:t*hubertHidden+(head+1)*hubertHeadDim], s.headout[t*hubertHeadDim:(t+1)*hubertHeadDim])
		}
	}
	return nil
}

func (h *Hubert) linearRows(dst, x, weight, bias []float32, rows, in, out int, usePackedLinear bool) error {
	if usePackedLinear && h.workspace != nil {
		scratch := h.workspace.packed[:in*16]
		if err := linearRowsPacked(dst, x, weight, bias, scratch, rows, in, out); err == nil {
			return nil
		}
	}
	return linearRowsNT(dst, x, weight, bias, rows, in, out)
}

func linearRowsPacked(dst, x, weight, bias, scratch []float32, rows, in, out int) error {
	if len(scratch) < in*16 {
		return fmt.Errorf("omnivoice: invalid hubert linear scratch")
	}
	return linearRowsGEMM(dst, x, weight, bias, rows, in, out, func() bool {
		return simd.SgemmNTPackedTo(dst, x, weight, scratch[:in*16], rows, out, in, 1, in, in, out)
	})
}

func linearRowsNT(dst, x, weight, bias []float32, rows, in, out int) error {
	return linearRowsGEMM(dst, x, weight, bias, rows, in, out, func() bool {
		return simd.SgemmNTTo(dst, x, weight, rows, out, in, 1, in, in, out)
	})
}

func linearRowsGEMM(dst, x, weight, bias []float32, rows, in, out int, gemm func() bool) error {
	if len(dst) != rows*out || len(x) != rows*in || len(weight) != out*in || (len(bias) != 0 && len(bias) != out) {
		return fmt.Errorf("omnivoice: invalid hubert linear")
	}
	clear(dst)
	if !gemm() {
		return fmt.Errorf("omnivoice: hubert linear GEMM shape")
	}
	if len(bias) != 0 {
		for r := 0; r < rows; r++ {
			row := dst[r*out : (r+1)*out]
			for j, b := range bias {
				row[j] += b
			}
		}
	}
	return nil
}

func layerNormRows(dst, src, weight, bias []float32, rows, width int, eps float32) {
	for r := 0; r < rows; r++ {
		row := src[r*width : (r+1)*width]
		mean := 0.0
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(width)
		variance := 0.0
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		inv := float32(1 / math.Sqrt(variance/float64(width)+float64(eps)))
		out := dst[r*width : (r+1)*width]
		for i, v := range row {
			out[i] = (float32(float64(v)-mean) * inv) * weight[i]
			if len(bias) != 0 {
				out[i] += bias[i]
			}
		}
	}
}

func groupNormSignal(x signal, weight, bias []float32, eps float32) {
	for c := 0; c < x.channels; c++ {
		base := c * x.frames
		channel := x.data[base : base+x.frames]
		mean := 0.0
		for _, v := range channel {
			mean += float64(v)
		}
		mean /= float64(x.frames)
		variance := 0.0
		for _, v := range channel {
			d := float64(v) - mean
			variance += d * d
		}
		inv := float32(1 / math.Sqrt(variance/float64(x.frames)+float64(eps)))
		for t, v := range channel {
			x.data[base+t] = (float32(float64(v)-mean)*inv)*weight[c] + bias[c]
		}
	}
}

func geluSlice(x []float32) {
	simd.GELUErfF32To(x, x)
}

func (h *Hubert) convSignal(x signal, name string, stride, padding, dilation, groups int) (signal, error) {
	op, ok := h.ops[name]
	if !ok {
		op = codecOperator{weight: h.weights.Tensors[name+".weight"], bias: h.weights.Tensors[name+".bias"], shape: h.weights.Shapes[name+".weight"]}
	}
	weight, shape, bias := op.weight, op.shape, op.bias
	if len(shape) != 3 || groups < 1 || shape[1]*groups != x.channels || shape[0]%groups != 0 {
		return signal{}, fmt.Errorf("omnivoice: invalid hubert conv %s", name)
	}
	outChannels, inPerGroup, kernel := shape[0], shape[1], shape[2]
	length := convOutputLength(x.frames, kernel, stride, padding, dilation)
	if length <= 0 {
		return signal{}, fmt.Errorf("omnivoice: short hubert conv input")
	}
	y := signal{data: h.buffer(outChannels * length), channels: outChannels, frames: length}
	outPerGroup := outChannels / groups
	const tile = 32
	k := inPerGroup * kernel
	var packed, result []float32
	if h.workspace != nil && k*tile <= len(h.workspace.packed) && outPerGroup*tile <= len(h.workspace.result) {
		packed = h.workspace.packed[:k*tile]
		result = h.workspace.result[:outPerGroup*tile]
	} else {
		packed = make([]float32, k*tile)
		result = make([]float32, outPerGroup*tile)
	}
	for start := 0; start < length; start += tile {
		n := min(tile, length-start)
		for g := 0; g < groups; g++ {
			p := packed[:k*n]
			clear(p)
			for c := 0; c < inPerGroup; c++ {
				channel := g*inPerGroup + c
				for j := 0; j < kernel; j++ {
					for t := 0; t < n; t++ {
						source := (start+t)*stride - padding + j*dilation
						if source >= 0 && source < x.frames {
							p[(c*kernel+j)*n+t] = x.data[channel*x.frames+source]
						}
					}
				}
			}
			r := result[:outPerGroup*n]
			clear(r)
			w := weight[g*outPerGroup*k : (g+1)*outPerGroup*k]
			if !simd.SgemmNNTo(r, w, p, outPerGroup, n, k, 1, k, n, n) {
				h.release(y)
				return signal{}, fmt.Errorf("omnivoice: hubert conv GEMM shape")
			}
			for oc := 0; oc < outPerGroup; oc++ {
				b := float32(0)
				if len(bias) != 0 {
					b = bias[g*outPerGroup+oc]
				}
				channel := g*outPerGroup + oc
				for t := 0; t < n; t++ {
					y.data[channel*length+start+t] = r[oc*n+t] + b
				}
			}
		}
	}
	return y, nil
}
