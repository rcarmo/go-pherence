package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"math"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanAudioLayer runs one complete, unmasked audio layer with resident F32
// activations. Input is one fully valid [rows,512] window; no speaker cache,
// frontend or head is included. It is fixed to the released layer's weights
// and row count. Call Close even after a failed Forward.
type VulkanAudioLayer struct {
	input, output *vk.VkTensorF32
	plan          *vk.VkF32Plan
	resources     []interface{ Close() error }
	rows          int
	closed        bool
	activations   [10]*vk.VkTensorF32 // normal, Q, K, V, mixed, attention, residual, MLP norm, intermediate, fc2
	weights       [14]*vk.VkTensorF32 // zero bias, norm1, Q/K/V, output, norm2, fc1/fc2
	frequency     *vk.VkTensorF32
	norm          *vk.VkLayerNormF32
	linear        *vk.VkLinearF32
	rope          *vk.VkRoPESequenceF32
	attn          *vk.VkAttentionF32
	add           *vk.VkAddF32
	gelu          *vk.VkGELUErfF32
}

func (l *VulkanAudioLayer) Close() error {
	if l == nil || l.closed {
		return nil
	}
	// Plans and stages can reference earlier resources. Stop at the first
	// failed close and retain it plus all upstream owners for VulkanDrain retry.
	for i := len(l.resources) - 1; i >= 0; i-- {
		if l.resources[i] == nil {
			continue
		}
		if err := l.resources[i].Close(); err != nil {
			return err
		}
		l.resources[i] = nil
	}
	l.closed = true
	l.resources = nil
	l.plan, l.input, l.output = nil, nil, nil
	l.activations, l.weights = [10]*vk.VkTensorF32{}, [14]*vk.VkTensorF32{}
	l.norm, l.linear, l.rope, l.attn, l.add, l.gelu, l.frequency = nil, nil, nil, nil, nil, nil, nil
	return nil
}

// NewVulkanAudioLayer prepares one reusable plan and uploaded layer-1..30
// weights. It never modifies or retains the source weight slices. A failed
// constructor releases every resource acquired so far.
func NewVulkanAudioLayer(ctx context.Context, source *Layer1Complete, rows int) (*VulkanAudioLayer, error) {
	return newVulkanAudioLayerFromTensor(ctx, source, rows, nil)
}

// newVulkanAudioLayerFromTensor binds the previous layer's resident output
// as this layer's input. Its owner must outlive this layer and close afterward.
func newVulkanAudioLayerFromTensor(ctx context.Context, source *Layer1Complete, rows int, previous *vk.VkTensorF32) (result *VulkanAudioLayer, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil Nemotron Vulkan layer context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil || source.attention == nil || source.attention.qkv == nil || rows < 1 || rows > maxPreparedDiarizationRows ||
		len(source.attention.qkv.gamma) != projectedWidth || len(source.attention.qkv.beta) != projectedWidth ||
		len(source.attention.qkv.q) != projectedWidth*projectedWidth || len(source.attention.qkv.k) != projectedWidth*projectedWidth || len(source.attention.qkv.v) != projectedWidth*projectedWidth ||
		len(source.attention.outWeight) != projectedWidth*projectedWidth || len(source.attention.outBias) != projectedWidth ||
		len(source.normWeight) != projectedWidth || len(source.normBias) != projectedWidth ||
		len(source.fc1Weight) != diarizationIntermediate*projectedWidth || len(source.fc1Bias) != diarizationIntermediate ||
		len(source.fc2Weight) != projectedWidth*diarizationIntermediate || len(source.fc2Bias) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron Vulkan layer weights or row count")
	}
	if !vk.VulkanInit() {
		return nil, fmt.Errorf("Nemotron Vulkan layer unavailable")
	}
	layer := &VulkanAudioLayer{rows: rows}
	defer func() {
		if err != nil {
			err = errors.Join(err, layer.Close())
			if len(layer.resources) != 0 {
				result = layer // retain cleanup ownership for retry
			}
		}
	}()
	arena, err := vk.NewVkTensorArena(ctx, 32<<20)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, arena)
	alloc := func(shape ...int) (*vk.VkTensorF32, error) { return arena.AllocF32(ctx, shape...) }
	upload := func(values []float32, shape ...int) (*vk.VkTensorF32, error) {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron Vulkan layer weight")
			}
		}
		t, e := alloc(shape...)
		if e != nil {
			return nil, e
		}
		return t, t.Upload(ctx, values)
	}
	width := projectedWidth
	if previous != nil {
		shape := previous.Shape()
		if len(shape) != 2 || shape[0] != rows || shape[1] != width {
			return nil, fmt.Errorf("invalid Nemotron Vulkan upstream layer tensor")
		}
		layer.input = previous
	} else {
		layer.input, err = alloc(rows, width)
		if err != nil {
			return nil, err
		}
	}
	normal, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	q, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	k, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	v, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	mixed, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	attention, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	residual, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	mlpNormal, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	intermediate, err := alloc(rows, diarizationIntermediate)
	if err != nil {
		return nil, err
	}
	fc2, err := alloc(rows, width)
	if err != nil {
		return nil, err
	}
	layer.output, err = alloc(rows, width)
	if err != nil {
		return nil, err
	}
	zero := make([]float32, width)
	zeroBias, err := upload(zero, width)
	if err != nil {
		return nil, err
	}
	norm1Weight, err := upload(source.attention.qkv.gamma, width)
	if err != nil {
		return nil, err
	}
	norm1Bias, err := upload(source.attention.qkv.beta, width)
	if err != nil {
		return nil, err
	}
	qWeight, err := upload(source.attention.qkv.q, width, width)
	if err != nil {
		return nil, err
	}
	kWeight, err := upload(source.attention.qkv.k, width, width)
	if err != nil {
		return nil, err
	}
	vWeight, err := upload(source.attention.qkv.v, width, width)
	if err != nil {
		return nil, err
	}
	outWeight, err := upload(source.attention.outWeight, width, width)
	if err != nil {
		return nil, err
	}
	outBias, err := upload(source.attention.outBias, width)
	if err != nil {
		return nil, err
	}
	norm2Weight, err := upload(source.normWeight, width)
	if err != nil {
		return nil, err
	}
	norm2Bias, err := upload(source.normBias, width)
	if err != nil {
		return nil, err
	}
	fc1Weight, err := upload(source.fc1Weight, diarizationIntermediate, width)
	if err != nil {
		return nil, err
	}
	fc1Bias, err := upload(source.fc1Bias, diarizationIntermediate)
	if err != nil {
		return nil, err
	}
	fc2Weight, err := upload(source.fc2Weight, width, diarizationIntermediate)
	if err != nil {
		return nil, err
	}
	fc2Bias, err := upload(source.fc2Bias, width)
	if err != nil {
		return nil, err
	}
	freqs := make([]float32, rows*(diarizationHeadWidth/2)*2)
	for row := 0; row < rows; row++ {
		for d := 0; d < diarizationHeadWidth/2; d++ {
			angle := float32(row) * float32(1/math.Pow(10000, float64(2*d)/diarizationHeadWidth))
			freqs[(row*(diarizationHeadWidth/2)+d)*2] = float32(math.Cos(float64(angle)))
			freqs[(row*(diarizationHeadWidth/2)+d)*2+1] = float32(math.Sin(float64(angle)))
		}
	}
	frequency, err := upload(freqs, rows, diarizationHeadWidth)
	if err != nil {
		return nil, err
	}
	newNorm := func() (*vk.VkLayerNormF32, error) { return vk.NewVkLayerNormF32(ctx) }
	norm, err := newNorm()
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, norm)
	linear, err := vk.NewVkLinearRegTileF32(ctx)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, linear)
	rope, err := vk.NewVkRoPESequenceF32(ctx)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, rope)
	attn, err := vk.NewVkAttentionF32(ctx)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, attn)
	add, err := vk.NewVkAddF32(ctx)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, add)
	gelu, err := vk.NewVkGELUErfF32(ctx)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, gelu)
	layer.activations = [10]*vk.VkTensorF32{normal, q, k, v, mixed, attention, residual, mlpNormal, intermediate, fc2}
	layer.weights = [14]*vk.VkTensorF32{zeroBias, norm1Weight, norm1Bias, qWeight, kWeight, vWeight, outWeight, outBias, norm2Weight, norm2Bias, fc1Weight, fc1Bias, fc2Weight, fc2Bias}
	layer.frequency = frequency
	layer.norm, layer.linear, layer.rope, layer.attn, layer.add, layer.gelu = norm, linear, rope, attn, add, gelu
	stages := make([]vk.VkF32Stage, 0, 15)
	appendStage := func(stage vk.VkF32Stage, e error) error {
		if e != nil {
			return e
		}
		stages = append(stages, stage)
		return nil
	}
	if err = appendStage(norm.Stage(ctx, normal, layer.input, norm1Weight, norm1Bias, 1e-5)); err != nil {
		return nil, err
	}
	if err = appendStage(linear.Stage(ctx, q, normal, qWeight, zeroBias)); err != nil {
		return nil, err
	}
	if err = appendStage(linear.Stage(ctx, k, normal, kWeight, zeroBias)); err != nil {
		return nil, err
	}
	if err = appendStage(linear.Stage(ctx, v, normal, vWeight, zeroBias)); err != nil {
		return nil, err
	}
	if err = appendStage(rope.Stage(ctx, q, frequency, diarizationHeads)); err != nil {
		return nil, err
	}
	if err = appendStage(rope.Stage(ctx, k, frequency, diarizationHeads)); err != nil {
		return nil, err
	}
	if err = appendStage(attn.Stage(ctx, mixed, q, k, v, diarizationHeads)); err != nil {
		return nil, err
	}
	if err = appendStage(linear.Stage(ctx, attention, mixed, outWeight, outBias)); err != nil {
		return nil, err
	}
	if err = appendStage(add.Stage(ctx, residual, layer.input, attention)); err != nil {
		return nil, err
	}
	if err = appendStage(norm.Stage(ctx, mlpNormal, residual, norm2Weight, norm2Bias, 1e-5)); err != nil {
		return nil, err
	}
	if err = appendStage(linear.Stage(ctx, intermediate, mlpNormal, fc1Weight, fc1Bias)); err != nil {
		return nil, err
	}
	if err = appendStage(gelu.Stage(ctx, intermediate, intermediate)); err != nil {
		return nil, err
	}
	if err = appendStage(linear.Stage(ctx, fc2, intermediate, fc2Weight, fc2Bias)); err != nil {
		return nil, err
	}
	if err = appendStage(add.Stage(ctx, layer.output, residual, fc2)); err != nil {
		return nil, err
	}
	layer.plan, err = vk.NewVkF32Plan(ctx, stages)
	if err != nil {
		return nil, err
	}
	layer.resources = append(layer.resources, layer.plan)
	return layer, nil
}

// rebindRows stages the exact prefix of each maximum-row activation; weight
// bindings and kernels remain resident. RoPE and attention see the live row
// count, including the exact softmax key count. Caller serializes plan use.
func (l *VulkanAudioLayer) rebindRows(ctx context.Context, rows int, previous *vk.VkTensorF32) error {
	if l == nil || l.closed || l.plan == nil || ctx == nil || rows < 1 || rows > l.rows || l.input == nil || l.output == nil || l.frequency == nil || l.norm == nil || l.linear == nil || l.rope == nil || l.attn == nil || l.add == nil || l.gelu == nil {
		return fmt.Errorf("invalid Nemotron Vulkan layer rebind")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	prefix := func(t *vk.VkTensorF32) (*vk.VkTensorF32, error) { return t.PrefixRows(ctx, rows) }
	input := l.input
	if previous != nil {
		input = previous
	} else {
		var err error
		input, err = prefix(input)
		if err != nil {
			return err
		}
	}
	shape := input.Shape()
	if len(shape) != 2 || shape[0] != rows || shape[1] != projectedWidth {
		return fmt.Errorf("invalid Nemotron Vulkan previous layer rows")
	}
	var activation [10]*vk.VkTensorF32
	for i, t := range l.activations {
		var err error
		activation[i], err = prefix(t)
		if err != nil {
			return err
		}
	}
	output, err := prefix(l.output)
	if err != nil {
		return err
	}
	frequency, err := prefix(l.frequency)
	if err != nil {
		return err
	}
	w := l.weights
	stages := make([]vk.VkF32Stage, 0, 14)
	appendStage := func(stage vk.VkF32Stage, stageErr error) error {
		if stageErr != nil {
			return stageErr
		}
		stages = append(stages, stage)
		return nil
	}
	if err = appendStage(l.norm.Stage(ctx, activation[0], input, w[1], w[2], 1e-5)); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if err = appendStage(l.linear.Stage(ctx, activation[1+i], activation[0], w[3+i], w[0])); err != nil {
			return err
		}
	}
	if err = appendStage(l.rope.Stage(ctx, activation[1], frequency, diarizationHeads)); err != nil {
		return err
	}
	if err = appendStage(l.rope.Stage(ctx, activation[2], frequency, diarizationHeads)); err != nil {
		return err
	}
	if err = appendStage(l.attn.Stage(ctx, activation[4], activation[1], activation[2], activation[3], diarizationHeads)); err != nil {
		return err
	}
	if err = appendStage(l.linear.Stage(ctx, activation[5], activation[4], w[6], w[7])); err != nil {
		return err
	}
	if err = appendStage(l.add.Stage(ctx, activation[6], input, activation[5])); err != nil {
		return err
	}
	if err = appendStage(l.norm.Stage(ctx, activation[7], activation[6], w[8], w[9], 1e-5)); err != nil {
		return err
	}
	if err = appendStage(l.linear.Stage(ctx, activation[8], activation[7], w[10], w[11])); err != nil {
		return err
	}
	if err = appendStage(l.gelu.Stage(ctx, activation[8], activation[8])); err != nil {
		return err
	}
	if err = appendStage(l.linear.Stage(ctx, activation[9], activation[8], w[12], w[13])); err != nil {
		return err
	}
	if err = appendStage(l.add.Stage(ctx, output, activation[6], activation[9])); err != nil {
		return err
	}
	return l.plan.Rebind(ctx, stages)
}

func (l *VulkanAudioLayer) Forward(ctx context.Context, input []float32) ([]float32, error) {
	if l == nil || l.closed || l.plan == nil || ctx == nil || len(input) != l.rows*projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron Vulkan layer input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, v := range input {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron Vulkan layer input")
		}
	}
	if err := l.input.Upload(ctx, input); err != nil {
		return nil, err
	}
	if err := l.runResident(ctx); err != nil {
		return nil, err
	}
	output := make([]float32, len(input))
	if err := l.output.Download(ctx, output); err != nil {
		return nil, err
	}
	for _, v := range output {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron Vulkan layer output")
		}
	}
	return output, nil
}

func (l *VulkanAudioLayer) runResident(ctx context.Context) error {
	if l == nil || l.closed || l.plan == nil || ctx == nil {
		return fmt.Errorf("invalid resident Nemotron Vulkan layer")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return l.plan.Run(ctx)
}
