package whisper

import (
	"context"
	"errors"
	"fmt"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"math"
	"strings"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanEncoder is an explicit fixed-frame encoder with F32 activations,
// accumulation and output. It owns all uploaded/packed weights, scratch tensors,
// operators and private plans. No model default or decoder/media path changes.
// Copies share ownership/serialisation. Call Close.
// Forward uploads mel once and downloads the final hidden state once; all
// intermediate activations remain on the device, with a fence per layer.
type VulkanEncoder struct{ s *vulkanEncoderState }
type VulkanEncoderStats struct {
	Frames, Rows, Width, Layers, Plans, Stages int
	WeightBytes, ScratchBytes                  uint64
}
type vkEncoderCloser interface{ Close() error }
type vulkanEncoderState struct {
	gate             chan struct{}
	stopping, closed bool
	stats            VulkanEncoderStats
	config           Config // immutable source geometry for checked PCM admission
	resources        []vkEncoderCloser
	plans            []*vk.VkF32Plan
	input, output    *vk.VkTensorF32
	tensors          map[string]*vk.VkTensorF32 // owned, private diagnostic views
	conv             *vk.VkConv1D3F32
	add              *vk.VkAddF32
	norm             *vk.VkLayerNormF32
	linear           *vk.VkLinearF32
	q8Linear         *vk.VkLinearQ8WeightSet
	q5Linear         *vk.VkLinearQ5Set
	q5Dot            *vk.VkLinearQ5IntegerDotSet
	gelu             *vk.VkGELUErfF32
	attention        *vk.VkAttentionF32
}

// NewVulkanEncoder requires explicit prior VulkanInit by the caller. It validates
// the complete F32 encoder before allocating, copies weights to owned arenas,
// and snapshots geometry for exactly frames input columns. Source slices may be
// released/changed after return, but must not be mutated during construction.
// The constructor uses the qualified register-tiled F32 linear kernel and has
// no implicit fallback. On construction failure normal rollback returns nil; if cleanup also fails, a
// nonnil stopping encoder is
// returned with the error so the caller can retry Close after VulkanDrain.
func NewVulkanEncoder(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoder(ctx, source, frames, vk.NewVkF32Plan)
}

// NewVulkanEncoderRegTile64 explicitly opts into the 64x64 F32 projection
// candidate. All encoder arithmetic and attention precision remain F32.
func NewVulkanEncoderRegTile64(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearF32RegTile64)
}

// NewVulkanEncoderTile64Key32 explicitly selects F32 tile64 projections and
// the separately qualified key32 FMA online-softmax attention candidate.
func NewVulkanEncoderTile64Key32(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearF32Tile64Key32)
}

// NewVulkanEncoderTile64Key32ScoreILP selects F32 tile64 projections and
// explicit key32 score-interleaved attention. Defaults remain unchanged.
func NewVulkanEncoderTile64Key32ScoreILP(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearF32Tile64Key32ScoreILP)
}

// NewVulkanEncoderOriginalQ5Decode4 retains original Q5 values and ordered F32
// FFN arithmetic with an explicit four-lane block decoder. No fallback/default.
func NewVulkanEncoderOriginalQ5Decode4(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5Decode4, file)
}

// NewVulkanEncoderOriginalQ5ExactCombined explicitly combines decode4 FFN and
// headDim64 attention unroll4. Stored values, ordered F32 arithmetic and resource
// geometry are unchanged. Requires matching original Q5 source; no fallback.
func NewVulkanEncoderOriginalQ5ExactCombined(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5ExactCombined, file)
}

// NewVulkanEncoderOriginalQ5AttentionOutputILP combines decode4 FFN and F32
// headDim64 attention with interleaved ordered output FMAs. Explicit, no fallback.
func NewVulkanEncoderOriginalQ5AttentionOutputILP(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5AttentionOutputILP, file)
}

// NewVulkanEncoderOriginalQ5AttentionUnroll4 keeps original Q5 FFN/F32 values
// and selects exact-order headDim64 attention. No fallback or default change.
func NewVulkanEncoderOriginalQ5AttentionUnroll4(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5AttentionUnroll4, file)
}

// NewVulkanEncoderOriginalQ5IntegerDotFC1 quantises only FC1 activations; FC2
// retains ordered packed Q5/F32 arithmetic over the same original storage. This
// explicit experimental mode requires integer-dot enablement and original Q5.
// Activation bounds/NaN rejection match the full-FFN integer-dot candidate;
// independent acoustic accuracy remains unqualified. No device/CPU fallback.
func NewVulkanEncoderOriginalQ5IntegerDotFC1(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5IntegerDotFC1, file)
}

// NewVulkanEncoderOriginalQ5IntegerDotMLP quantises both FFN activations to
// Q8_1. Requires explicit integer-dot and original Q5. Magnitude<=1000/nonzero
// block maximum>=1e-30; invalid blocks yield nonfinite output rejected by Forward.
// This mode has a recorded Portuguese timing regression and is not qualified.
func NewVulkanEncoderOriginalQ5IntegerDotMLP(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5IntegerDotMLP, file)
}

// NewVulkanEncoderOriginalQ5MLP retains original Q5_0 FFN weights with F32
// activations/attention. Source/file must identify the same exact values. File
// is read synchronously; no fallback, requantisation or retained handle.
func NewVulkanEncoderOriginalQ5MLP(ctx context.Context, source *Encoder, frames int, file *legacy.File) (*VulkanEncoder, error) {
	if file == nil {
		return nil, fmt.Errorf("whisper Vulkan Q5: nil pinned file")
	}
	return newVulkanEncoderPackedMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearOriginalQ5MLP, file)
}

// NewVulkanEncoderQ8Weight explicitly selects per-output-row symmetric Q8 for
// transformer projection weights only. Stem convolutions, normalization,
// activations, attention, accumulation and output remain F32. This candidate is
// never selected by NewVulkanEncoder or a serving default.
func NewVulkanEncoderQ8Weight(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearQ8Weight)
}

// NewVulkanEncoderQ8MLPWeight selects Q8 only for FC1/FC2 projection weights.
// Attention projections remain on the existing F32 kernel. This narrower
// explicit candidate preserves strict multi-window timestamps on retained gates.
func NewVulkanEncoderQ8MLPWeight(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearQ8MLPWeight)
}

// NewVulkanEncoderQ8KVMLPWeight adds K/V projections to MLP-only Q8 while
// retaining Q/O projections in F32. This broad selective placement preserves
// strict timestamps on retained robustness gates and remains explicit/non-default.
func NewVulkanEncoderQ8KVMLPWeight(ctx context.Context, source *Encoder, frames int) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, vk.NewVkF32Plan, vulkanLinearQ8KVMLPWeight)
}

// Private plan-construction seam for fault injection. The public constructor
// always uses NewVkF32Plan; no stages escape its owner.
func newVulkanEncoder(ctx context.Context, source *Encoder, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error)) (*VulkanEncoder, error) {
	return newVulkanEncoderMode(ctx, source, frames, makePlan, vulkanDefaultLinearMode)
}

// Legacy/candidate selection remains private for controlled comparisons.
func newVulkanEncoderVariant(ctx context.Context, source *Encoder, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error), registerTile bool) (result *VulkanEncoder, err error) {
	mode := vulkanLinearF32
	if registerTile {
		mode = vulkanLinearF32RegTile
	}
	return newVulkanEncoderMode(ctx, source, frames, makePlan, mode)
}

type vulkanLinearMode uint8

const (
	vulkanLinearF32 vulkanLinearMode = iota
	vulkanLinearF32RegTile
	vulkanLinearF32RegTile64
	vulkanLinearF32Tile64Key32
	vulkanLinearF32Tile64Key32ScoreILP
	vulkanLinearOriginalQ5MLP
	vulkanLinearOriginalQ5Decode4
	vulkanLinearOriginalQ5ExactCombined
	vulkanLinearOriginalQ5AttentionOutputILP
	vulkanLinearOriginalQ5AttentionUnroll4
	vulkanLinearOriginalQ5IntegerDotMLP
	vulkanLinearOriginalQ5IntegerDotFC1
	vulkanLinearQ8Weight
	vulkanLinearQ8MLPWeight
	vulkanLinearQ8KVMLPWeight
)

const vulkanDefaultLinearMode = vulkanLinearF32RegTile

func vulkanQ8WeightSelected(mode vulkanLinearMode, name string) bool {
	if mode == vulkanLinearQ8Weight {
		return true
	}
	mlp := strings.HasSuffix(name, ".fc1.w") || strings.HasSuffix(name, ".fc2.w")
	if mode == vulkanLinearQ8MLPWeight {
		return mlp
	}
	return mode == vulkanLinearQ8KVMLPWeight && (mlp || strings.HasSuffix(name, ".k.w") || strings.HasSuffix(name, ".v.w"))
}

func newVulkanEncoderMode(ctx context.Context, source *Encoder, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error), linearMode vulkanLinearMode) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedMode(ctx, source, frames, makePlan, linearMode, nil)
}
func newVulkanEncoderPackedMode(ctx context.Context, source *Encoder, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error), linearMode vulkanLinearMode, file *legacy.File) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedOnlyMode(ctx, source, frames, makePlan, linearMode, file, false)
}
func newVulkanEncoderPackedOnlyMode(ctx context.Context, source *Encoder, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error), linearMode vulkanLinearMode, file *legacy.File, packedOnly bool) (*VulkanEncoder, error) {
	return newVulkanEncoderPackedSelected(ctx, source, frames, makePlan, linearMode, file, packedOnly, nil)
}

// Private selector seam for numerical attribution only. Public constructors use
// nil (all FFN). Partial selection requires the complete widened source.
func newVulkanEncoderPackedSelected(ctx context.Context, source *Encoder, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error), linearMode vulkanLinearMode, file *legacy.File, packedOnly bool, dotSelect func(string) bool) (result *VulkanEncoder, err error) {
	if dotSelect != nil && (packedOnly || linearMode != vulkanLinearOriginalQ5IntegerDotMLP) {
		return nil, fmt.Errorf("Whisper: diagnostic selection requires widened integer-dot source")
	}
	if makePlan == nil {
		return nil, fmt.Errorf("whisper Vulkan: nil plan constructor")
	}
	if packedOnly && (file == nil || (linearMode != vulkanLinearOriginalQ5MLP && linearMode != vulkanLinearOriginalQ5Decode4 && linearMode != vulkanLinearOriginalQ5ExactCombined && linearMode != vulkanLinearOriginalQ5AttentionOutputILP && linearMode != vulkanLinearOriginalQ5AttentionUnroll4 && linearMode != vulkanLinearOriginalQ5IntegerDotMLP && linearMode != vulkanLinearOriginalQ5IntegerDotFC1)) {
		return nil, fmt.Errorf("whisper Vulkan: packed-only mode requires Q5 source")
	}
	layout, err := describeVulkanEncoderMode(ctx, source, frames, packedOnly)
	if err != nil {
		return nil, err
	}
	limits, err := vk.VulkanLimits()
	if err != nil {
		return nil, err
	}
	hybrid := linearMode == vulkanLinearOriginalQ5IntegerDotFC1
	dot := linearMode == vulkanLinearOriginalQ5IntegerDotMLP || hybrid
	if dot && !vk.VulkanIntegerDotEnabled() {
		return nil, fmt.Errorf("Whisper integer-dot requires explicit device enablement")
	}
	outputILP := linearMode == vulkanLinearOriginalQ5AttentionOutputILP
	unroll := linearMode == vulkanLinearOriginalQ5AttentionUnroll4 || linearMode == vulkanLinearOriginalQ5ExactCombined || outputILP
	if unroll && layout.cfg.HeadDim != 64 {
		return nil, fmt.Errorf("Whisper attention unroll requires headDim64")
	}
	decode4 := linearMode == vulkanLinearOriginalQ5Decode4 || linearMode == vulkanLinearOriginalQ5ExactCombined || outputILP
	q5 := linearMode == vulkanLinearOriginalQ5MLP || dot || unroll || decode4
	if dot && !hybrid {
		layout.scratch = append(append([]vkEncoderTensor(nil), layout.scratch...), vkEncoderTensor{name: "q8fc1", shape: []int{layout.rows * (layout.cfg.EncoderDModel / 32) * 9}}, vkEncoderTensor{name: "q8fc2", shape: []int{layout.rows * (layout.cfg.EncoderFFNDim / 32) * 9}})
	}
	if hybrid {
		layout.scratch = append(append([]vkEncoderTensor(nil), layout.scratch...), vkEncoderTensor{name: "q8fc1", shape: []int{layout.rows * (layout.cfg.EncoderDModel / 32) * 9}})
	}
	if q5 && file == nil {
		return nil, fmt.Errorf("whisper Vulkan Q5: require pinned original file")
	}
	weightSpecs := layout.weights
	linearWeights := map[string]vkEncoderTensor{}
	linearIndexes := map[string]int{}
	if linearMode == vulkanLinearQ8Weight || linearMode == vulkanLinearQ8MLPWeight || linearMode == vulkanLinearQ8KVMLPWeight || q5 {
		weightSpecs = make([][]vkEncoderTensor, len(layout.weights))
		for _, plan := range layout.plans {
			for _, step := range plan {
				if step.op == "linear" && (vulkanQ8WeightSelected(linearMode, step.in[1]) || q5 && isVulkanMLPWeight(step.in[1]) && (dotSelect == nil || dotSelect(step.in[1]))) {
					linearWeights[step.in[1]] = vkEncoderTensor{}
				}
			}
		}
		for i, specs := range layout.weights {
			for _, spec := range specs {
				if _, quantized := linearWeights[spec.name]; quantized {
					linearWeights[spec.name] = spec
				} else {
					weightSpecs[i] = append(weightSpecs[i], spec)
				}
			}
		}
	}
	allSpecs := append(append([][]vkEncoderTensor(nil), weightSpecs...), layout.scratch)
	sizes := make([]uint64, len(allSpecs))
	for i, specs := range allSpecs {
		sizes[i], err = vkEncoderArenaBytes(specs, limits.StorageBufferOffsetAlignment)
		if err != nil {
			return nil, err
		}
		if sizes[i] > uint64(limits.StorageBufferRange) || sizes[i] > uint64(^uint(0)>>1) {
			return nil, fmt.Errorf("whisper Vulkan: arena%d exceeds device range", i)
		}
	}
	s := &vulkanEncoderState{gate: make(chan struct{}, 1), config: layout.cfg, stats: VulkanEncoderStats{Frames: frames, Rows: layout.rows, Width: layout.cfg.EncoderDModel, Layers: layout.cfg.EncoderLayers, Plans: len(layout.plans), ScratchBytes: sizes[len(sizes)-1]}}
	for _, n := range sizes[:len(sizes)-1] {
		s.stats.WeightBytes += n
	}
	for _, plan := range layout.plans {
		s.stats.Stages += len(plan)
	}
	e := &VulkanEncoder{s: s}
	defer func() {
		if err != nil {
			if closeErr := e.Close(); closeErr != nil {
				result = e
				err = errors.Join(err, fmt.Errorf("whisper Vulkan: rollback: %w", closeErr))
			}
		}
	}()
	// All created objects join the rollback list immediately. Reverse close order
	// releases plans first, then arena backing storage, then operator pipelines.
	if s.conv, err = vk.NewVkConv1D3F32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.conv)
	if s.add, err = vk.NewVkAddF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.add)
	if s.norm, err = vk.NewVkLayerNormF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.norm)
	if q5 {
		shapes := make([]vk.VkLinearQ5Shape, 0, len(linearWeights))
		var names, originalNames []string
		for layer := 0; layer < layout.cfg.EncoderLayers; layer++ {
			for _, part := range []struct {
				step string
				mlp  int
			}{{"fc1", 0}, {"fc2", 2}} {
				name := fmt.Sprintf("layer%d.%s.w", layer, part.step)
				spec, selected := linearWeights[name]
				if !selected {
					continue
				}
				if len(spec.shape) != 2 {
					return nil, fmt.Errorf("whisper Vulkan Q5: shape %s", name)
				}
				linearIndexes[name] = len(shapes)
				shapes = append(shapes, vk.VkLinearQ5Shape{OutDim: spec.shape[0], InDim: spec.shape[1]})
				names = append(names, name)
				originalNames = append(originalNames, fmt.Sprintf("encoder.blocks.%d.mlp.%d.weight", layer, part.mlp))
			}
		}
		reader := func(ctx context.Context, index int) ([]byte, error) {
			spec := linearWeights[names[index]]
			raw, shape, e := file.Q5Blocks(ctx, originalNames[index])
			if e != nil {
				return nil, e
			}
			if len(shape) != 2 || shape[0] != spec.shape[1] || shape[1] != spec.shape[0] {
				return nil, fmt.Errorf("whisper Vulkan Q5: shape %s", names[index])
			}
			if !packedOnly {
				if e := checkOriginalQ5Values(ctx, raw, spec.data); e != nil {
					return nil, e
				}
			}
			return raw, nil
		}
		if dot {
			if hybrid {
				s.q5Dot, err = vk.NewVkLinearQ5IntegerDotHybridSetStream(ctx, shapes, reader)
			} else {
				s.q5Dot, err = vk.NewVkLinearQ5IntegerDotSetStream(ctx, shapes, reader)
			}
			if s.q5Dot != nil {
				s.resources = append(s.resources, s.q5Dot)
				s.stats.WeightBytes += s.q5Dot.StorageBytes()
			}
			if hybrid {
				s.stats.Stages += layout.cfg.EncoderLayers
			} else {
				s.stats.Stages += len(shapes)
			}
		} else {
			if decode4 {
				s.q5Linear, err = vk.NewVkLinearQ5Decode4SetStream(ctx, shapes, reader)
			} else {
				s.q5Linear, err = vk.NewVkLinearQ5SetStream(ctx, shapes, reader)
			}
		}
		if s.q5Linear != nil {
			s.resources = append(s.resources, s.q5Linear)
			s.stats.WeightBytes += s.q5Linear.StorageBytes()
		}
		if err != nil {
			return nil, err
		}
	} else if linearMode == vulkanLinearQ8Weight || linearMode == vulkanLinearQ8MLPWeight || linearMode == vulkanLinearQ8KVMLPWeight {
		matrices := make([]vk.VkLinearQ8WeightMatrix, 0, len(linearWeights))
		for _, plan := range layout.plans {
			for _, step := range plan {
				if step.op != "linear" || !vulkanQ8WeightSelected(linearMode, step.in[1]) {
					continue
				}
				weight := linearWeights[step.in[1]]
				if len(weight.shape) != 2 || len(weight.data) == 0 {
					return nil, fmt.Errorf("whisper Vulkan: missing Q8 weight %q", step.in[1])
				}
				linearIndexes[step.in[1]] = len(matrices)
				matrices = append(matrices, vk.VkLinearQ8WeightMatrix{Weights: weight.data, OutDim: weight.shape[0], InDim: weight.shape[1]})
			}
		}
		s.q8Linear, err = vk.NewVkLinearQ8WeightSet(ctx, matrices)
		if s.q8Linear != nil {
			s.resources = append(s.resources, s.q8Linear)
			s.stats.WeightBytes += s.q8Linear.StorageBytes()
		}
		if err != nil {
			return nil, err
		}
	}
	if linearMode != vulkanLinearQ8Weight {
		if linearMode == vulkanLinearF32RegTile64 || linearMode == vulkanLinearF32Tile64Key32 || (linearMode == vulkanLinearF32Tile64Key32ScoreILP || q5) {
			s.linear, err = vk.NewVkLinearRegTile64F32(ctx)
		} else if linearMode == vulkanLinearF32RegTile {
			s.linear, err = vk.NewVkLinearRegTileF32(ctx)
		} else {
			s.linear, err = vk.NewVkLinearF32(ctx)
		}
		if err != nil {
			return nil, err
		}
		s.resources = append(s.resources, s.linear)
	}
	if s.gelu, err = vk.NewVkGELUErfF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.gelu)
	if outputILP {
		s.attention, err = vk.NewVkAttentionKey32OutputILPF32(ctx)
	} else if unroll {
		s.attention, err = vk.NewVkAttentionKey32ScoreILPUnroll4F32(ctx)
	} else if linearMode == vulkanLinearF32Tile64Key32ScoreILP || q5 {
		s.attention, err = vk.NewVkAttentionKey32ScoreILPF32(ctx)
	} else if linearMode == vulkanLinearF32Tile64Key32 {
		s.attention, err = vk.NewVkAttentionKey32F32(ctx)
	} else {
		s.attention, err = vk.NewVkAttentionF32(ctx)
	}
	if err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.attention)
	tensors := map[string]*vk.VkTensorF32{}
	s.tensors = tensors
	for i, specs := range allSpecs {
		var arena *vk.VkTensorArena
		arena, err = vk.NewVkTensorArena(ctx, int(sizes[i]))
		if err != nil {
			return nil, err
		}
		s.resources = append(s.resources, arena)
		for _, spec := range specs {
			var t *vk.VkTensorF32
			t, err = arena.AllocF32(ctx, spec.shape...)
			if err != nil {
				return nil, err
			}
			tensors[spec.name] = t
			if spec.data != nil {
				err = t.Upload(ctx, spec.data)
			} else if spec.zero {
				err = t.Upload(ctx, make([]float32, t.Elements()))
			}
			if err != nil {
				return nil, err
			}
		}
	}
	s.input, s.output = tensors["mel"], tensors["h"]
	for _, steps := range layout.plans {
		stages := make([]vk.VkF32Stage, 0, len(steps))
		for _, step := range steps {
			var stage vk.VkF32Stage
			out := tensors[step.out]
			in := make([]*vk.VkTensorF32, len(step.in))
			for i, name := range step.in {
				in[i] = tensors[name]
			}
			switch step.op {
			case "conv":
				inputLayout := vk.VkConvTimeMajor
				if step.channelsFirst {
					inputLayout = vk.VkConvChannelsFirst
				}
				stage, err = s.conv.Stage(ctx, out, in[0], in[1], in[2], step.stride, inputLayout)
			case "add":
				stage, err = s.add.Stage(ctx, out, in[0], in[1])
			case "norm":
				stage, err = s.norm.Stage(ctx, out, in[0], in[1], in[2], 1e-5)
			case "linear":
				if index, quantized := linearIndexes[step.in[1]]; quantized && hybrid && strings.Contains(step.in[1], ".fc2.") {
					stage, err = s.q5Dot.F32Stage(ctx, index, out, in[0], in[2])
				} else if index, quantized := linearIndexes[step.in[1]]; quantized && dot {
					scratch := tensors["q8fc1"]
					if strings.Contains(step.in[1], "fc2") {
						scratch = tensors["q8fc2"]
					}
					pair, e := s.q5Dot.Stages(ctx, index, out, in[0], in[2], scratch)
					if e != nil {
						return nil, e
					}
					stages = append(stages, pair...)
					continue
				} else if index, quantized := linearIndexes[step.in[1]]; quantized && q5 {
					stage, err = s.q5Linear.Stage(ctx, index, out, in[0], in[2])
				} else if index, quantized := linearIndexes[step.in[1]]; quantized {
					stage, err = s.q8Linear.Stage(ctx, index, out, in[0], in[2])
				} else {
					stage, err = s.linear.Stage(ctx, out, in[0], in[1], in[2])
				}
			case "gelu":
				stage, err = s.gelu.Stage(ctx, out, in[0])
			case "attention":
				stage, err = s.attention.Stage(ctx, out, in[0], in[1], in[2], layout.cfg.EncoderHeads)
			default:
				return nil, fmt.Errorf("whisper Vulkan: unknown graph operator %q", step.op)
			}
			if err != nil {
				return nil, err
			}
			stages = append(stages, stage)
		}
		var p *vk.VkF32Plan
		p, err = makePlan(ctx, stages)
		// Adopt a partial result before checking the error, so injected/private
		// constructors cannot strand an owned plan during rollback.
		if p != nil {
			s.resources = append(s.resources, p)
		}
		if err != nil {
			return nil, err
		}
		if p == nil {
			return nil, fmt.Errorf("whisper Vulkan: plan constructor returned nil")
		}
		s.plans = append(s.plans, p)
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return e, nil
}
func (e *VulkanEncoder) acquire(ctx context.Context) (*vulkanEncoderState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("whisper Vulkan: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if e == nil || e.s == nil {
		return nil, fmt.Errorf("whisper Vulkan: uninitialised encoder")
	}
	s := e.s
	select {
	case s.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-s.gate
			return nil, err
		}
		return s, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Stats is a copied immutable description, not live native allocation accounting.
func (e *VulkanEncoder) Stats() VulkanEncoderStats {
	if e == nil || e.s == nil {
		return VulkanEncoderStats{}
	}
	return e.s.stats
}

// checkPCMConfig checks geometry/lifetime, not weight identity. The caller must
// pair this owned encoder with the decoder from the same checkpoint and exclude
// Close/other uses for the entire checked transcription call.
func (e *VulkanEncoder) checkPCMConfig(ctx context.Context, c Config) error {
	s, err := e.acquire(ctx)
	if err != nil {
		return err
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed {
		return vk.ErrVulkanClosed
	}
	v := s.config
	if s.stats.Frames != c.MaxLength || v.MaxLength != c.MaxLength || v.NumMelBins != c.NumMelBins || v.EncoderDModel != c.EncoderDModel || v.EncoderLayers != c.EncoderLayers || v.EncoderHeads != c.EncoderHeads || v.HeadDim != c.HeadDim || v.EncoderFFNDim != c.EncoderFFNDim {
		return fmt.Errorf("whisper Vulkan: checked PCM encoder geometry mismatch")
	}
	return nil
}
func (e *VulkanEncoder) Forward(ctx context.Context, mel []float32) ([]float32, error) {
	s, err := e.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed {
		return nil, vk.ErrVulkanClosed
	}
	if len(mel) != s.input.Elements() {
		return nil, fmt.Errorf("whisper Vulkan: mel length %d, want %d", len(mel), s.input.Elements())
	}
	for i, v := range mel {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("whisper Vulkan: nonfinite mel[%d]", i)
		}
	}
	if err := s.input.Upload(ctx, mel); err != nil {
		return nil, err
	}
	for i, p := range s.plans {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := p.Run(ctx); err != nil {
			return nil, fmt.Errorf("whisper Vulkan: plan%d: %w", i, err)
		}
	}
	out := make([]float32, s.output.Elements())
	if err := s.output.Download(ctx, out); err != nil {
		return nil, err
	}
	for i, v := range out {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("whisper Vulkan: nonfinite output[%d]", i)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// Close permanently blocks new Forward calls and attempts reverse teardown.
// Failed resources remain owned for a later Close retry. An unresolved native
// submission may require VulkanDrain; this method never drains or restarts it.
func (e *VulkanEncoder) Close() error {
	if e == nil || e.s == nil {
		return nil
	}
	s, err := e.acquire(context.Background())
	if err != nil {
		return err
	}
	defer func() { <-s.gate }()
	if s.closed {
		return nil
	}
	s.stopping = true
	var failures []error
	for i := len(s.resources) - 1; i >= 0; i-- {
		if c := s.resources[i]; c != nil {
			if err := c.Close(); err != nil {
				failures = append(failures, err)
			} else {
				s.resources[i] = nil
			}
		}
	}
	if len(failures) > 0 {
		return errors.Join(failures...)
	}
	s.closed = true
	s.resources = nil
	s.plans = nil
	s.tensors = nil
	s.input = nil
	s.output = nil
	return nil
}

func isVulkanMLPWeight(name string) bool {
	return strings.HasSuffix(name, ".fc1.w") || strings.HasSuffix(name, ".fc2.w")
}
