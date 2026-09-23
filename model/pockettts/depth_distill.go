package pockettts

import (
	"fmt"
	"math"
	"strings"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// SelectDistillationLayers matches pinned upstream's champion "ends" seed:
// the larger half from the bottom and the remainder from the top.
func SelectDistillationLayers(teacher, student int) ([]int, error) {
	if teacher <= 0 || student <= 0 || student > teacher {
		return nil, fmt.Errorf("invalid Pocket TTS distillation depths %d -> %d", teacher, student)
	}
	head := (student + 1) / 2
	tail := student - head
	out := make([]int, 0, student)
	for i := 0; i < head; i++ {
		out = append(out, i)
	}
	for i := teacher - tail; i < teacher; i++ {
		out = append(out, i)
	}
	return out, nil
}

// ShrinkFlowLMState remaps transformer layer names and copies every non-layer
// tensor unchanged. Returned values are owned so teacher state cannot alias the
// student seed. Both prefixed and unprefixed upstream keys are accepted.
func ShrinkFlowLMState(state map[string][]float32, student int) (map[string][]float32, []int, error) {
	if len(state) == 0 {
		return nil, nil, fmt.Errorf("Pocket TTS distillation state is empty")
	}
	depth := -1
	for name := range state {
		if index, _, ok := distillLayerName(name); ok && index > depth {
			depth = index
		}
	}
	depth++
	keep, err := SelectDistillationLayers(depth, student)
	if err != nil {
		return nil, nil, err
	}
	remap := map[int]int{}
	for to, from := range keep {
		remap[from] = to
	}
	out := make(map[string][]float32, len(state))
	for name, values := range state {
		if name == "" || values == nil || !finiteF32(values) {
			return nil, nil, fmt.Errorf("invalid Pocket TTS distillation tensor %q", name)
		}
		newName := name
		if index, suffix, ok := distillLayerName(name); ok {
			to, keepLayer := remap[index]
			if !keepLayer {
				continue
			}
			prefix := ""
			if strings.HasPrefix(name, "flow_lm.") {
				prefix = "flow_lm."
			}
			newName = fmt.Sprintf("%stransformer.layers.%d.%s", prefix, to, suffix)
		}
		if _, duplicate := out[newName]; duplicate {
			return nil, nil, fmt.Errorf("duplicate Pocket TTS distillation tensor %q", newName)
		}
		out[newName] = append([]float32(nil), values...)
	}
	return out, keep, nil
}

func distillLayerName(name string) (int, string, bool) {
	rest := name
	if strings.HasPrefix(rest, "flow_lm.") {
		rest = strings.TrimPrefix(rest, "flow_lm.")
	}
	const prefix = "transformer.layers."
	if !strings.HasPrefix(rest, prefix) {
		return 0, "", false
	}
	rest = strings.TrimPrefix(rest, prefix)
	dot := strings.IndexByte(rest, '.')
	if dot <= 0 || dot == len(rest)-1 {
		return 0, "", false
	}
	index := 0
	for _, c := range []byte(rest[:dot]) {
		if c < '0' || c > '9' {
			return 0, "", false
		}
		if index > (math.MaxInt-int(c-'0'))/10 {
			return 0, "", false
		}
		index = index*10 + int(c-'0')
	}
	return index, rest[dot+1:], true
}

// DepthDistillResult is the exact one-row upstream backbone-regression output.
type DepthDistillResult struct {
	Loss                                             float64
	Student, TeacherConditioned, TeacherNull, Target []float32
	ShiftedMask                                      []bool
}

type DepthDistillGradients struct {
	Student *FlowLMTrainingGradients
	Inputs  FlowLMTrainingInputGradients
}

// DepthDistillForwardBackward runs a fully conditioned student and frozen
// conditioned/null teacher. The teacher target is null+cfg*(conditioned-null).
// Only student conditioning/backbone parameters receive gradients; EOS, flow
// head, w_s_t, teacher and target are outside this boundary.
func DepthDistillForwardBackward(student, teacher *FlowLMTrainingCPU, batch FlowLMTrainingBatch, mask []bool, cfgCoef float32) (DepthDistillResult, DepthDistillGradients, error) {
	var result DepthDistillResult
	var gradients DepthDistillGradients
	if student == nil || teacher == nil || batch.Frames <= 0 || len(mask) != batch.Frames || !isFinite(cfgCoef) || cfgCoef <= 0 || student.Hidden != teacher.Hidden || student.LatentDim != teacher.LatentDim {
		return result, gradients, fmt.Errorf("invalid Pocket TTS depth distillation boundary")
	}
	h := student.Hidden
	zElements, ok := checked.MulInt(batch.Frames, h)
	if !ok {
		return result, gradients, fmt.Errorf("Pocket TTS depth distillation shape overflows")
	}
	zeroZ := make([]float32, zElements)
	zeroEOS := make([]float32, batch.Frames)
	studentTape, err := student.forwardTrainingTape(batch, zeroZ, zeroEOS)
	if err != nil {
		return result, gradients, err
	}
	teacherConditioned, err := teacher.forwardTrainingTape(batch, zeroZ, zeroEOS)
	if err != nil {
		return result, gradients, err
	}
	nullBatch := batch
	nullBatch.VoiceFrames = 0
	nullBatch.VoiceLatents = nil
	nullBatch.TextTokens = nil
	teacherNull, err := teacher.forwardTrainingTapeMode(nullBatch, zeroZ, zeroEOS, true)
	if err != nil {
		return result, gradients, err
	}
	shifted := make([]bool, batch.Frames)
	shifted[0] = mask[0]
	copy(shifted[1:], mask[:len(mask)-1])
	active := 0
	for _, v := range shifted {
		if v {
			active++
		}
	}
	if active == 0 {
		return result, gradients, fmt.Errorf("Pocket TTS depth distillation mask has no active rows")
	}
	target := make([]float32, zElements)
	dZ := make([]float32, zElements)
	denom := float32(active * h)
	loss := float64(0)
	for row, on := range shifted {
		for i := 0; i < h; i++ {
			index := row*h + i
			target[index] = teacherNull.output.Z[index] + cfgCoef*(teacherConditioned.output.Z[index]-teacherNull.output.Z[index])
			if on {
				residual := studentTape.output.Z[index] - target[index]
				loss += float64(residual*residual) / float64(h*active)
				dZ[index] = 2 * residual / denom
			}
		}
	}
	studentGradients, inputGradients, err := student.backwardTrainingTape(studentTape, batch, dZ, zeroEOS)
	if err != nil {
		return result, gradients, err
	}
	result = DepthDistillResult{Loss: loss, Student: append([]float32(nil), studentTape.output.Z...), TeacherConditioned: append([]float32(nil), teacherConditioned.output.Z...), TeacherNull: append([]float32(nil), teacherNull.output.Z...), Target: target, ShiftedMask: shifted}
	gradients = DepthDistillGradients{Student: studentGradients, Inputs: inputGradients}
	return result, gradients, nil
}
