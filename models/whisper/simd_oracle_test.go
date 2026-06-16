package whisper

import (
	"math"
	"os"
	"testing"
)

func TestWhisperConv1DFastMatchesScalarOracle(t *testing.T) {
	for _, stride := range []int{1, 2} {
		stride := stride
		t.Run("stride", func(t *testing.T) {
			inCh, inLen, outCh := 5, 13, 7
			input := make([]float32, inCh*inLen)
			weight := make([]float32, outCh*inCh*3)
			bias := make([]float32, outCh)
			for i := range input {
				input[i] = float32((i%17)-8) * 0.041
			}
			for i := range weight {
				weight[i] = float32((i%19)-9) * 0.023
			}
			for i := range bias {
				bias[i] = float32((i%7)-3) * 0.037
			}
			want := conv1dForward(input, weight, bias, inCh, inLen, outCh, 3, stride, 1)
			got := conv1dForwardFast(input, weight, bias, inCh, inLen, outCh, 3, stride, 1)
			assertClose(t, got, want, 1e-5)
		})
	}
}

func TestWhisperLayerNormUsesSIMDOracleMatchesScalar(t *testing.T) {
	dim := 1280
	x := make([]float32, dim)
	weight := make([]float32, dim)
	bias := make([]float32, dim)
	for i := 0; i < dim; i++ {
		x[i] = float32((i%29)-14) * 0.017
		weight[i] = 0.75 + float32(i%11)*0.013
		bias[i] = float32((i%7)-3) * 0.021
	}
	got := make([]float32, dim)
	want := make([]float32, dim)
	layerNormInto(got, x, weight, bias, dim)
	layerNormScalarOracleInto(want, x, weight, bias, dim, 1e-5)
	assertClose(t, got, want, 2e-5)
}

func TestWhisperFullAttentionMatchesScalarOracle(t *testing.T) {
	oldWorkers := linearWorkers
	oldInt8 := attnInt8
	oldF16 := attnF16
	oldHeadBatch := os.Getenv("WHISPER_FP16_HEAD_BATCH")
	linearWorkers = 2
	attnInt8 = false
	attnF16 = false
	_ = os.Unsetenv("WHISPER_FP16_HEAD_BATCH")
	defer func() {
		linearWorkers = oldWorkers
		attnInt8 = oldInt8
		attnF16 = oldF16
		_ = os.Setenv("WHISPER_FP16_HEAD_BATCH", oldHeadBatch)
	}()

	seqQ, seqKV, numHeads, headDim := 7, 9, 3, 5
	q, k, v := deterministicQKV(seqQ, seqKV, numHeads, headDim)
	got := fullAttention(q, k, v, seqQ, seqKV, numHeads, headDim)
	want := fullAttentionScalarOracle(q, k, v, seqQ, seqKV, numHeads, headDim)
	assertClose(t, got, want, 2e-5)
}

func fullAttentionScalarOracle(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, seqQ*dModel)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := make([]float32, seqKV)
	for tq := 0; tq < seqQ; tq++ {
		for h := 0; h < numHeads; h++ {
			hOff := h * headDim
			for tk := 0; tk < seqKV; tk++ {
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[tq*dModel+hOff+d] * k[tk*dModel+hOff+d]
				}
				scores[tk] = dot * scale
			}
			softmax(scores)
			for d := 0; d < headDim; d++ {
				var sum float32
				for tk := 0; tk < seqKV; tk++ {
					sum += scores[tk] * v[tk*dModel+hOff+d]
				}
				out[tq*dModel+hOff+d] = sum
			}
		}
	}
	return out
}
