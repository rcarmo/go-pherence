package pockettts

import (
	"context"
	"errors"
	"math"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

type namedAttentionBackend struct {
	attentionBackendFunc
	name string
}

func (b namedAttentionBackend) TrainingAttentionBackendName() string { return b.name }

type attentionBackendFunc func(q, k, v, p, dy []float32, s, h, d, c int) ([]float32, []float32, []float32, error)

func (f attentionBackendFunc) AttentionBackward(q, k, v, p, dy []float32, s, h, d, c int) ([]float32, []float32, []float32, error) {
	return f(q, k, v, p, dy, s, h, d, c)
}

// Independent F64 softmax Jacobian-vector product and QKV accumulation. It
// consumes the recorded probabilities, but does not reuse GPU/CPU backward.
func referenceAttentionBackend(q, k, v, p, dy []float32, s, h, d, contextRows int) ([]float32, []float32, []float32, error) {
	width := h * d
	dq, dk, dv := make([]float64, len(q)), make([]float64, len(k)), make([]float64, len(v))
	scale := 1 / math.Sqrt(float64(d))
	for head := 0; head < h; head++ {
		for query := 0; query < s; query++ {
			start := 0
			if contextRows > 0 {
				start = max(0, query-contextRows+1)
			}
			dp := make([]float64, query-start+1)
			sum := float64(0)
			for key := start; key <= query; key++ {
				for col := 0; col < d; col++ {
					dp[key-start] += float64(dy[query*width+head*d+col]) * float64(v[key*width+head*d+col])
				}
				sum += dp[key-start] * float64(p[(head*s+query)*s+key])
			}
			for key := start; key <= query; key++ {
				prob := float64(p[(head*s+query)*s+key])
				ds := prob * (dp[key-start] - sum) * scale
				for col := 0; col < d; col++ {
					qi, ki := query*width+head*d+col, key*width+head*d+col
					dq[qi] += ds * float64(k[ki])
					dk[ki] += ds * float64(q[qi])
					dv[ki] += prob * float64(dy[qi])
				}
			}
		}
	}
	convert := func(values []float64) []float32 {
		out := make([]float32, len(values))
		for i, v := range values {
			out[i] = float32(v)
		}
		return out
	}
	return convert(dq), convert(dk), convert(dv), nil
}
func TestTrainingAttentionHybridModelSeam(t *testing.T) {
	model := tinyTrainableTransformer()
	sequence := []float32{.2, -.4, .1, .5, -.3, .7, .6, -.2, .8, -.1, .4, -.5}
	seed := []float32{.1, -.2, .3, -.4, .5, -.6, .7, -.8, -.3, .4, -.5, .6}
	for _, c := range []int{0, 1, 2, 8} {
		model.Context = c
		out, cpu, dx, err := model.ForwardBackward(sequence, seed, 3)
		if err != nil {
			t.Fatal(err)
		}
		calls := 0
		backend := attentionBackendFunc(func(q, k, v, p, dy []float32, s, h, d, c int) ([]float32, []float32, []float32, error) {
			calls++
			return referenceAttentionBackend(q, k, v, p, dy, s, h, d, c)
		})
		hybrid, grad, ddx, err := model.ForwardBackwardAttention(sequence, seed, 3, backend)
		if err != nil || calls != 1 {
			t.Fatal(err, calls)
		}
		assertSliceClose(t, "hybrid forward", hybrid, out, 0)
		assertSliceClose(t, "hybrid dSequence", ddx, dx, 3e-5)
		_, a := transformerParameterPairs(model, cpu)
		_, b := transformerParameterPairs(model, grad)
		for i := range a {
			assertSliceClose(t, a[i].name, b[i].values, a[i].values, 3e-5)
		}
	}
}
func TestTrainingAttentionBackendFailure(t *testing.T) {
	model := tinyFlowLMTraining()
	batch := tinyFlowLMBatch()
	sentinel := errors.New("device failed")
	model.AttentionBackwardBackend = attentionBackendFunc(func(q, k, v, p, dy []float32, s, h, d, c int) ([]float32, []float32, []float32, error) {
		return nil, nil, nil, sentinel
	})
	_, _, _, err := model.ForwardBackward(batch, make([]float32, 12), make([]float32, 3))
	if !errors.Is(err, sentinel) {
		t.Fatal("backend failure lost", err)
	}
	model.AttentionBackwardBackend = attentionBackendFunc(func(q, k, v, p, dy []float32, s, h, d, c int) ([]float32, []float32, []float32, error) {
		return q, append([]float32(nil), k...), append([]float32(nil), v...), nil
	})
	if _, _, _, err := model.ForwardBackward(batch, make([]float32, 12), make([]float32, 3)); err == nil {
		t.Fatal("borrowed output admitted")
	}
	if validateAttentionDerivatives([]float32{0}, []float32{0}, []float32{float32(math.NaN())}, 1) == nil {
		t.Fatal("nonfinite output admitted")
	}
	shared := []float32{0}
	if validateAttentionDerivatives(shared, shared, []float32{0}, 1) == nil {
		t.Fatal("overlapping output admitted")
	}
}
func TestTrainingAttentionAdaptersRejectBeforeNative(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, backend := range []TrainingAttentionBackend{
		&PTXTrainingAttention{Context: ctx, Operator: &nvidia.TrainingF32{}, MaxBytes: 1 << 20},
		&VulkanTrainingAttention{Context: ctx, Operator: &vulkan.VkTrainingF32{}, MaxBytes: 1 << 20},
	} {
		if _, _, _, err := backend.AttentionBackward(nil, nil, nil, nil, nil, 1, 1, 1, 0); !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	}
	for _, backend := range []TrainingAttentionBackend{
		&PTXTrainingAttention{Context: context.Background(), Operator: &nvidia.TrainingF32{}, MaxBytes: 1 << 20},
		&VulkanTrainingAttention{Context: context.Background(), Operator: &vulkan.VkTrainingF32{}, MaxBytes: 1 << 20},
	} {
		if _, _, _, err := backend.AttentionBackward([]float32{1}, []float32{1}, []float32{1}, []float32{1}, []float32{1}, 1, 1, 1, 0); err == nil {
			t.Fatal("zero native operator admitted")
		}
	}
	for _, backend := range []TrainingAttentionBackend{
		&PTXTrainingAttention{Context: context.Background(), Operator: &nvidia.TrainingF32{}, MaxBytes: 1},
		&VulkanTrainingAttention{Context: context.Background(), Operator: &vulkan.VkTrainingF32{}, MaxBytes: 1},
	} {
		if _, _, _, err := backend.AttentionBackward([]float32{1}, []float32{1}, []float32{1}, []float32{1}, []float32{1}, 1, 1, 1, 0); err == nil {
			t.Fatal("budget admitted before native")
		}
		if _, _, _, err := backend.AttentionBackward(nil, nil, nil, nil, nil, 0, 1, 1, 0); err == nil {
			t.Fatal("shape admitted before native")
		}
	}
}
