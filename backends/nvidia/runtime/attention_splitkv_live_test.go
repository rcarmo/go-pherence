package nvidia

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// Keep the production decode-attention dispatch on DevAttentionOK unless this
// opt-in candidate stays within current live parity tolerances and wins on real
// sequence lengths.
func TestAttentionSplitKVLiveParity(t *testing.T) {
	requireAttentionSplitKVLive(t)
	const (
		nHeads   = 6
		nKVHeads = 2
		headDim  = 96
	)
	candidate := NewAttentionSplitKVCandidate()
	defer candidate.Free()
	for _, seqLen := range []int{31, 32, 255, 256, 257, 512, 2048, 4096, 16384} {
		t.Run(fmt.Sprintf("seq_%d_heads_%d_kv_%d_dim_%d", seqLen, nHeads, nKVHeads, headDim), func(t *testing.T) {
			runAttentionSplitKVParityCase(t, candidate, seqLen, nHeads, nKVHeads, headDim)
		})
	}
}

func TestAttentionSplitKVLiveHeadDimTail(t *testing.T) {
	requireAttentionSplitKVLive(t)
	const (
		seqLen   = 257
		nHeads   = 4
		nKVHeads = 2
		headDim  = 320
	)
	candidate := NewAttentionSplitKVCandidate()
	defer candidate.Free()
	runAttentionSplitKVParityCase(t, candidate, seqLen, nHeads, nKVHeads, headDim)
}

func BenchmarkAttentionSplitKVCandidates(b *testing.B) {
	requireAttentionSplitKVKernels(b)
	const (
		nHeads   = 6
		nKVHeads = 2
		headDim  = 96
	)
	for _, seqLen := range []int{512, 2048, 4096, 16384} {
		b.Run(fmt.Sprintf("seq_%d_heads_%d_kv_%d_dim_%d", seqLen, nHeads, nKVHeads, headDim), func(b *testing.B) {
			q, k, v := makeAttentionSplitKVInputs(seqLen, nHeads, nKVHeads, headDim)
			qBuf := uploadAttentionSplitKVTensor(b, "q", q)
			kBuf := uploadAttentionSplitKVTensor(b, "k", k)
			vBuf := uploadAttentionSplitKVTensor(b, "v", v)
			outSplit := allocAttentionSplitKVTensor(b, "out_split", len(q))
			candidate := NewAttentionSplitKVCandidate()
			b.Cleanup(candidate.Free)
			scale := float32(1 / math.Sqrt(float64(headDim)))

			b.Run("split_kv_candidate", func(b *testing.B) {
				if !candidate.RunOK(outSplit, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
					b.Fatal("warmup split-kv candidate launch failed")
				}
				if err := SyncErr(); err != nil {
					b.Fatalf("warmup split-kv sync: %v", err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if !candidate.RunOK(outSplit, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
						b.Fatal("split-kv candidate launch failed")
					}
					SyncForTiming()
				}
			})

			if seqLen <= 2048 {
				outCurrent := allocAttentionSplitKVTensor(b, "out_current", len(q))
				b.Run("current_shared_scores", func(b *testing.B) {
					if !DevAttentionOK(outCurrent, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
						b.Fatal("warmup current attention launch failed")
					}
					if err := SyncErr(); err != nil {
						b.Fatalf("warmup current sync: %v", err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if !DevAttentionOK(outCurrent, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
							b.Fatal("current attention launch failed")
						}
						SyncForTiming()
					}
				})
			}
		})
	}
}

func requireAttentionSplitKVKernels(tb testing.TB) {
	tb.Helper()
	if !DevAttentionSplitKVReady() || attnFn == 0 {
		tb.Skip("split-KV decode-attention candidate kernels not available")
	}
}

func requireAttentionSplitKVLive(tb testing.TB) {
	tb.Helper()
	requireAttentionSplitKVKernels(tb)
	if _, ok := tb.(*testing.T); ok && os.Getenv("GO_PHERENCE_RUN_ATTENTION_SPLIT_KV") != "1" {
		tb.Skip("set GO_PHERENCE_RUN_ATTENTION_SPLIT_KV=1 to run split-KV decode-attention live parity tests")
	}
}

func runAttentionSplitKVParityCase(t *testing.T, candidate *AttentionSplitKVCandidate, seqLen, nHeads, nKVHeads, headDim int) {
	t.Helper()
	q, k, v := makeAttentionSplitKVInputs(seqLen, nHeads, nKVHeads, headDim)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	want := attentionCPU(q, k, v, seqLen, nHeads, nKVHeads, headDim, scale)
	split := runAttentionSplitKVCandidate(t, candidate, q, k, v, seqLen, nHeads, nKVHeads, headDim, scale)
	assertRuntimeClose(t, split, want, 3e-3)
	if seqLen <= 2048 {
		current := runCurrentAttentionKernel(t, q, k, v, seqLen, nHeads, nKVHeads, headDim, scale)
		assertRuntimeClose(t, current, want, 2e-3)
		assertRuntimeClose(t, split, current, 3e-3)
	}
}

func runAttentionSplitKVCandidate(tb testing.TB, candidate *AttentionSplitKVCandidate, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) []float32 {
	tb.Helper()
	qBuf := uploadAttentionSplitKVTensor(tb, "q", q)
	kBuf := uploadAttentionSplitKVTensor(tb, "k", k)
	vBuf := uploadAttentionSplitKVTensor(tb, "v", v)
	outBuf := allocAttentionSplitKVTensor(tb, "out_split", len(q))
	if !candidate.RunOK(outBuf, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
		tb.Fatal("split-KV candidate launch failed")
	}
	if err := SyncErr(); err != nil {
		tb.Fatalf("split-KV candidate sync: %v", err)
	}
	got := append([]float32(nil), outBuf.Data()[:len(q)]...)
	return got
}

func runCurrentAttentionKernel(tb testing.TB, q, k, v []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) []float32 {
	tb.Helper()
	qBuf := uploadAttentionSplitKVTensor(tb, "q", q)
	kBuf := uploadAttentionSplitKVTensor(tb, "k", k)
	vBuf := uploadAttentionSplitKVTensor(tb, "v", v)
	outBuf := allocAttentionSplitKVTensor(tb, "out_current", len(q))
	if !DevAttentionOK(outBuf, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
		tb.Fatal("current attention launch failed")
	}
	if err := SyncErr(); err != nil {
		tb.Fatalf("current attention sync: %v", err)
	}
	got := append([]float32(nil), outBuf.Data()[:len(q)]...)
	return got
}

func makeAttentionSplitKVInputs(seqLen, nHeads, nKVHeads, headDim int) ([]float32, []float32, []float32) {
	q := make([]float32, nHeads*headDim)
	kvDim := nKVHeads * headDim
	k := make([]float32, seqLen*kvDim)
	v := make([]float32, seqLen*kvDim)
	for i := range q {
		qv := float32((i*17)%41-20) * 0.03125
		if (i/5)&1 == 1 {
			qv = -qv
		}
		q[i] = qv
	}
	for i := range k {
		kv := float32((i*19)%43-21) * 0.02734375
		if (i/7)&1 == 1 {
			kv = -kv
		}
		vv := float32((i*23)%47-23) * 0.0234375
		if (i/3)&1 == 1 {
			vv = -vv
		}
		k[i] = kv
		v[i] = vv
	}
	return q, k, v
}

func uploadAttentionSplitKVTensor(tb testing.TB, name string, data []float32) *DevBuf {
	tb.Helper()
	buf := NewDevBufFrom(data)
	tb.Cleanup(buf.Free)
	if err := buf.ToGPU(); err != nil {
		tb.Fatalf("%s ToGPU: %v", name, err)
	}
	return buf
}

func allocAttentionSplitKVTensor(tb testing.TB, name string, n int) *DevBuf {
	tb.Helper()
	buf, err := NewDevBufGPU(n)
	if err != nil {
		tb.Fatalf("%s NewDevBufGPU: %v", name, err)
	}
	tb.Cleanup(buf.Free)
	return buf
}
