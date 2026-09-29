package nvidia

import (
	"fmt"
	"math"
	"testing"
)

// Keep the default attention_full dispatch honest against the online candidate.
// Do not switch production dispatch unless the candidate stays within the
// current 2e-3 live parity tolerance and is measurably faster.
func TestWhisperAttentionFullOnlineLiveParity(t *testing.T) {
	requireWhisperAttentionFullKernels(t)
	const (
		heads   = 16
		headDim = 64
	)
	for _, seq := range []int{375, 1500} {
		t.Run(fmt.Sprintf("seq_%d_heads_%d_dim_%d", seq, heads, headDim), func(t *testing.T) {
			current, online := runWhisperAttentionFullPair(t, seq, heads, headDim)
			assertRuntimeClose(t, online, current, 2e-3)
		})
	}
}

// The 103-key shape previously raced when one warp reused reduce_shared[0]
// before another had read the shared maximum. Reuse identical resident buffers
// across launches and compare every value with an independent F64 softmax.
func TestWhisperAttentionFullOnlineRepeated103Keys(t *testing.T) {
	requireWhisperAttentionFullKernels(t)
	const seq, heads, dim = 103, 8, 64
	q, k, v := makeWhisperAttentionInputs(seq, heads, dim)
	qBuf := uploadWhisperAttentionTensor(t, "q", q)
	kBuf := uploadWhisperAttentionTensor(t, "k", k)
	vBuf := uploadWhisperAttentionTensor(t, "v", v)
	out := allocWhisperAttentionTensor(t, "out", len(q))
	scale := float32(1 / math.Sqrt(dim))
	want := make([]float32, len(q))
	for row := 0; row < seq; row++ {
		for head := 0; head < heads; head++ {
			scores := make([]float64, seq)
			maxScore := math.Inf(-1)
			for key := 0; key < seq; key++ {
				var dot float64
				for col := 0; col < dim; col++ {
					dot += float64(q[(row*heads+head)*dim+col]) * float64(k[(key*heads+head)*dim+col])
				}
				scores[key] = dot * float64(scale)
				maxScore = math.Max(maxScore, scores[key])
			}
			var sum float64
			for key := range scores {
				scores[key] = math.Exp(scores[key] - maxScore)
				sum += scores[key]
			}
			for col := 0; col < dim; col++ {
				var value float64
				for key := 0; key < seq; key++ {
					value += scores[key] * float64(v[(key*heads+head)*dim+col])
				}
				want[(row*heads+head)*dim+col] = float32(value / sum)
			}
		}
	}
	got := make([]float32, len(q))
	for trial := 0; trial < 20; trial++ {
		if err := WhisperAttentionFullOnlineBuffer(out, qBuf, kBuf, vBuf, seq, seq, heads, dim, scale); err != nil {
			t.Fatal(err)
		}
		if err := SyncErr(); err != nil {
			t.Fatal(err)
		}
		if err := out.Download(got); err != nil {
			t.Fatal(err)
		}
		var maxAbs float64
		var outside int
		for i, value := range got {
			delta := math.Abs(float64(value - want[i]))
			maxAbs = math.Max(maxAbs, delta)
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 2e-5 {
				outside++
			}
		}
		t.Logf("trial=%d max_abs=%g outside=%d", trial, maxAbs, outside)
		if outside != 0 {
			t.Fatal("online attention differs from F64 softmax")
		}
	}
}

func BenchmarkWhisperAttentionFullCandidates(b *testing.B) {
	requireWhisperAttentionFullKernels(b)
	const (
		heads   = 16
		headDim = 64
	)
	for _, seq := range []int{375, 1500} {
		b.Run(fmt.Sprintf("seq_%d_heads_%d_dim_%d", seq, heads, headDim), func(b *testing.B) {
			q, k, v := makeWhisperAttentionInputs(seq, heads, headDim)
			qBuf := uploadWhisperAttentionTensor(b, "q", q)
			kBuf := uploadWhisperAttentionTensor(b, "k", k)
			vBuf := uploadWhisperAttentionTensor(b, "v", v)
			outCurrent := allocWhisperAttentionTensor(b, "out_current", len(q))
			outOnline := allocWhisperAttentionTensor(b, "out_online", len(q))
			scale := float32(1 / math.Sqrt(float64(headDim)))
			for _, tc := range []struct {
				name   string
				out    *Buffer
				launch func(*Buffer, *Buffer, *Buffer, *Buffer, int, int, int, int, float32) error
			}{
				{name: "current_shared_scores", out: outCurrent, launch: WhisperAttentionFullBuffer},
				{name: "online_candidate", out: outOnline, launch: WhisperAttentionFullOnlineBuffer},
			} {
				b.Run(tc.name, func(b *testing.B) {
					if err := tc.launch(tc.out, qBuf, kBuf, vBuf, seq, seq, heads, headDim, scale); err != nil {
						b.Fatalf("warmup launch: %v", err)
					}
					if err := SyncErr(); err != nil {
						b.Fatalf("warmup sync: %v", err)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if err := tc.launch(tc.out, qBuf, kBuf, vBuf, seq, seq, heads, headDim, scale); err != nil {
							b.Fatal(err)
						}
						SyncForTiming()
					}
				})
			}
		})
	}
}

func requireWhisperAttentionFullKernels(tb testing.TB) {
	tb.Helper()
	if !SgemmReady() || fnWhisperAttentionFull == 0 || fnWhisperAttentionFullOnline == 0 {
		tb.Skip("Whisper full attention kernels not available")
	}
}

func runWhisperAttentionFullPair(tb testing.TB, seq, heads, headDim int) ([]float32, []float32) {
	tb.Helper()
	q, k, v := makeWhisperAttentionInputs(seq, heads, headDim)
	qBuf := uploadWhisperAttentionTensor(tb, "q", q)
	kBuf := uploadWhisperAttentionTensor(tb, "k", k)
	vBuf := uploadWhisperAttentionTensor(tb, "v", v)
	outCurrent := allocWhisperAttentionTensor(tb, "out_current", len(q))
	outOnline := allocWhisperAttentionTensor(tb, "out_online", len(q))
	scale := float32(1 / math.Sqrt(float64(headDim)))
	if err := WhisperAttentionFullBuffer(outCurrent, qBuf, kBuf, vBuf, seq, seq, heads, headDim, scale); err != nil {
		tb.Fatalf("attention_full: %v", err)
	}
	if err := WhisperAttentionFullOnlineBuffer(outOnline, qBuf, kBuf, vBuf, seq, seq, heads, headDim, scale); err != nil {
		tb.Fatalf("attention_full_online: %v", err)
	}
	if err := SyncErr(); err != nil {
		tb.Fatalf("sync: %v", err)
	}
	current := make([]float32, len(q))
	online := make([]float32, len(q))
	if err := outCurrent.Download(current); err != nil {
		tb.Fatalf("download current: %v", err)
	}
	if err := outOnline.Download(online); err != nil {
		tb.Fatalf("download online: %v", err)
	}
	return current, online
}

func makeWhisperAttentionInputs(seq, heads, headDim int) ([]float32, []float32, []float32) {
	n := seq * heads * headDim
	q := make([]float32, n)
	k := make([]float32, n)
	v := make([]float32, n)
	for i := 0; i < n; i++ {
		qv := float32((i*17)%37-18) * 0.03125
		if (i/7)&1 == 1 {
			qv = -qv
		}
		kv := float32((i*19)%41-20) * 0.02734375
		if (i/5)&1 == 1 {
			kv = -kv
		}
		vv := float32((i*23)%43-21) * 0.0234375
		if (i/3)&1 == 1 {
			vv = -vv
		}
		q[i] = qv
		k[i] = kv
		v[i] = vv
	}
	return q, k, v
}

func uploadWhisperAttentionTensor(tb testing.TB, name string, data []float32) *Buffer {
	tb.Helper()
	buf, err := Malloc(len(data))
	if err != nil {
		tb.Fatalf("%s malloc: %v", name, err)
	}
	tb.Cleanup(buf.Free)
	if err := buf.Upload(data); err != nil {
		tb.Fatalf("%s upload: %v", name, err)
	}
	return buf
}

func allocWhisperAttentionTensor(tb testing.TB, name string, n int) *Buffer {
	tb.Helper()
	buf, err := Malloc(n)
	if err != nil {
		tb.Fatalf("%s malloc: %v", name, err)
	}
	tb.Cleanup(buf.Free)
	return buf
}
