package whisper

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
)

func TestWhisperFullAttentionPackedQueryBatchedMatchesCurrentExact(t *testing.T) {
	oldWorkers := linearWorkers
	oldInt8 := attnInt8
	oldF16 := attnF16
	oldHeadBatch := os.Getenv("WHISPER_FP16_HEAD_BATCH")
	linearWorkers = 1
	attnInt8 = false
	attnF16 = false
	_ = os.Unsetenv("WHISPER_FP16_HEAD_BATCH")
	defer func() {
		linearWorkers = oldWorkers
		attnInt8 = oldInt8
		attnF16 = oldF16
		_ = os.Setenv("WHISPER_FP16_HEAD_BATCH", oldHeadBatch)
	}()

	cases := []struct {
		name       string
		seqQ       int
		seqKV      int
		numHeads   int
		headDim    int
		workers    int
		queryBatch int
	}{
		{name: "single_worker", seqQ: 137, seqKV: 149, numHeads: 3, headDim: 16, workers: 1, queryBatch: 64},
		{name: "multi_worker_medium", seqQ: 257, seqKV: 263, numHeads: 4, headDim: 64, workers: 4, queryBatch: fullAttentionQueryBatchDefault},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			linearWorkers = tc.workers
			q, k, v := deterministicQKV(tc.seqQ, tc.seqKV, tc.numHeads, tc.headDim)
			want := fullAttentionPerHead(q, k, v, tc.seqQ, tc.seqKV, tc.numHeads, tc.headDim)
			got := fullAttentionPackedQueryBatched(q, k, v, tc.seqQ, tc.seqKV, tc.numHeads, tc.headDim, tc.queryBatch)
			assertExactFloat32Slice(t, got, want)
		})
	}
}

func BenchmarkWhisperFullAttentionMedium(b *testing.B) {
	oldWorkers := linearWorkers
	oldInt8 := attnInt8
	oldF16 := attnF16
	oldHeadBatch := os.Getenv("WHISPER_FP16_HEAD_BATCH")
	attnInt8 = false
	attnF16 = false
	_ = os.Unsetenv("WHISPER_FP16_HEAD_BATCH")
	defer func() {
		linearWorkers = oldWorkers
		attnInt8 = oldInt8
		attnF16 = oldF16
		_ = os.Setenv("WHISPER_FP16_HEAD_BATCH", oldHeadBatch)
	}()

	impls := []struct {
		name string
		fn   func(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32
	}{
		{name: "current", fn: fullAttentionPerHead},
	}
	for _, queryBatch := range []int{64, 96, 128, 192, 256} {
		queryBatch := queryBatch
		impls = append(impls, struct {
			name string
			fn   func(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32
		}{
			name: fmt.Sprintf("packed_query_batch_%d", queryBatch),
			fn: func(q, k, v []float32, seqQ, seqKV, numHeads, headDim int) []float32 {
				return fullAttentionPackedQueryBatched(q, k, v, seqQ, seqKV, numHeads, headDim, queryBatch)
			},
		})
	}
	workers := []int{1, 2, 4}
	if maxWorkers := runtime.GOMAXPROCS(0); maxWorkers < 4 {
		workers = workers[:maxWorkers]
	}
	for _, seq := range []int{375, 1500} {
		q, k, v := deterministicQKV(seq, seq, 16, 64)
		for _, nw := range workers {
			nw := nw
			for _, impl := range impls {
				impl := impl
				b.Run(fmt.Sprintf("seq_%d/workers_%d/%s", seq, nw, impl.name), func(b *testing.B) {
					linearWorkers = nw
					b.ReportAllocs()
					var out []float32
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						out = impl.fn(q, k, v, seq, seq, 16, 64)
					}
					b.StopTimer()
					if len(out) != seq*16*64 {
						b.Fatalf("unexpected output len=%d", len(out))
					}
				})
			}
		}
	}
}

func assertExactFloat32Slice(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got=%d want=%d", len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("bits mismatch at %d: got=%08x want=%08x (got=%g want=%g)", i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
		}
	}
}
