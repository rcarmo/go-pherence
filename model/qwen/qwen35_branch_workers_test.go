package qwen

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"testing"
)

func TestBranchAttentionInto(t *testing.T) {
	oldDot, oldVec := simd.HasDotAsm, simd.HasVecAsm
	defer func() { simd.HasDotAsm, simd.HasVecAsm = oldDot, oldVec }()
	for _, native := range []bool{true, false} {
		simd.HasDotAsm, simd.HasVecAsm = oldDot && native, oldVec && native
		t.Run(fmt.Sprint("assembly_", native), testBranchAttentionInto)
	}
}

func testBranchAttentionInto(t *testing.T) {
	for _, n := range []int{1, 2, 7, 31, 128} {
		q, k, v := make([]float32, 2048), make([]float32, n*512), make([]float32, n*512)
		for i := range q {
			q[i] = float32(i%19-9) * 0.017
		}
		for i := range k {
			k[i] = float32(i%11-5) * 0.013
			v[i] = float32(i%17-8) * 0.023
		}
		want := qwenMTPGroupedAttention(q, k, v, 8, 2, 256)
		out, scores := make([]float32, 2048), make([]float32, n)
		for range 2 {
			if err := qwen35BranchAttentionInto(out, scores, q, k, v); err != nil {
				t.Fatal(err)
			}
			for i := range out {
				if math.Abs(float64(out[i]-want[i])) > 1e-6 {
					t.Fatal("attention mismatch", n, i, out[i], want[i])
				}
			}
		}
		if a := testing.AllocsPerRun(10, func() {
			if err := qwen35BranchAttentionInto(out, scores, q, k, v); err != nil {
				panic(err)
			}
		}); a != 0 {
			t.Fatal("attention allocations", a)
		}
	}
	if err := qwen35BranchAttentionInto(nil, nil, nil, nil, nil); err == nil {
		t.Fatal("invalid buffers")
	}
}

func TestBranchProjectionWorkers(t *testing.T) {
	for range 3 {
		s := &Qwen35SIMDBranch{}
		stop := s.startProjectionWorkers()
		if s.jobs != nil { // Exercise failed dispatch and wait before teardown.
			s.jobWait.Add(1)
			s.jobs <- branchProjectionJob{rows: 1, cols: 16, in: 1, stride: 16}
			s.jobWait.Wait()
			if !s.jobFailed[0] {
				t.Fatal("invalid job accepted")
			}
		}
		stop()
		if s.jobs != nil {
			t.Fatal("workers retained")
		}
	}
}

func TestBranchIntoRejectsAliases(t *testing.T) {
	s := &Qwen35SIMDBranch{maxTokens: 3, model: &Qwen35BaseModel{}}
	dst, rope := make([]float32, 3*1024), make([]float32, 3*64)
	inputs := [][]float32{dst[:1024], make([]float32, 1024), make([]float32, 1024)}
	if err := s.ForwardInto(dst, inputs, 1, 1, rope, 1e-6); err == nil {
		t.Fatal("input alias accepted")
	}
	inputs[0] = make([]float32, 1024)
	if err := s.ForwardInto(dst, inputs, 1, 1, dst[:3*64], 1e-6); err == nil {
		t.Fatal("RoPE alias accepted")
	}
	if branchSlicesOverlap(nil, dst) || branchSlicesOverlap(dst, nil) {
		t.Fatal("empty overlap")
	}
}
