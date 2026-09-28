package qwen

import (
	"reflect"
	"testing"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func TestTreeAttentionSkipsSiblings(t *testing.T) {
	old := simd.HasDotAsm
	defer func() { simd.HasDotAsm = old }()
	for _, native := range []bool{true, false} {
		simd.HasDotAsm = old && native
		const prefix, start, end = 3, 6, 10
		q, k, v := make([]float32, 2048), make([]float32, end*512), make([]float32, end*512)
		for i := range q {
			q[i] = float32(i%17-8) * 0.04
		}
		for i := range k {
			k[i] = float32(i%13-6) * 0.03
			v[i] = float32(i%11-5) * 0.07
		}
		compactK := append(append([]float32{}, k[:prefix*512]...), k[start*512:]...)
		compactV := append(append([]float32{}, v[:prefix*512]...), v[start*512:]...)
		want, got := make([]float32, 2048), make([]float32, 2048)
		scores := make([]float32, prefix+end-start)
		if err := qwen35BranchAttentionInto(want, scores, q, compactK, compactV); err != nil {
			t.Fatal(err)
		}
		for repeat := 0; repeat < 2; repeat++ {
			for i := prefix * 512; i < start*512; i++ {
				k[i] = float32(1234 + repeat)
				v[i] = float32(-5678 - repeat)
			}
			if err := qwen35TreeAttentionInto(got, scores, q, k, v, prefix, start, end); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatal("sibling contamination")
			}
		}
		if a := testing.AllocsPerRun(10, func() {
			if err := qwen35TreeAttentionInto(got, scores, q, k, v, prefix, start, end); err != nil {
				panic(err)
			}
		}); a != 0 {
			t.Fatal("tree attention allocations", a)
		}
	}
}

func TestTreeForwardRejectsBoundaries(t *testing.T) {
	s := &Qwen35SIMDBranch{model: &Qwen35BaseModel{}, maxTokens: 8}
	inputs := make([][]float32, 5)
	for i := range inputs {
		inputs[i] = make([]float32, 1024)
	}
	dst, rope := make([]float32, 5*1024), make([]float32, 5*64)
	for i := range dst {
		dst[i] = 17
	}
	for _, ends := range [][]int{nil, {2}, {4}, {6}, {3, 3, 5}, {5, 4}, make([]int, 65)} {
		if err := s.ForwardTreeInto(dst, inputs, 1, 1, ends, rope, 1e-6); err == nil {
			t.Fatal("invalid boundaries", ends)
		}
		for _, v := range dst {
			if v != 17 {
				t.Fatal("partial result")
			}
		}
	}
}
