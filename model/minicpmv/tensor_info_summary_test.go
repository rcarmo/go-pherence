package minicpmv

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestSummarizeTensorInfos(t *testing.T) {
	s := SummarizeTensorInfos(map[string]safetensors.TensorInfo{
		"a": {DType: "F32", Shape: []int{2, 3}},
		"b": {DType: "F16", Shape: []int{4}},
		"c": {DType: "F32", Shape: []int{1, 2, 3}},
	})
	if s.Total != 3 || s.DTypes["F32"] != 2 || s.DTypes["F16"] != 1 || s.Ranks[2] != 1 || s.Ranks[1] != 1 || s.Ranks[3] != 1 || s.TotalElements != 16 {
		t.Fatalf("bad tensor info summary: %+v", s)
	}
}
