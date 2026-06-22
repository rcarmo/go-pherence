package minicpmv

import "github.com/rcarmo/go-pherence/loader/safetensors"

type TensorInfoSummary struct {
	Total         int            `json:"total"`
	DTypes        map[string]int `json:"dtypes,omitempty"`
	Ranks         map[int]int    `json:"ranks,omitempty"`
	TotalElements int64          `json:"total_elements,omitempty"`
}

func SummarizeTensorInfos(infos map[string]safetensors.TensorInfo) TensorInfoSummary {
	out := TensorInfoSummary{Total: len(infos), DTypes: map[string]int{}, Ranks: map[int]int{}}
	for _, info := range infos {
		out.DTypes[info.DType]++
		out.Ranks[len(info.Shape)]++
		if n := tensorNumel(info.Shape); n > 0 {
			out.TotalElements += int64(n)
		}
	}
	return out
}

func tensorNumel(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}
