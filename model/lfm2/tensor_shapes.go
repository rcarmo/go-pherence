package lfm2

import (
	"sort"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type TensorShapeSummary struct {
	Total    int                    `json:"total"`
	ByGroup  map[string]int         `json:"by_group"`
	DTypes   map[string]int         `json:"dtypes"`
	Examples map[string]TensorShape `json:"examples,omitempty"`
}

type TensorShape struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}

func InspectTensorShapes(infos map[string]safetensors.TensorInfo) TensorShapeSummary {
	s := TensorShapeSummary{Total: len(infos), ByGroup: map[string]int{}, DTypes: map[string]int{}, Examples: map[string]TensorShape{}}
	names := make([]string, 0, len(infos))
	for name := range infos {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		info := infos[name]
		group := TensorGroup(name)
		s.ByGroup[group]++
		s.DTypes[info.DType]++
		if _, ok := s.Examples[group]; !ok {
			s.Examples[group] = TensorShape{Name: name, DType: info.DType, Shape: append([]int(nil), info.Shape...)}
		}
	}
	if len(s.Examples) == 0 {
		s.Examples = nil
	}
	return s
}
