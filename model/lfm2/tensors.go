package lfm2

import (
	"sort"
	"strings"
)

type TensorCoverage struct {
	Total     int               `json:"total"`
	Embedding int               `json:"embedding"`
	Layers    int               `json:"layers"`
	Router    int               `json:"router"`
	Experts   int               `json:"experts"`
	LMHead    int               `json:"lm_head"`
	Other     int               `json:"other"`
	Examples  map[string]string `json:"examples,omitempty"`
}

func InspectTensorNames(names []string) TensorCoverage {
	cov := TensorCoverage{Total: len(names), Examples: map[string]string{}}
	sorted := append([]string(nil), names...)
	sort.Strings(sorted)
	for _, name := range sorted {
		g := TensorGroup(name)
		switch g {
		case "embedding":
			cov.Embedding++
		case "layers":
			cov.Layers++
		case "router":
			cov.Router++
		case "experts":
			cov.Experts++
		case "lm_head":
			cov.LMHead++
		default:
			cov.Other++
		}
		if _, ok := cov.Examples[g]; !ok {
			cov.Examples[g] = name
		}
	}
	if len(cov.Examples) == 0 {
		cov.Examples = nil
	}
	return cov
}

func TensorGroup(name string) string {
	s := strings.ToLower(name)
	switch {
	case strings.Contains(s, "embed_tokens") || strings.Contains(s, "embedding"):
		return "embedding"
	case strings.Contains(s, "lm_head"):
		return "lm_head"
	case strings.Contains(s, "router") || strings.Contains(s, "gate"):
		return "router"
	case strings.Contains(s, "experts"):
		return "experts"
	case strings.Contains(s, "layers"):
		return "layers"
	default:
		return "other"
	}
}
