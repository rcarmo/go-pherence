package lfm2

import (
	"sort"
	"strings"
)

var requiredTensorMarkers = map[string][]string{
	"embedding": {"embed_tokens", "embedding"},
	"layers":    {"layers"},
	"router":    {"router", "gate"},
	"experts":   {"experts"},
}

var optionalTensorMarkers = map[string][]string{
	"lm_head": {"lm_head"},
}

type TensorCoverage struct {
	Total     int               `json:"total"`
	Embedding int               `json:"embedding"`
	Layers    int               `json:"layers"`
	Router    int               `json:"router"`
	Experts   int               `json:"experts"`
	LMHead    int               `json:"lm_head"`
	Other     int               `json:"other"`
	Examples  map[string]string `json:"examples,omitempty"`
	Readiness TensorReadiness   `json:"readiness"`
}

type TensorReadiness struct {
	Ready           bool            `json:"ready"`
	PresentRequired map[string]bool `json:"present_required"`
	MissingRequired []string        `json:"missing_required,omitempty"`
	PresentOptional map[string]bool `json:"present_optional,omitempty"`
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
	cov.Readiness = InspectTensorReadiness(sorted)
	return cov
}

func InspectTensorReadiness(names []string) TensorReadiness {
	presentRequired := make(map[string]bool, len(requiredTensorMarkers))
	var missing []string
	for group, markers := range requiredTensorMarkers {
		present := anyTensorMarker(names, markers)
		presentRequired[group] = present
		if !present {
			missing = append(missing, group)
		}
	}
	sort.Strings(missing)
	presentOptional := make(map[string]bool, len(optionalTensorMarkers))
	for group, markers := range optionalTensorMarkers {
		presentOptional[group] = anyTensorMarker(names, markers)
	}
	return TensorReadiness{Ready: len(missing) == 0, PresentRequired: presentRequired, MissingRequired: missing, PresentOptional: presentOptional}
}

func anyTensorMarker(names, markers []string) bool {
	for _, name := range names {
		s := strings.ToLower(name)
		for _, marker := range markers {
			if strings.Contains(s, marker) {
				return true
			}
		}
	}
	return false
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
