package tensorinspect

import (
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
	"github.com/rcarmo/go-pherence/model/inspect"
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
	Readiness TensorReadiness   `json:"readiness"`
}

type TensorReadiness struct {
	Ready           bool            `json:"ready"`
	PresentRequired map[string]bool `json:"present_required"`
	MissingRequired []string        `json:"missing_required,omitempty"`
	PresentOptional map[string]bool `json:"present_optional,omitempty"`
}

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

func InspectTensorNames(names []string, required, optional map[string][]string) TensorCoverage {
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
	cov.Readiness = InspectTensorReadiness(sorted, required, optional)
	return cov
}

func InspectTensorReadiness(names []string, required, optional map[string][]string) TensorReadiness {
	presentRequired := make(map[string]bool, len(required))
	var missing []string
	for group, markers := range required {
		present := inspect.AnyTensorMarker(names, markers)
		presentRequired[group] = present
		if !present {
			missing = append(missing, group)
		}
	}
	sort.Strings(missing)
	presentOptional := make(map[string]bool, len(optional))
	for group, markers := range optional {
		presentOptional[group] = inspect.AnyTensorMarker(names, markers)
	}
	return TensorReadiness{Ready: len(missing) == 0, PresentRequired: presentRequired, MissingRequired: missing, PresentOptional: presentOptional}
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
