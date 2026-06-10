package lfm2

import tensorinspect "github.com/rcarmo/go-pherence/model/internal/tensorinspect"

var requiredTensorMarkers = map[string][]string{
	"embedding": {"embed_tokens", "embedding"},
	"layers":    {"layers"},
	"router":    {"router", "gate"},
	"experts":   {"experts"},
}

var optionalTensorMarkers = map[string][]string{
	"lm_head": {"lm_head"},
}

type TensorCoverage = tensorinspect.TensorCoverage
type TensorReadiness = tensorinspect.TensorReadiness

func InspectTensorNames(names []string) TensorCoverage {
	return tensorinspect.InspectTensorNames(names, requiredTensorMarkers, optionalTensorMarkers)
}

func InspectTensorReadiness(names []string) TensorReadiness {
	return tensorinspect.InspectTensorReadiness(names, requiredTensorMarkers, optionalTensorMarkers)
}

func TensorGroup(name string) string {
	return tensorinspect.TensorGroup(name)
}
