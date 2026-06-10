package lfm2

import (
	"github.com/rcarmo/go-pherence/loader/safetensors"
	tensorinspect "github.com/rcarmo/go-pherence/model/internal/tensorinspect"
)

type TensorShapeSummary = tensorinspect.TensorShapeSummary
type TensorShape = tensorinspect.TensorShape

func InspectTensorShapes(infos map[string]safetensors.TensorInfo) TensorShapeSummary {
	return tensorinspect.InspectTensorShapes(infos, TensorGroup)
}
