package hunyuan3d

import (
	"fmt"
	"sort"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// TensorCoverage summarizes whether a Hunyuan3D safetensors manifest contains
// the three required checkpoint groups used by the shape pipeline.
type TensorCoverage struct {
	Total       int
	Model       int
	VAE         int
	Conditioner int
	Other       int
	Missing     []string
	Examples    map[loaderconfig.Hunyuan3DTensorGroup][]string
}

// ValidateTensorCoverage checks a tensor-name inventory before any runtime
// binding is attempted. It deliberately validates group presence rather than
// exact per-layer keys because upstream Hunyuan3D variants differ in naming and
// this package still stops at metadata/parity scaffolding.
func ValidateTensorCoverage(names []string) (TensorCoverage, error) {
	inv := loaderconfig.SummarizeHunyuan3DTensors(names)
	out := TensorCoverage{
		Total:       inv.Total,
		Model:       inv.Model,
		VAE:         inv.VAE,
		Conditioner: inv.Conditioner,
		Other:       inv.Other,
		Examples:    inv.Examples,
	}
	if out.Model == 0 {
		out.Missing = append(out.Missing, "model.*")
	}
	if out.VAE == 0 {
		out.Missing = append(out.Missing, "vae.*")
	}
	if out.Conditioner == 0 {
		out.Missing = append(out.Missing, "conditioner.*")
	}
	if len(out.Missing) > 0 {
		return out, fmt.Errorf("invalid Hunyuan3D tensor coverage: missing %s", strings.Join(out.Missing, ", "))
	}
	return out, nil
}

// TensorNames returns sorted tensor names from any map keyed by tensor name.
// It is useful for adapting safetensors manifests without coupling this package
// to a concrete loader type.
func TensorNames[T any](manifest map[string]T) []string {
	names := make([]string, 0, len(manifest))
	for name := range manifest {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
