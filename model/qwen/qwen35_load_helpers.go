package qwen

import (
	"fmt"
	"math"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

func loadQwen35DenseOrNVFP4(src Qwen35TensorSource, name string, dense **tensor.Tensor, q **Qwen35NVFP4Weight, want []int) error {
	if dense == nil {
		return fmt.Errorf("nil dense destination for %s", name)
	}
	t, err := src.Get(name, want)
	if err == nil {
		*dense = t
		return nil
	}
	if q == nil {
		return err
	}
	raw, ok := unwrapQwen35RawTensorSource(src)
	if !ok {
		return err
	}
	if len(want) != 2 {
		return err
	}
	qw, qerr := LoadQwen35NVFP4WeightCandidates(raw, name, []int{want[1], want[0]})
	if qerr != nil {
		qw, qerr = LoadQwen35NVFP4WeightCandidates(raw, name, want)
	}
	if qerr != nil {
		return fmt.Errorf("%v; NVFP4 fallback: %w", err, qerr)
	}
	*q = qw
	return nil
}

func loadQwen35DenseA(src Qwen35TensorSource, name string, want []int) (*tensor.Tensor, error) {
	base := src
	if cand, ok := src.(CandidateQwen35TensorSource); ok {
		base = cand.Source
	}
	var errs []string
	for _, candidate := range qwen35DirectTensorNameCandidates(name) {
		t, err := base.Get(candidate, want)
		if err == nil {
			return t, nil
		}
		errs = append(errs, fmt.Sprintf("%s: %v", candidate, err))
	}
	for _, candidate := range qwen35DirectTensorNameCandidates(strings.TrimSuffix(name, ".A") + ".A_log") {
		t, err := base.Get(candidate, want)
		if err == nil {
			for i, v := range t.Data() {
				t.Data()[i] = -float32(math.Exp(float64(v)))
			}
			return t, nil
		}
		errs = append(errs, fmt.Sprintf("%s: %v", candidate, err))
	}
	return nil, fmt.Errorf("load %s: no Qwen3.5 A/A_log tensor-name candidate matched (%s)", name, strings.Join(errs, "; "))
}

func qwen35DirectTensorNameCandidates(name string) []string {
	norm := loaderconfig.NormalizeQwen35TensorName(name)
	out := []string{norm}
	for _, candidate := range []string{loaderconfig.Qwen35NestedTextPrefix + norm, "model.language_model." + strings.TrimPrefix(norm, "model."), "language_model." + norm} {
		seen := false
		for _, existing := range out {
			if existing == candidate {
				seen = true
				break
			}
		}
		if !seen {
			out = append(out, candidate)
		}
	}
	return out
}

func unwrapQwen35RawTensorSource(src Qwen35TensorSource) (Qwen35RawTensorSource, bool) {
	if raw, ok := src.(Qwen35RawTensorSource); ok {
		return raw, true
	}
	if cand, ok := src.(CandidateQwen35TensorSource); ok {
		if raw, ok := cand.Source.(Qwen35RawTensorSource); ok {
			return raw, true
		}
	}
	return nil, false
}
