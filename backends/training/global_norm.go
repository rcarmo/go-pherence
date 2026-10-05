package training

import (
	"context"
	"fmt"
	"math"
)

// ResidentGlobalNormDriver provides device reduction of scaled chunk summaries
// and a clipping scale. Validation methods are pure host range/kernel admission.
// Merge modes 0=max scan, 2=fixed-max compensated sum return [maximum, scaledSquare, invalid, correction]; clip mode1 replaces correction with scale.
// The coordinator downloads the final four scalars for admission/diagnostics;
// AdamW still receives the device-computed scale through its existing constants.
type ResidentGlobalNormDriver interface {
	ValidateNormMerge(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, int64, uint32) error
	NormMerge(context.Context, ResidentBuffer, ResidentBuffer, ResidentBuffer, int, int64, uint32) error
	ValidateNormClip(context.Context, ResidentBuffer, ResidentBuffer, int64, float32) error
	NormClip(context.Context, ResidentBuffer, ResidentBuffer, int64, float32) error
}

func NormMergeShape(n int, previous int64) (int, error) {
	count, err := NormStatsShape(n)
	if err != nil {
		return 0, err
	}
	if previous < 0 || previous > (1<<38)-int64(n) {
		return 0, fmt.Errorf("global norm element ceiling")
	}
	return count, nil
}
func NormClipShape(elements int64, maxNorm float32) error {
	if elements < 1 || elements > 1<<38 || !residentFinite(maxNorm) || maxNorm <= 0 {
		return fmt.Errorf("global norm clipping bound")
	}
	return nil
}

// DecodeGlobalNorm validates the device diagnostic/scale boundary. F64 sqrt is
// diagnostic only; no host clipping decision replaces the device scale.
func DecodeGlobalNorm(values []float32, elements int64) (float64, float32, error) {
	if len(values) != 4 || elements < 1 || elements > 1<<38 {
		return 0, 0, fmt.Errorf("global norm result shape")
	}
	maximum, scaled, flag, scale := values[0], values[1], values[2], values[3]
	if !residentFinite(maximum) || !residentFinite(scaled) || !residentFinite(flag) || !residentFinite(scale) || flag != 0 || maximum < 0 || scaled < 0 || scaled > float32(elements)*(1+2e-6) || (maximum == 0 && scaled != 0) || (maximum > 0 && scaled < 1) || scale < 0 || scale > 1 {
		return 0, 0, fmt.Errorf("invalid/nonfinite global norm boundary")
	}
	norm := float64(maximum) * math.Sqrt(float64(scaled))
	if math.IsInf(norm, 0) || math.IsNaN(norm) {
		return 0, 0, fmt.Errorf("global norm diagnostic overflow")
	}
	return norm, scale, nil
}
func (s *ResidentState) reduceGlobalNorm(ctx context.Context, maxNorm float32) (float64, float32, error) {
	driver := s.driver.(ResidentGlobalNormDriver)
	a, b := s.globalA, s.globalB
	var elements int64
	for _, mode := range []uint32{0, 2} {
		var previous int64
		for _, e := range s.entries {
			var err error
			if s.shared != nil {
				err = s.sharedStats(ctx, e.gradient, e.n, false, b, a, previous, mode, false)
			} else {
				err = driver.ValidateNormMerge(ctx, b, a, e.stats, e.n, previous, mode)
			}
			if err != nil {
				return 0, 0, err
			}
			previous += int64(e.n)
			a, b = b, a
		}
		elements = previous
	}
	if err := driver.ValidateNormClip(ctx, s.globalClip, a, elements, maxNorm); err != nil {
		return 0, 0, err
	}
	a, b = s.globalA, s.globalB
	if err := a.Upload(ctx, []float32{0, 0, 0, 0}); err != nil {
		s.terminal = err
		return 0, 0, err
	}
	for _, mode := range []uint32{0, 2} {
		var previous int64
		for _, e := range s.entries {
			if s.shared != nil {
				if err := s.sharedStats(ctx, e.gradient, e.n, false, b, a, previous, mode, true); err != nil {
					s.terminal = err
					return 0, 0, err
				}
			} else {
				if mode == 0 {
					if err := s.driver.NormStats(ctx, e.stats, e.gradient, e.n, false); err != nil {
						s.terminal = err
						return 0, 0, err
					}
				}
				if err := driver.NormMerge(ctx, b, a, e.stats, e.n, previous, mode); err != nil {
					s.terminal = err
					return 0, 0, err
				}
			}
			previous += int64(e.n)
			a, b = b, a
		}
	}
	if err := driver.NormClip(ctx, s.globalClip, a, elements, maxNorm); err != nil {
		s.terminal = err
		return 0, 0, err
	}
	values := make([]float32, 4)
	if err := s.globalClip.Download(ctx, values); err != nil {
		s.terminal = err
		return 0, 0, err
	}
	norm, scale, err := DecodeGlobalNorm(values, elements)
	if err != nil {
		s.terminal = err
	}
	return norm, scale, err
}
