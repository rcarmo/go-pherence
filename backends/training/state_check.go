package training

import (
	"context"
	"fmt"
)

// NewStateCheckResidentState additionally merges post-update mutable-state
// diagnostics on device. Reuses global A/B buffers after gradient clipping;
// existing constructors retain per-parameter checks/downloads. No extra bytes.
func NewStateCheckResidentState(ctx context.Context, driver ResidentDriver, snapshot ResidentSnapshot, maxBytes int64) (*ResidentState, error) {
	return newResidentState(ctx, driver, snapshot, maxBytes, true, true, true)
}
func DecodeStateCheck(values []float32) error {
	if len(values) != 4 {
		return fmt.Errorf("resident state-check diagnostic shape")
	}
	for _, v := range values {
		if !residentFinite(v) {
			return fmt.Errorf("resident state-check nonfinite diagnostic")
		}
	}
	// Max-scan mode validates chunk values, carries the invalid flag and resets
	// sum/correction to zero. No host square or norm is needed for state checks.
	if values[0] < 0 || values[1] != 0 || values[2] != 0 || values[3] != 0 {
		return fmt.Errorf("invalid resident mutable-state diagnostic")
	}
	return nil
}
func (s *ResidentState) validateDeviceStateCheck(ctx context.Context) error {
	driver := s.driver.(ResidentGlobalNormDriver)
	a, b := s.globalA, s.globalB
	var previous int64
	for _, e := range s.entries {
		for i, source := range []ResidentBuffer{e.parameter, e.m, e.v, e.ema} {
			var err error
			if s.shared != nil {
				err = s.sharedStats(ctx, source, e.n, i == 2, b, a, previous, 0, false)
			} else {
				err = driver.ValidateNormMerge(ctx, b, a, e.stats, e.n, previous, 0)
			}
			if err != nil {
				return err
			}
			previous += int64(e.n)
			a, b = b, a
		}
	}
	return nil
}
func (s *ResidentState) checkDeviceState(ctx context.Context) error {
	driver := s.driver.(ResidentGlobalNormDriver)
	a, b := s.globalA, s.globalB
	if err := a.Upload(ctx, []float32{0, 0, 0, 0}); err != nil {
		return err
	}
	var previous int64
	for _, e := range s.entries {
		for _, slot := range []struct {
			buffer      ResidentBuffer
			nonnegative bool
		}{{e.parameter, false}, {e.m, false}, {e.v, true}, {e.ema, false}} {
			if s.shared != nil {
				if err := s.sharedStats(ctx, slot.buffer, e.n, slot.nonnegative, b, a, previous, 0, true); err != nil {
					return err
				}
			} else {
				if err := s.driver.NormStats(ctx, e.stats, slot.buffer, e.n, slot.nonnegative); err != nil {
					return err
				}
				if err := driver.NormMerge(ctx, b, a, e.stats, e.n, previous, 0); err != nil {
					return err
				}
			}
			previous += int64(e.n)
			a, b = b, a
		}
	}
	values := make([]float32, 4)
	if err := a.Download(ctx, values); err != nil {
		return err
	}
	return DecodeStateCheck(values)
}
