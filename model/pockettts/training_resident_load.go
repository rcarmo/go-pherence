package pockettts

import (
	"fmt"
	"sort"
	"unsafe"
)

// loadResidentCompletedState is private to single-owner resident updates. The
// supplied state is the independent owned backend boundary, not arbitrary
// checkpoint restore input. Preserve that ownership while reusing valid live
// moment/EMA arrays. Validate everything, including aliases, before any write.
// Public LoadState retains its replacement/owned-copy restore behavior.
func (t *FullTrainer) loadResidentCompletedState(s FullTrainingState) error {
	return t.loadResidentOwnedState(s, false)
}

// loadSharedRestoredState accepts only a fresh trainer and a positive-step
// independently decoded checkpoint. Step-zero restores retain the public path.
// Unlike completed updates, the restored step can be any positive boundary.
func (t *FullTrainer) loadSharedRestoredState(s FullTrainingState) error {
	return t.loadResidentOwnedState(s, true)
}

func (t *FullTrainer) loadResidentOwnedState(s FullTrainingState, restore bool) error {
	if t == nil {
		return fmt.Errorf("nil resident trainer load")
	}
	if restore && (t.StepCount != 0 || len(t.Moments) != 0) {
		return fmt.Errorf("shared restore requires fresh trainer")
	}
	// Admit the complete state against live inventory once. Calling Validate()
	// first would scan every source array twice through the validators below.
	if err := s.AdamW.validate(); err != nil {
		return err
	}
	if err := t.refreshParameterBindings(); err != nil {
		return err
	}
	if s.Version != 1 || !isFinite(s.EMADecay) || s.EMADecay < 0 || s.EMADecay >= 1 || (s.EMADecay == 0 && len(s.EMA) != 0) || s.Step <= 0 || (!restore && s.Step != t.StepCount+1) || s.AdamW != t.Config || s.EMADecay != t.EMADecay {
		return fmt.Errorf("resident completed load header")
	}
	params, err := validateTensorList(s.Params, t.params, "resident parameter")
	if err != nil {
		return err
	}
	buffers := fullBufferMap(t.FlowLM, t.Flow)
	loadedBuffers, err := validateTensorList(s.Buffers, buffers, "resident buffer")
	if err != nil {
		return err
	}
	moments, err := validateAdamList(s.Adam, t.params, true)
	if err != nil {
		return err
	}
	var shadows map[string][]float32
	if t.EMADecay > 0 {
		shadows, err = validateTensorList(s.EMA, t.params, "resident EMA")
		if err != nil {
			return err
		}
	}
	if (t.StepCount > 0 || len(t.Moments) > 0) && len(t.Moments) != len(t.params) {
		return fmt.Errorf("resident load live moment inventory")
	}
	for name := range t.Moments {
		if _, ok := t.params[name]; !ok {
			return fmt.Errorf("resident load foreign live moment")
		}
	}
	if t.EMADecay > 0 && len(t.EMA) != len(t.params) {
		return fmt.Errorf("resident load live EMA inventory")
	}
	// Stage maps/one-time missing storage without changing the trainer. All source
	// and destination arrays must be pairwise disjoint, including frozen buffers.
	// This prevents cross-tensor copy corruption and backend-boundary aliases.
	nextMoments := make(map[string]NamedAdamState, len(t.params))
	var nextEMA map[string][]float32
	if t.EMADecay > 0 {
		nextEMA = make(map[string][]float32, len(t.params))
	}
	ranges := make([]residentLoadRange, 0, 8*len(t.params)+2*len(buffers))
	add := func(x []float32) {
		ranges = append(ranges, residentLoadRange{uintptr(unsafe.Pointer(&x[0])), uintptr(unsafe.Pointer(&x[0])) + uintptr(len(x))*4})
	}
	for name, x := range t.params {
		if len(x) == 0 || !finiteF32(x) {
			return fmt.Errorf("resident load empty/nonfinite live parameter")
		}
		src := moments[name]
		for _, v := range src.V {
			if v < 0 {
				return fmt.Errorf("resident load negative second moment")
			}
		}
		dst, ok := t.Moments[name]
		if !ok {
			if t.StepCount > 0 || len(t.Moments) > 0 {
				return fmt.Errorf("resident load missing live moment")
			}
			dst = NamedAdamState{M: make([]float32, len(x)), V: make([]float32, len(x))}
		}
		if len(dst.M) != len(x) || len(dst.V) != len(x) || !finiteF32(dst.M) || !finiteF32(dst.V) {
			return fmt.Errorf("resident load live moment shape/nonfinite")
		}
		for _, v := range dst.V {
			if v < 0 {
				return fmt.Errorf("resident load negative live second moment")
			}
		}
		dst.Name = name
		nextMoments[name] = dst
		for _, a := range [][]float32{x, params[name], src.M, src.V, dst.M, dst.V} {
			add(a)
		}
		if t.EMADecay > 0 {
			shadow, ok := t.EMA[name]
			if !ok {
				return fmt.Errorf("resident load missing live EMA")
			}
			if len(shadow) != len(x) || !finiteF32(shadow) {
				return fmt.Errorf("resident load live EMA shape/nonfinite")
			}
			nextEMA[name] = shadow
			add(shadow)
			add(shadows[name])
		}
	}
	for name, x := range buffers {
		if len(x) == 0 || !finiteF32(x) {
			return fmt.Errorf("resident load empty/nonfinite live buffer")
		}
		add(x)
		add(loadedBuffers[name])
	}
	if err := validateResidentLoadRanges(ranges); err != nil {
		return err
	}
	// No fallible work remains after complete-pass validation.
	for name, x := range t.params {
		copy(x, params[name])
		dst, src := nextMoments[name], moments[name]
		copy(dst.M, src.M)
		copy(dst.V, src.V)
		if t.EMADecay > 0 {
			copy(nextEMA[name], shadows[name])
		}
	}
	for name, x := range buffers {
		copy(x, loadedBuffers[name])
	}
	t.Moments, t.EMA, t.StepCount = nextMoments, nextEMA, s.Step
	return nil
}

type residentLoadRange struct{ start, end uintptr }

func validateResidentLoadRanges(ranges []residentLoadRange) error {
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
	for i, x := range ranges {
		if x.start == 0 || x.end <= x.start || (i > 0 && ranges[i-1].end > x.start) {
			return fmt.Errorf("resident completed load storage alias/range")
		}
	}
	return nil
}
