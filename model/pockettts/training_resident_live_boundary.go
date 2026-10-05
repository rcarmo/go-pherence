package pockettts

// sameResidentLiveBoundary compares live trainer arrays against the owned,
// validated completed snapshot without constructing another full-state copy.
// Equality here is the exact mutation/checkpoint contract, not a generated
// signal acceptance test. It never borrows live arrays into the stored snapshot.
// The trainer is single-owner; concurrent mutation was never supported.
func sameResidentLiveBoundary(host FullTrainingState, t *FullTrainer) bool {
	if t == nil || t.refreshParameterBindings() != nil || host.Step != t.StepCount || host.AdamW != t.Config || host.EMADecay != t.EMADecay {
		return false
	}
	same := func(list []NamedTrainingTensor, live map[string][]float32) bool {
		if len(list) != len(live) {
			return false
		}
		for _, p := range list {
			x, ok := live[p.Name]
			if !ok || !sameResidentValues(p.Values, x) {
				return false
			}
		}
		return true
	}
	if !same(host.Params, t.params) || !same(host.Buffers, fullBufferMap(t.FlowLM, t.Flow)) || len(host.Adam) != len(t.Moments) {
		return false
	}
	for _, p := range host.Adam {
		x, ok := t.Moments[p.Name]
		if !ok || !sameResidentValues(p.M, x.M) || !sameResidentValues(p.V, x.V) {
			return false
		}
	}
	// State() ignores any live EMA map when EMA is disabled; preserve that rule.
	return t.EMADecay == 0 || same(host.EMA, t.EMA)
}
func sameResidentValues(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}
