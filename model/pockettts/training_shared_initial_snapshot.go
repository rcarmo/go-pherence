package pockettts

import train "github.com/rcarmo/go-pherence/backends/training"

// sharedInitialSnapshot borrows only the backend's freshly owned State() copy
// during synchronous constructor validation/uploads. It never borrows live
// trainer arrays, escapes to a tape, or changes public snapshot ownership.
// The backend retains the independent host copy as its mutation guard. Native
// constructors must not retain/mutate these sources after returning.
func sharedInitialSnapshot(host FullTrainingState) (train.ResidentSnapshot, error) {
	if err := host.Validate(); err != nil {
		return train.ResidentSnapshot{}, err
	}
	c := host.AdamW
	s := train.ResidentSnapshot{Step: uint64(host.Step), Config: train.ResidentConfig{LearningRate: c.LearningRate, Beta1: c.Beta1, Beta2: c.Beta2, Epsilon: c.Epsilon, WeightDecay: c.WeightDecay, EMADecay: host.EMADecay}, Parameters: map[string]train.ResidentParameter{}}
	moments := map[string]NamedAdamState{}
	for _, m := range host.Adam {
		moments[m.Name] = m
	}
	ema := map[string][]float32{}
	for _, p := range host.EMA {
		ema[p.Name] = p.Values
	}
	for _, p := range host.Params {
		m, ok := moments[p.Name]
		if !ok {
			m = NamedAdamState{M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))}
		}
		shadow := ema[p.Name]
		if host.EMADecay == 0 {
			shadow = make([]float32, len(p.Values))
		}
		s.Parameters[p.Name] = train.ResidentParameter{Parameter: p.Values, M: m.M, V: m.V, EMA: shadow}
	}
	if _, err := train.ResidentBytes(s); err != nil {
		return train.ResidentSnapshot{}, err
	}
	return s, nil
}
