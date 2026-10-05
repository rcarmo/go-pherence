package pockettts

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// ResidentSnapshotFromFullState converts a validated checkpoint into native
// optimiser initial state. Frozen model buffers remain CPU checkpoint-owned.
// Step-zero missing moments become explicit zeros; disabled EMA binds distinct
// scratch. No device initialisation or live trainer mutation occurs here.
func ResidentSnapshotFromFullState(state FullTrainingState) (train.ResidentSnapshot, error) {
	var snapshot train.ResidentSnapshot
	if err := state.Validate(); err != nil {
		return snapshot, err
	}
	c := state.AdamW
	snapshot = train.ResidentSnapshot{Step: uint64(state.Step), Config: train.ResidentConfig{LearningRate: c.LearningRate, Beta1: c.Beta1, Beta2: c.Beta2, Epsilon: c.Epsilon, WeightDecay: c.WeightDecay, EMADecay: state.EMADecay}, Parameters: map[string]train.ResidentParameter{}}
	moments := map[string]NamedAdamState{}
	for _, m := range state.Adam {
		moments[m.Name] = m
	}
	ema := map[string][]float32{}
	for _, p := range state.EMA {
		ema[p.Name] = p.Values
	}
	for _, p := range state.Params {
		m, ok := moments[p.Name]
		if !ok {
			m = NamedAdamState{M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))}
		}
		shadow := ema[p.Name]
		if state.EMADecay == 0 {
			shadow = make([]float32, len(p.Values))
		}
		snapshot.Parameters[p.Name] = train.ResidentParameter{Parameter: append([]float32(nil), p.Values...), M: append([]float32(nil), m.M...), V: append([]float32(nil), m.V...), EMA: append([]float32(nil), shadow...)}
	}
	if _, err := train.ResidentBytes(snapshot); err != nil {
		return train.ResidentSnapshot{}, err
	}
	return snapshot, nil
}

// FullStateFromResidentSnapshot joins one completed coherent native snapshot
// with the immutable checkpoint buffers/topology. Validate the entire state
// before publishing or calling LoadState; no partial host writes are performed.
// This is checkpoint conversion only, not a device-aware runner step.
func FullStateFromResidentSnapshot(template FullTrainingState, snapshot train.ResidentSnapshot) (FullTrainingState, error) {
	return fullStateFromResidentSnapshot(template, snapshot, true)
}

// takeFullStateFromResidentSnapshot is private to completed resident updates.
// Only accept the fresh uniquely owned arrays from ResidentState.Snapshot(),
// never an arbitrary/user-supplied snapshot. Move those arrays into
// the backend's owned boundary instead of cloning them again. The input is
// consumed only after complete validation; errors leave it intact. Callers must
// not retain aliases to the consumed arrays. Public conversions still clone.
// LoadState copies into the trainer, preserving a separate mutation guard.
// Consumption occurs at conversion success, not at LoadState success. A later
// LoadState failure is terminal in callers; the consumed source cannot retry.
func takeFullStateFromResidentSnapshot(template FullTrainingState, snapshot *train.ResidentSnapshot) (FullTrainingState, error) {
	if snapshot == nil {
		return FullTrainingState{}, fmt.Errorf("nil resident snapshot handoff")
	}
	state, err := fullStateFromResidentSnapshot(template, *snapshot, false)
	if err != nil {
		return FullTrainingState{}, err
	}
	*snapshot = train.ResidentSnapshot{}
	return state, nil
}
func fullStateFromResidentSnapshot(template FullTrainingState, snapshot train.ResidentSnapshot, clone bool) (FullTrainingState, error) {
	var state FullTrainingState
	if err := template.Validate(); err != nil {
		return state, err
	}
	if _, err := train.ResidentBytes(snapshot); err != nil {
		return state, err
	}
	c := template.AdamW
	expected := train.ResidentConfig{LearningRate: c.LearningRate, Beta1: c.Beta1, Beta2: c.Beta2, Epsilon: c.Epsilon, WeightDecay: c.WeightDecay, EMADecay: template.EMADecay}
	if snapshot.Config != expected || snapshot.Step < uint64(template.Step) || snapshot.Step > uint64(math.MaxInt) || len(snapshot.Parameters) != len(template.Params) {
		return state, fmt.Errorf("resident snapshot config/step/inventory mismatch")
	}
	state = FullTrainingState{Version: template.Version, Step: int(snapshot.Step), AdamW: c, EMADecay: template.EMADecay}
	for _, buffer := range template.Buffers {
		state.Buffers = append(state.Buffers, NamedTrainingTensor{Name: buffer.Name, Values: append([]float32(nil), buffer.Values...)})
	}
	for _, p := range template.Params {
		value, ok := snapshot.Parameters[p.Name]
		if !ok || len(value.Parameter) != len(p.Values) {
			return FullTrainingState{}, fmt.Errorf("resident snapshot parameter %s mismatch", p.Name)
		}
		if clone {
			value.Parameter = append([]float32(nil), value.Parameter...)
			value.M = append([]float32(nil), value.M...)
			value.V = append([]float32(nil), value.V...)
			if state.EMADecay > 0 {
				value.EMA = append([]float32(nil), value.EMA...)
			}
		}
		state.Params = append(state.Params, NamedTrainingTensor{Name: p.Name, Values: value.Parameter})
		state.Adam = append(state.Adam, NamedAdamState{Name: p.Name, M: value.M, V: value.V})
		if state.EMADecay > 0 {
			state.EMA = append(state.EMA, NamedTrainingTensor{Name: p.Name, Values: value.EMA})
		}
	}
	if err := state.Validate(); err != nil {
		return FullTrainingState{}, err
	}
	return state, nil
}
