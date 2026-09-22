package pockettts

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestFullTrainingStateRejectsMalformedWithoutMutation(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, weight := trainingStepModelsFromOracle(t, oracle)
	config := AdamWConfig{LearningRate: .001, Beta1: .9, Beta2: .95, Epsilon: 1e-8, WeightDecay: .1}
	trainer, err := NewFullTrainer(lm, flow, weight, config, .9)
	if err != nil {
		t.Fatal(err)
	}
	before, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	cases := []func(*FullTrainingState){func(s *FullTrainingState) { s.Buffers = nil }, func(s *FullTrainingState) { s.Params = append(s.Params, s.Params[0]) }, func(s *FullTrainingState) { s.Params[0].Values[0] = float32(math.NaN()) }, func(s *FullTrainingState) { s.EMADecay = .9; s.EMA = nil }}
	for i, mutate := range cases {
		bad := cloneFullState(before)
		mutate(&bad)
		if err = trainer.LoadState(bad); err == nil {
			t.Fatalf("case %d accepted", i)
		}
		after, e := trainer.State()
		if e != nil {
			t.Fatal(e)
		}
		if !reflect.DeepEqual(before, after) {
			t.Fatalf("case %d mutated trainer", i)
		}
	}
}
func cloneFullState(s FullTrainingState) FullTrainingState {
	data, _ := json.Marshal(s)
	var out FullTrainingState
	_ = json.Unmarshal(data, &out)
	return out
}
