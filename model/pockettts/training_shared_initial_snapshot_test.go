package pockettts

import (
	"bytes"
	"errors"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

func TestPocketSharedInitialSnapshotOwnership(t *testing.T) {
	for _, decay := range []float32{0, .9} {
		lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
		flow.Condition = tinyLinear(4, 4, -.02)
		tr, e := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), decay)
		if e != nil {
			t.Fatal(e)
		}
		host := mustFullState(t, tr)
		before := mustFullState(t, tr)
		s, e := sharedInitialSnapshot(host)
		if e != nil {
			t.Fatal(e)
		}
		public, e := ResidentSnapshotFromFullState(host)
		if e != nil {
			t.Fatal(e)
		}
		if !reflect.DeepEqual(s, public) {
			t.Fatal("borrowed initial values drift")
		}
		for _, p := range host.Params {
			got := s.Parameters[p.Name]
			if &got.Parameter[0] != &p.Values[0] {
				t.Fatal("private P still cloned")
			}
			if f32SlicesOverlap(got.M, got.V) || f32SlicesOverlap(got.M, got.Parameter) || f32SlicesOverlap(got.V, got.Parameter) || f32SlicesOverlap(got.EMA, got.Parameter) {
				t.Fatal("initial state arrays alias")
			}
			if decay > 0 {
				for _, shadow := range host.EMA {
					if shadow.Name == p.Name && &got.EMA[0] != &shadow.Values[0] {
						t.Fatal("private EMA still cloned")
					}
				}
			}
		}
		name := host.Params[0].Name
		s.Parameters[name].Parameter[0] += 1
		if !reflect.DeepEqual(before, mustFullState(t, tr)) {
			t.Fatal("private snapshot aliases live trainer")
		}
		if public.Parameters[name].Parameter[0] == s.Parameters[name].Parameter[0] {
			t.Fatal("public clone contract changed")
		}
		// Restored/completed moments are borrowed only from independent host state.
		host = before
		host.Step = 1
		for _, p := range host.Params {
			host.Adam = append(host.Adam, NamedAdamState{Name: p.Name, M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))})
		}
		s, e = sharedInitialSnapshot(host)
		if e != nil {
			t.Fatal(e)
		}
		for _, m := range host.Adam {
			v := s.Parameters[m.Name]
			if &v.M[0] != &m.M[0] || &v.V[0] != &m.V[0] {
				t.Fatal("restore moments cloned")
			}
		}
	}
}
func TestPocketSharedInitialSnapshotAdmission(t *testing.T) {
	if _, e := sharedInitialSnapshot(FullTrainingState{}); e == nil {
		t.Fatal("invalid header")
	}
	lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	tr, e := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"nonfinite", "duplicate", "negativeV", "missingEMA"} {
		t.Run(kind, func(t *testing.T) {
			host := mustFullState(t, tr)
			switch kind {
			case "nonfinite":
				host.Params[0].Values[0] = float32(math.NaN())
			case "duplicate":
				host.Params = append(host.Params, host.Params[0])
			case "negativeV":
				p := host.Params[0]
				host.Adam = []NamedAdamState{{Name: p.Name, M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))}}
				host.Adam[0].V[0] = -1
			case "missingEMA":
				host.EMA = nil
			}
			if _, e := sharedInitialSnapshot(host); e == nil {
				t.Fatal("invalid state admitted")
			}
		})
	}
}
func TestPocketSharedCheckpointBorrowAdmission(t *testing.T) {
	lm, flow, w := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	tr, e := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if e != nil {
		t.Fatal(e)
	}
	for _, p := range tr.names {
		v := tr.params[p]
		tr.Moments[p] = NamedAdamState{Name: p, M: make([]float32, len(v)), V: make([]float32, len(v))}
	}
	tr.StepCount = 1
	b := &ResidentTrainingBackend{Scope: "vulkan-resident-sharedworkspace-hybrid-v1", trainer: tr, host: mustFullState(t, tr), state: &train.ResidentState{}}
	// A real state's completed count is required; synthetic zero cannot fake it.
	if _, e = b.sharedCheckpointState(tr); e == nil {
		t.Fatal("step-count mismatch accepted")
	}
	tr.StepCount = 0
	b.host = mustFullState(t, tr)
	state, e := b.sharedCheckpointState(tr)
	if e != nil {
		t.Fatal(e)
	}
	if &state.Params[0].Values[0] != &b.host.Params[0].Values[0] {
		t.Fatal("checkpoint state copied")
	}
	before := mustFullState(t, tr)
	var output bytes.Buffer
	if e = writeFullTrainingBinary(&output, state); e != nil {
		t.Fatal(e)
	}
	loaded, e := ReadFullTrainingStateBinary(bytes.NewReader(output.Bytes()), int64(output.Len()))
	if e != nil || !reflect.DeepEqual(loaded, state) {
		t.Fatal("borrowed serialisation", e)
	}
	if !reflect.DeepEqual(before, mustFullState(t, tr)) {
		t.Fatal("serialisation mutated live state")
	}
	for _, kind := range []string{"active", "terminal", "closed", "mutated", "scope", "missing-state", "invalid-host"} {
		t.Run(kind, func(t *testing.T) {
			copyBackend := *b
			switch kind {
			case "active":
				copyBackend.active = true
			case "terminal":
				copyBackend.terminal = errors.New("terminal fixture")
			case "closed":
				copyBackend.closed = true
			case "mutated":
				old := tr.params[tr.names[0]][0]
				tr.params[tr.names[0]][0] += 1
				defer func() { tr.params[tr.names[0]][0] = old }()
			case "scope":
				copyBackend.Scope = "vulkan-resident-devicehandoff-hybrid-v1"
			case "missing-state":
				copyBackend.state = nil
				copyBackend.host = FullTrainingState{}
			case "invalid-host":
				copyBackend.host.Version = 0
			}
			if _, e = copyBackend.sharedCheckpointState(tr); e == nil {
				t.Fatal("invalid borrowed boundary")
			}
		})
	}
}

func BenchmarkPocketSharedInitialSnapshot(b *testing.B) {
	const n = 32768
	host := FullTrainingState{Version: 1, AdamW: DefaultAdamWConfig(), EMADecay: .9, Params: []NamedTrainingTensor{{Name: "p", Values: make([]float32, n)}}, Buffers: []NamedTrainingTensor{{Name: "fixed", Values: []float32{1}}}, EMA: []NamedTrainingTensor{{Name: "p", Values: make([]float32, n)}}}
	for _, private := range []bool{false, true} {
		label := "owned-public"
		if private {
			label = "borrowed-private"
		}
		b.Run(label, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				var e error
				if private {
					_, e = sharedInitialSnapshot(host)
				} else {
					_, e = ResidentSnapshotFromFullState(host)
				}
				if e != nil {
					b.Fatal(e)
				}
			}
		})
	}
}
