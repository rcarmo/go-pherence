//go:build linux

package pockettts

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestPocketSharedStorageEmptySlotsAdmission(t *testing.T) {
	for _, kind := range []string{"host-M", "host-V", "host-EMA", "host-buffer", "live-buffer", "live-M", "live-V", "live-EMA", "live-topology"} {
		t.Run(kind, func(t *testing.T) {
			r, b := sharedCompletedCheckpointFixture(t)
			live := r.trainer
			switch kind {
			case "host-M":
				b.host.Adam[0].M = nil
			case "host-V":
				b.host.Adam[0].V = nil
			case "host-EMA":
				b.host.EMA[0].Values = nil
			case "host-buffer":
				b.host.Buffers[0].Values = nil
			case "live-buffer":
				live.FlowLM.LatentMean = nil
			case "live-M":
				name := live.names[0]
				m := live.Moments[name]
				m.M = nil
				live.Moments[name] = m
			case "live-V":
				name := live.names[0]
				m := live.Moments[name]
				m.V = nil
				live.Moments[name] = m
			case "live-EMA":
				live.EMA[live.names[0]] = nil
			case "live-topology":
				live.Weighting.Layers = append(live.Weighting.Layers, live.Weighting.Layers[0])
			}
			step := live.StepCount
			if err := validateSharedSnapshotStorage(b.host, live); err == nil {
				t.Fatal("empty/invalid storage accepted")
			}
			if live.StepCount != step {
				t.Fatal("storage admission advanced live state")
			}
		})
	}
}

func TestPocketSharedCheckpointRejectsEqualValueLiveAliases(t *testing.T) {
	for _, kind := range []string{"P", "M", "V", "EMA", "buffer"} {
		t.Run(kind, func(t *testing.T) {
			r, b := sharedCompletedCheckpointFixture(t)
			switch kind {
			case "P":
				b.host.Params[0].Values = r.trainer.params[b.host.Params[0].Name]
			case "M":
				b.host.Adam[0].M = r.trainer.Moments[b.host.Adam[0].Name].M
			case "V":
				b.host.Adam[0].V = r.trainer.Moments[b.host.Adam[0].Name].V
			case "EMA":
				b.host.EMA[0].Values = r.trainer.EMA[b.host.EMA[0].Name]
			case "buffer":
				b.host.Buffers[0].Values = fullBufferMap(r.trainer.FlowLM, r.trainer.Flow)[b.host.Buffers[0].Name]
			}
			if !sameResidentLiveBoundary(b.host, r.trainer) {
				t.Fatal("alias fixture values should match")
			}
			before := mustFullState(t, r.trainer)
			if _, err := b.sharedCheckpointState(r.trainer); err == nil {
				t.Fatal("equal-value live alias borrowed")
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) {
				t.Fatal("checkpoint alias admission wrote live")
			}
		})
	}
}

func TestPocketSharedCompletedSnapshotOwnerAdmission(t *testing.T) {
	if err := (*ResidentTrainingBackend)(nil).loadSharedCompletedSnapshot(context.Background(), nil); err == nil {
		t.Fatal("nil owner")
	}
	for _, kind := range []string{"terminal", "closed", "scope", "state", "trainer", "header", "negativeV"} {
		t.Run(kind, func(t *testing.T) {
			r, b := sharedCompletedCheckpointFixture(t)
			copyBackend := *b
			switch kind {
			case "terminal":
				copyBackend.terminal = errors.New("fixture terminal")
			case "closed":
				copyBackend.closed = true
			case "scope":
				copyBackend.Scope = "vulkan-resident-hybrid-v1"
			case "state":
				copyBackend.state = nil
			case "trainer":
				copyBackend.trainer = nil
			case "header":
				copyBackend.host.Version = 0
			case "negativeV":
				copyBackend.host.Adam = append([]NamedAdamState(nil), b.host.Adam...)
				m := copyBackend.host.Adam[0]
				m.V = append([]float32(nil), m.V...)
				m.V[0] = -1
				copyBackend.host.Adam[0] = m
			}
			before := mustFullState(t, r.trainer)
			if err := copyBackend.loadSharedCompletedSnapshot(context.Background(), r.trainer); err == nil {
				t.Fatal("bad completed source/owner admitted")
			}
			if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) {
				t.Fatal("source admission wrote live")
			}
		})
	}
}

func TestPocketSharedInitialSnapshotNativeAdmissionLimits(t *testing.T) {
	r := requireRunner(t)
	for _, kind := range []string{"negativeV", "long-name"} {
		t.Run(kind, func(t *testing.T) {
			host := mustFullState(t, r.trainer)
			for _, p := range host.Params {
				host.Adam = append(host.Adam, NamedAdamState{Name: p.Name, M: make([]float32, len(p.Values)), V: make([]float32, len(p.Values))})
			}
			if kind == "negativeV" {
				host.Adam[0].V[0] = -1
			} else {
				name := string(make([]byte, 513))
				old := host.Params[0].Name
				host.Params[0].Name = name
				for i := range host.Adam {
					if host.Adam[i].Name == old {
						host.Adam[i].Name = name
					}
				}
				for i := range host.EMA {
					if host.EMA[i].Name == old {
						host.EMA[i].Name = name
					}
				}
			}
			if err := host.Validate(); err != nil {
				t.Fatal("fixture must reach native snapshot validator", err)
			}
			if _, err := sharedInitialSnapshot(host); err == nil {
				t.Fatal("native snapshot limit bypass")
			}
		})
	}
	// F32 invalid header still rejects before building native array views.
	host := mustFullState(t, r.trainer)
	host.AdamW.Epsilon = float32(math.NaN())
	if _, err := sharedInitialSnapshot(host); err == nil {
		t.Fatal("nonfinite config")
	}
}
