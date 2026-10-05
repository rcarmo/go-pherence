package pockettts

import (
	"context"
	"reflect"
	"testing"
)

func TestPocketGroupedLocalLivePGAliasAdmission(t *testing.T) {
	for _, kind := range []string{"same-name", "cross-name", "partial-same-name", "partial-cross-name", "disjoint"} {
		t.Run(kind, func(t *testing.T) {
			tape, bindings, results, policy := pocketGroupTiny()
			// Two-element parameters make the partial cases real overlaps of
			// unequal slice starts, not another exact-slice alias.
			for _, id := range []int{0, 1, 2, 3} {
				tape.Tensors[id].Shape = []int{2}
			}
			tape.Tensors[0].Values = []float32{2, 4}
			tape.Tensors[1].Values = []float32{3, 5}
			p := newPocketGroup(tape, bindings)
			p.state.g["a"] = []float32{.25, -.5}
			p.state.g["b"] = []float32{.75, -1}
			p.state.ready["a"] = true
			p.state.ready["b"] = false
			var backing []float32
			switch kind {
			case "same-name":
				p.state.g["a"] = p.state.p["a"]
			case "cross-name":
				p.state.g["a"] = p.state.p["b"]
			case "partial-same-name":
				backing = []float32{2, 4, 7}
				p.state.p["a"] = backing[:2:2]
				p.state.g["a"] = backing[1:3:3]
			case "partial-cross-name":
				backing = []float32{3, 5, 7}
				p.state.p["b"] = backing[:2:2]
				p.state.g["a"] = backing[1:3:3]
			case "disjoint":
				// Adjacent cap-limited views share an allocation but not elements.
				backing = []float32{2, 4, .25, -.5}
				p.state.p["a"] = backing[:2:2]
				p.state.g["a"] = backing[2:4:4]
			}
			beforeP, beforeG := map[string][]float32{}, map[string][]float32{}
			for name, v := range p.state.p {
				beforeP[name] = append([]float32(nil), v...)
			}
			for name, v := range p.state.g {
				beforeG[name] = append([]float32(nil), v...)
			}
			beforeReady := map[string]bool{"a": true, "b": false}
			beforeBacking := append([]float32(nil), backing...)
			gA, gB := &p.state.g["a"][0], &p.state.g["b"][0]
			callbacks, dispatched, published := 0, 0, 0
			p.callback = func(stage string, group, item int) {
				callbacks++
				if stage == "dispatch" {
					dispatched++
				}
				if stage == "publish" {
					published++
				}
			}
			out, err := p.run(context.Background(), tape, bindings, nil, results, [][]string{{"a"}, {"b"}}, 1, policy)
			if !reflect.DeepEqual(beforeP, p.state.p) {
				t.Fatal("live P mutated", kind, beforeP, p.state.p)
			}
			if p.state.active || p.state.terminal != nil || p.state.closed || len(p.state.retained) != 0 || p.live != 0 {
				t.Fatal("admission/success left state fenced or retained", kind)
			}
			if kind != "disjoint" {
				if err == nil || err.Error() != "group live G/P alias" || out != nil {
					t.Fatal("live P/G overlap admitted", kind, err, out)
				}
				if !reflect.DeepEqual(beforeG, p.state.g) || !reflect.DeepEqual(beforeReady, p.state.ready) || !reflect.DeepEqual(beforeBacking, backing) {
					t.Fatal("rejection mutated G/readiness/backing", kind)
				}
				if p.ledger.FreezeCopies != 0 || p.ledger.OperationFrozen != 0 || p.ledger.FrozenP != 0 || p.ledger.Plans != 0 || p.ledger.Commands != 0 || p.ledger.Constructed != 0 || len(p.ledger.Groups) != 0 || p.ledger.CandidatePayload != 0 || p.ledger.PublishedPayload != 0 || callbacks != 0 || dispatched != 0 || published != 0 {
					t.Fatal("overlap rejected after copying/planning/execution/publication", kind, p.ledger)
				}
			} else {
				if err != nil || len(out) != 1 || out[0][0] != .75 || !p.state.ready["a"] || !p.state.ready["b"] || gA != &p.state.g["a"][0] || gB != &p.state.g["b"][0] || published != 1 || dispatched != 2 || p.ledger.Closed != 2 {
					t.Fatal("disjoint accumulation/lifetime", err, p.ledger)
				}
				var stats pocketAdapterErrors
				stats.compare(t, "disjoint A prior + gradient", p.state.g["a"], []float32{8.25, 13.5}, 2e-6)
				stats.compare(t, "disjoint B prior + gradient", p.state.g["b"], []float32{9.75, 16}, 2e-6)
			}
			t.Logf("%s live P preserved; freezes=%d plans=%d construct=%d dispatch=%d publish=%d", kind, p.ledger.FreezeCopies, p.ledger.Plans, p.ledger.Constructed, dispatched, published)
		})
	}
}
