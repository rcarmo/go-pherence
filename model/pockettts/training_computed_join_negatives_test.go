package pockettts

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func TestPocketComputedJoinContributionAdmission(t *testing.T) {
	f, _, _ := pocketJoinInventory()
	clone := func(m map[string][]float32) map[string][]float32 {
		n := map[string][]float32{}
		for k, v := range m {
			n[k] = append([]float32(nil), v...)
		}
		return n
	}
	base := map[string][]float32{}
	totalBase := map[string][]float32{}
	for n, z := range f {
		base[n] = make([]float32, z)
		totalBase[n] = make([]float32, z)
		for i := range base[n] {
			base[n][i] = .25
			totalBase[n][i] = .125
		}
	}
	key := "flow_lm.flow_net.cond_embed.bias"
	for _, kind := range []string{"missing", "foreign", "extent", "nonfinite", "scale", "total-extent", "sum-overflow"} {
		t.Run(kind, func(t *testing.T) {
			got, total := clone(base), clone(totalBase)
			scale := float32(.125)
			switch kind {
			case "missing":
				delete(got, key)
			case "foreign":
				got["flow_lm.flow_net.emb_mean"] = got[key]
				delete(got, key)
			case "extent":
				got[key] = got[key][:3]
			case "nonfinite":
				got[key][0] = float32(math.NaN())
			case "scale":
				scale = float32(math.NaN())
			case "total-extent":
				total[key] = total[key][:3]
			case "sum-overflow":
				got[key][0] = math.MaxFloat32
				total[key][0] = math.MaxFloat32
				scale = 1
			}
			before := clone(total)
			if e := pocketJoinAccumulate(total, got, f, scale); e == nil {
				t.Fatal("invalid contribution admitted", kind)
			}
			if !reflect.DeepEqual(total, before) {
				t.Fatal("rejected contribution wrote private candidates")
			}
		})
	}
	total := clone(totalBase)
	if e := pocketJoinAccumulate(total, base, f, .125); e != nil {
		t.Fatal(e)
	}
	for _, v := range total {
		for _, z := range v {
			if z != .15625 {
				t.Fatal("single scale/previous candidate lost")
			}
		}
	}
	if _, e := pocketJoinUnion(f, map[string]int{key: 4}); e == nil {
		t.Fatal("duplicate later contribution admitted")
	}
}

func TestPocketComputedJoinGradientRootAdmission(t *testing.T) {
	// Metadata-only extraction negatives; no alternative full-step config/oracle.
	inv := map[string]int{"flow_lm.flow_net.cond_embed.bias": 4}
	key := "flow_lm.flow_net.cond_embed.bias"
	base := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{4}, Values: []float32{1, 2, 3, 4}}, {Shape: []int{4}}, {Shape: []int{4}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 0}, Output: 1, Scale: 1}, {Op: "vector", Inputs: []int{1, 0}, Output: 2, Scale: 1}}, Outputs: []int{1, 2}}
	for _, kind := range []string{"missing", "foreign", "duplicate-root", "duplicate-P", "undeclared", "misassigned", "selected-duplicate", "selected-missing", "output-count", "nonfinite", "extent"} {
		t.Run(kind, func(t *testing.T) {
			tape := pocketAdapterFreeze(base)
			bindings := map[string]train.ResidentTapeBinding{key: {Parameter: 0, Gradient: 1}}
			out := [][]float32{{2, 4, 6, 8}, {3, 6, 9, 12}}
			switch kind {
			case "missing":
				delete(bindings, key)
			case "foreign":
				bindings["foreign"] = bindings[key]
				delete(bindings, key)
			case "duplicate-root":
				tape.Outputs = []int{1, 1}
			case "duplicate-P":
				bindings["foreign"] = train.ResidentTapeBinding{Parameter: 0, Gradient: 2}
			case "undeclared":
				tape.Outputs = []int{2}
				out = out[:1]
			case "misassigned":
				bindings[key] = train.ResidentTapeBinding{Parameter: 0, Gradient: 0}
			case "output-count":
				out = out[:1]
			case "nonfinite":
				out[0][0] = float32(math.Inf(1))
			case "extent":
				out[0] = out[0][:3]
			}
			if strings.HasPrefix(kind, "selected-") {
				roots := []int{1, 1}
				if kind == "selected-missing" {
					roots = []int{9}
				}
				if _, e := pocketJoinSelected(tape, out, roots); e == nil {
					t.Fatal("invalid selected roots")
				}
				return
			}
			extents := map[string][]float32{key: make([]float32, 4)}
			e := adapterOracleInventory(tape, bindings, extents)
			if e == nil {
				_, e = pocketJoinGradients(tape, bindings, out, inv)
			}
			if e == nil {
				t.Fatal("invalid gradient contribution admitted", kind)
			}
		})
	}
}

func TestPocketComputedJoinBoundaryFailureDiscards(t *testing.T) {
	for _, kind := range []string{"shape", "nonfinite", "alias"} {
		t.Run(kind, func(t *testing.T) {
			x, _ := loadPocketJoinFixture(t)
			trace := &pocketJoinTrace{afterStage: func(name string, out [][]float32) {
				if name != "row1-distill-backward-primary-only" {
					return
				}
				switch kind {
				case "shape":
					out[0] = out[0][:3]
				case "nonfinite":
					out[0][0] = float32(math.NaN())
				case "alias":
					out[1] = out[0][:2]
				}
			}}
			r, e := computePocketJoin(t, context.Background(), x, trace)
			if e == nil || r != nil || len(trace.Contributions) != 3 || len(trace.Phases) != 11 {
				t.Fatal("late boundary failure published candidates/continued", e, trace.Contributions, len(trace.Phases))
			}
			t.Logf("%s rejected after three private flow contributions, no later graph/result: %v", kind, e)
		})
	}
}

func TestPocketComputedJoinLateCancellationDiscards(t *testing.T) {
	x, _ := loadPocketJoinFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	trace := &pocketJoinTrace{beforeStage: func(name string) {
		if name == "later-reduction-weight-eos-backbone-finalinput" {
			cancel()
		}
	}}
	r, e := computePocketJoin(t, ctx, x, trace)
	if !errors.Is(e, context.Canceled) || r != nil || len(trace.Contributions) != 4 || len(trace.Phases) != 12 {
		t.Fatal("late cancellation published candidates/continued", e, trace.Contributions, len(trace.Phases))
	}
	last := trace.Phases[len(trace.Phases)-1]
	if last.Adapter.Dispatched != 0 || last.Adapter.SlotPayload != 0 {
		t.Fatal("cancelled later adapter dispatched", last)
	}
	t.Log("cancellation after all four private row contributions returns no metrics/G/result; no optimiser/checkpoint/update path exists")
}

func TestPocketComputedJoinOracleIsolation(t *testing.T) {
	x, o := loadPocketJoinFixture(t)
	// Corrupt every result category in the comparison-only object. Mutating its
	// P/input storage additionally proves the execution freeze owns its sources.
	for _, v := range o.Gradients {
		for i := range v {
			v[i] = float32(math.NaN())
		}
	}
	for _, v := range [][]float32{o.DiagonalLogVariance, o.DistillLogVariance, o.DDiagonalLogVariance, o.DDistillLogVariance, o.DNormalizedLatents, o.DVoiceLatents} {
		for i := range v {
			v[i] = float32(math.NaN())
		}
	}
	for n := range o.Metrics {
		o.Metrics[n] = math.NaN()
	}
	for _, v := range o.Parameters {
		for i := range v {
			v[i] = float32(math.NaN())
		}
	}
	o.NormalizedLatents[0] = float32(math.NaN())
	o.Noise[0] = float32(math.NaN())
	o.Mask[0] = false
	trace := &pocketJoinTrace{}
	r, e := computePocketJoin(t, context.Background(), x, trace)
	if e != nil || r == nil {
		t.Fatal("comparison-only arrays leaked into execution", e)
	}
	_, fresh := loadPocketJoinFixture(t)
	var stats pocketAdapterErrors
	for n, v := range r.G {
		stats.compare(t, n, v, fresh.Gradients[n], 8e-5)
	}
	stats.compare(t, "audio", r.Later.Inputs.NormalizedLatents, fresh.DNormalizedLatents, 5e-5)
	stats.compare(t, "voice", r.Later.Inputs.VoiceLatents, fresh.DVoiceLatents, 5e-5)
	stats.compare(t, "diagonal seed", r.Later.Reduction.DiagonalLogvar[:2], fresh.DDiagonalLogVariance, 5e-5)
	stats.compare(t, "distill seed", r.Later.Reduction.DistillLogvar[:2], fresh.DDistillLogVariance, 5e-5)
	m := r.Later.Reduction.Metrics
	for _, p := range [][2]float64{{m.FlowDiagonal, fresh.Metrics["flow_diag"]}, {m.FlowDistill, fresh.Metrics["flow_distill"]}, {m.FlowLoss, fresh.Metrics["flow_loss"]}, {m.EOS, fresh.Metrics["eos_loss"]}, {m.Loss, fresh.Metrics["loss"]}} {
		assertClose64(t, "isolated metric", p[0], p[1], 3e-5)
	}
	stats.log(t, "poisoned comparison storage, independently pinned final comparisons")
}
