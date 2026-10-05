package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"strings"
	"testing"
)

func TestTrainingFusedBoundaryDecoders(t *testing.T) {
	m := tinyTrainableFlowHead()
	primary := primaryFixture()
	distill := distillFixture(t, m, .3, .8)
	forward := DistillForwardInputs{Condition: primary.Condition, Target: primary.Target, Noise: primary.Noise, S: .3, T: .8}
	for _, kind := range []string{"diagonal", "primary", "distill", "forward"} {
		t.Run(kind, func(t *testing.T) {
			var values [][]float32
			var decode func([][]float32) error
			switch kind {
			case "diagonal":
				p, err := m.BuildDiagonalTrainingTape(primary.Condition, .3, primary.Input, []float32{.1, .2}, primary.Target, .2, .25, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				values = interpretFlowTape(t, p.Tape)[:3]
				decode = func(v [][]float32) error {
					_, err := decodeDiagonalBoundary(v, primary.Condition, primary.Input, []float32{.1, .2}, primary.Target)
					return err
				}
			case "primary":
				p, err := m.BuildPrimaryTrainingTape(primary, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				values = interpretFlowTape(t, p.Tape)[:4]
				decode = func(v [][]float32) error { _, err := decodePrimaryBoundary(v, primary); return err }
			case "distill":
				p, err := m.BuildDistillTrainingTape(distill, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				values = interpretFlowTape(t, p.Tape)[:5]
				decode = func(v [][]float32) error { _, err := decodeDistillBoundary(v, distill); return err }
			case "forward":
				p, err := m.BuildDistillForwardTape(forward, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				values = interpretFlowTape(t, p.Tape)[:6]
				decode = func(v [][]float32) error { _, err := decodeDistillForwardBoundary(v, forward); return err }
			}
			if err := decode(values); err != nil {
				t.Fatal(err)
			}
			for _, bad := range []string{"count", "shape", "nonfinite", "alias", "borrowed", "stats"} {
				v := make([][]float32, len(values))
				for i, x := range values {
					v[i] = append([]float32(nil), x...)
				}
				switch bad {
				case "count":
					v = v[:len(v)-1]
				case "shape":
					v[len(v)-1] = nil
				case "nonfinite":
					v[0][0] = float32(math.NaN())
				case "alias":
					v[1] = v[0][:len(v[1])]
				case "borrowed":
					v[0] = primary.Condition
					if kind == "distill" {
						v[0] = distill.Condition
					}
					if kind == "forward" {
						v[0] = forward.Target
					}
				case "stats":
					if kind != "diagonal" && kind != "distill" {
						continue
					}
					v[len(v)-1][0] = -1
				}
				if err := decode(v); err == nil {
					t.Fatal(bad)
				}
			}
		})
	}
}
func TestTrainingForwardCompositionEdges(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	zero := make([]float32, len(in.Input))
	p, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{.3, .8}, in.Input, 1, zero, zero, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"input", "inventory", "extent"} {
		t.Run(kind, func(t *testing.T) {
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{len(in.Input)}, Values: append([]float32(nil), in.Input...)}}, Outputs: []int{0}}
			var bindings map[string]train.ResidentTapeBinding
			if kind == "input" {
				tape.Tensors[0] = train.TensorSpec{Shape: []int{1}, Values: []float32{0}}
			} else {
				bindings = map[string]train.ResidentTapeBinding{}
				if kind == "extent" {
					for name := range p.ResidentBindings {
						bindings[name] = train.ResidentTapeBinding{Parameter: 0, Gradient: 0}
					}
				}
			}
			if _, _, err := appendAdmittedFlowGraph(&tape, p, 0, bindings); err == nil {
				t.Fatal(kind)
			}
		})
	}
}
func TestTrainingBackboneMissingGradient(t *testing.T) {
	_, m := backboneOracle(t)
	batch := eosForwardBatch(2, 1, []uint32{1, 2})
	rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
	_, err := m.buildBackboneTrainingTapeSeed(batch, make([]float32, rows*m.Hidden), 64<<20, false, func(g *transformerTapeGraph, output, seed int) (int, map[string]train.ResidentTapeBinding) {
		g.param("unmatched.internal.parameter", []int{1}, []float32{0})
		return seed, nil
	})
	if err == nil || !strings.Contains(err.Error(), "missing backbone gradient") {
		t.Fatal("missing gradient inventory", err)
	}
}

func TestTrainingBackboneCompositionAdmission(t *testing.T) {
	for _, kind := range []string{"graph", "binding"} {
		t.Run(kind, func(t *testing.T) {
			_, m := backboneOracle(t)
			batch := eosForwardBatch(2, 1, []uint32{1, 2})
			rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
			_, err := m.buildBackboneTrainingTapeSeed(batch, make([]float32, rows*m.Hidden), 64<<20, false, func(g *transformerTapeGraph, output, seed int) (int, map[string]train.ResidentTapeBinding) {
				if kind == "graph" {
					for i := 0; i < 4096; i++ {
						g.uniqueZero(1)
					}
					return seed, nil
				}
				parameter := g.param("invalid.composition.binding", []int{2}, []float32{0, 0})
				gradient := g.zero(1)
				return seed, map[string]train.ResidentTapeBinding{"invalid.composition.binding": {Parameter: parameter, Gradient: gradient}}
			})
			want := "graph ceiling"
			if kind == "binding" {
				want = "binding constant/gradient extent"
			}
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatal("invalid composition", err)
			}
		})
	}
}
