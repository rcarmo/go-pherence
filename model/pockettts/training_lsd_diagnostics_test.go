//go:build linux

package pockettts

import (
	"context"
	"fmt"
	"testing"
)

var lsdDiagnosticsResultSink LSDDistillResult
var lsdDiagnosticsGradientSink LSDDistillGradients

// Complete tiny CPU/mock row, including tape building/pruning/execution and
// gradient accumulation. Preparation is outside the measured region. Run with
// -benchtime=20x for bounded samples; these timings do not measure native GPU.
func TestTrainingLSDPrivateDiagnosticsAllocation(t *testing.T) {
	for _, stepOnly := range []bool{false, true} {
		name := "public"
		if stepOnly {
			name = "private"
		}
		t.Run(name, func(t *testing.T) {
			r, backend, _ := residentGraphRunner(t, "ptx-resident-forwardresults-hybrid-v1")
			in := residentDistillFixture(t, r.trainer.Flow, .3, .8)
			if err := backend.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			// Warm builder/interpreter paths before measurements.
			if _, _, err := r.trainer.Flow.lsdDistillTrainingBackendResults(backend, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, 1, stepOnly); err != nil {
				t.Fatal(err)
			}
			result := testing.Benchmark(func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					result, g, err := r.trainer.Flow.lsdDistillTrainingBackendResults(backend, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, 1, stepOnly)
					if err != nil {
						b.Fatal(err)
					}
					lsdDiagnosticsResultSink = result
					lsdDiagnosticsGradientSink = g
				}
			})
			t.Logf("tiny complete %s row: %d B/op %d allocs/op %d ns/op", name, result.AllocedBytesPerOp(), result.AllocsPerOp(), result.NsPerOp())
		})
	}
}

func TestTrainingLSDPrivateDiagnosticsParity(t *testing.T) {
	for _, scope := range []string{"ptx-resident-distillresults-hybrid-v1", "vulkan-resident-distillresults-hybrid-v1", "ptx-resident-forwardresults-hybrid-v1", "vulkan-resident-forwardresults-hybrid-v1"} {
		for _, scale := range []float32{0, .25, 1} {
			t.Run(scope+"/"+fmt.Sprint(scale), func(t *testing.T) {
				public, pb, pd := residentGraphRunner(t, scope)
				private, b, d := residentGraphRunner(t, scope)
				m := private.trainer.Flow
				in := residentDistillFixture(t, m, .3, .8)
				if err := b.PrepareTrainingStep(context.Background(), private.trainer); err != nil {
					t.Fatal(err)
				}
				if err := pb.PrepareTrainingStep(context.Background(), public.trainer); err != nil {
					t.Fatal(err)
				}
				got, g, err := m.lsdDistillTrainingStepBackendScaled(b, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, scale)
				if err != nil {
					t.Fatal(err)
				}
				want, wg, err := public.trainer.Flow.lsdDistillTrainingBackendScaled(pb, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, scale)
				if err != nil {
					t.Fatal(err)
				}
				if got.Velocity != nil || got.TimeDerivative != nil || got.Endpoint != nil || got.Residual != nil || g.Noise != nil {
					t.Fatal("discarded diagnostics materialised")
				}
				assertClose64(t, "raw square", got.RawSquare, want.RawSquare, 2e-6)
				assertClose64(t, "loss", got.Loss, want.Loss, 2e-6)
				assertSliceClose(t, "raw condition", g.Condition, wg.Condition, 2e-5)
				assertSliceClose(t, "raw target", g.Target, wg.Target, 2e-5)
				if g.DLogVariance != wg.DLogVariance {
					t.Fatal("logvar")
				}
				diagnostics := [][]float32{want.Velocity, want.TimeDerivative, want.Endpoint, want.Residual}
				for i, x := range diagnostics {
					if len(x) != len(in.Target) || !finiteF32(x) {
						t.Fatal("public diagnostics")
					}
					for _, y := range diagnostics[:i] {
						if f32SlicesOverlap(x, y) {
							t.Fatal("public diagnostics alias")
						}
					}
					for _, source := range [][]float32{in.Condition, in.Target, in.Noise, wg.Condition, wg.Target, wg.Noise} {
						if f32SlicesOverlap(x, source) {
							t.Fatal("public borrowed diagnostic")
						}
					}
				}
				// Mark other roots ready without changing accumulated flow gradients.
				zeros := map[string][]float32{}
				for name, x := range private.trainer.params {
					zeros[name] = make([]float32, len(x))
				}
				if err := b.state.Accumulate(context.Background(), zeros, 0); err != nil {
					t.Fatal(err)
				}
				if err := pb.state.Accumulate(context.Background(), zeros, 0); err != nil {
					t.Fatal(err)
				}
				if _, err := b.state.Update(context.Background(), 1e20); err != nil {
					t.Fatal(err)
				}
				if _, err := pb.state.Update(context.Background(), 1e20); err != nil {
					t.Fatal(err)
				}
				for i, x := range d.validatedGradients {
					assertSliceClose(t, "primary-only roots", x, pd.validatedGradients[i], 2e-5)
				}
			})
		}
	}
}
func TestTrainingLSDPrivateDiagnosticsLegacy(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-forwardfused-hybrid-v1")
	in := residentDistillFixture(t, r.trainer.Flow, .3, .8)
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	got, g, err := r.trainer.Flow.lsdDistillTrainingStepBackendScaled(b, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Velocity) != len(in.Target) || len(got.TimeDerivative) != len(in.Target) || len(got.Endpoint) != len(in.Target) || len(got.Residual) != len(in.Target) || len(g.Noise) != len(in.Noise) {
		t.Fatal("legacy diagnostic policy changed")
	}
}
func TestTrainingLSDPrivateDiagnosticsFailure(t *testing.T) {
	for _, kind := range []string{"input", "forward", "backward"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-forwardresults-hybrid-v1")
			m := r.trainer.Flow
			in := residentDistillFixture(t, m, .3, .8)
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "input":
				in.Noise = nil
			case "forward":
				d.failTape = 1
			case "backward":
				d.afterDownload = func(v [][]float32) {
					if len(v) == 4 {
						v[2][0] = -1
					}
				}
			}
			got, _, err := m.lsdDistillTrainingStepBackendScaled(b, in.Condition, in.S, in.T, in.Noise, in.Target, in.LogVariance, 1)
			if err == nil || got.Velocity != nil || got.TimeDerivative != nil || got.Endpoint != nil || got.Residual != nil {
				t.Fatal("bad result")
			}
			if kind != "input" {
				if b.terminal == nil || b.Close() == nil || b.ValidateBoundary(r.trainer) == nil {
					t.Fatal("failure released")
				}
			}
		})
	}
}
