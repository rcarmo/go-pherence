//go:build linux

package pockettts

import (
	"context"
	"strings"
	"testing"
)

func TestTrainingResidentFusedBoundaryRetention(t *testing.T) {
	for _, kind := range []string{"diagonal", "primary", "distill", "forward", "backbone", "stepweight", "stepbackbone"} {
		for _, bad := range []string{"alias", "borrowed"} {
			t.Run(kind+"/"+bad, func(t *testing.T) {
				r, b, d := residentGraphRunner(t, "ptx-resident-eosforward-hybrid-v1")
				m := r.trainer.Flow
				in := residentDistillFixture(t, m, .3, .8)
				primary := primaryFixture()
				primary.Condition = append([]float32(nil), in.Condition...)
				forward := DistillForwardInputs{Condition: in.Condition, Target: in.Target, Noise: in.Noise, S: .3, T: .8}
				_, _, batch, hidden, reduction, samples := stepBackboneFixture(t, 1)
				rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
				seed := make([]float32, rows*r.trainer.FlowLM.Hidden)
				if kind == "stepbackbone" {
					b.Scope = "ptx-resident-stepbackbone-hybrid-v1"
				} else if kind == "stepweight" {
					b.Scope = "ptx-resident-stepweight-hybrid-v1"
				}
				if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
					t.Fatal(err)
				}
				d.afterDownload = func(v [][]float32) {
					if bad == "alias" {
						switch kind {
						case "backbone":
							v[1] = v[0]
						case "stepweight", "stepbackbone":
							v[3] = v[2]
						default:
							v[1] = v[0][:len(v[1])]
						}
					} else {
						switch kind {
						case "backbone":
							v[0] = seed
						case "stepweight", "stepbackbone":
							copy(samples.DiagonalTime, v[2]) // preserve padding before ownership rejection
							v[2] = samples.DiagonalTime
						case "primary":
							v[0] = primary.Condition
						case "forward":
							v[0] = in.Target
						default:
							v[0] = in.Condition
						}
					}
				}
				var err error
				switch kind {
				case "diagonal":
					_, err = b.EvaluateDiagonal(m, in.Condition, .3, in.PrimaryInput, in.Prediction, in.Desired, .2, .25)
				case "primary":
					_, err = b.EvaluatePrimary(m, primary, .25)
				case "distill":
					_, err = b.EvaluateDistill(m, in, .25)
				case "forward":
					_, err = b.EvaluateDistillForward(m, forward)
				case "backbone":
					_, err = b.EvaluateBackbone(r.trainer.FlowLM, batch, seed, false)
				case "stepweight":
					_, err = b.evaluateStepWeight(m, r.trainer.Weighting, reduction, DefaultTrainingStepConfig(), samples)
				case "stepbackbone":
					_, err = b.evaluateStepBackbone(r.trainer.FlowLM, m, r.trainer.Weighting, batch, hidden, reduction, DefaultTrainingStepConfig(), samples)
				}
				if err == nil || b.terminal == nil {
					t.Fatal("bad output admitted", err)
				}
				// Prove the decoder, not executor shape/finite admission, rejected ownership.
				if !strings.Contains(err.Error(), "alias") && !strings.Contains(err.Error(), "overlap") && !strings.Contains(err.Error(), "borrowed") {
					t.Fatal("wrong rejection", err)
				}
				tapes := d.tapes
				if b.PrepareTrainingStep(context.Background(), r.trainer) == nil || d.tapes != tapes {
					t.Fatal("retry")
				}
				if b.ValidateBoundary(r.trainer) == nil || b.Close() == nil {
					t.Fatal("uncertain owner released")
				}
			})
		}
	}
}
