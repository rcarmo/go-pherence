package pockettts

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"reflect"
	"strings"
	"testing"
)

func randomRecipeFixture() TrainingRandomRecipe {
	return TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: runIdentityFixture().OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}
}
func randomLimitsFixture() TrainingRandomLimits {
	return TrainingRandomLimits{TrainingSampleLimits{4, 12}, 64}
}
func randomProposalFixture(t *testing.T, state *TrainingRunState) *TrainingRowProposal {
	t.Helper()
	p, err := state.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func rejectRandomUnchanged(t *testing.T, dst TrainingStepSamples, p *TrainingRowProposal, frames, valid, channels int, recipe TrainingRandomRecipe, limits TrainingRandomLimits) {
	t.Helper()
	mask := append([]bool(nil), dst.Mask...)
	var before [4][]uint32
	for i, values := range sampleFloatOutputs(dst) {
		before[i] = sampleBits(values)
	}
	var nextBefore, parentBefore trainingRunCore
	if p != nil {
		nextBefore = p.next.clone()
		if p.parent != nil {
			parentBefore = p.parent.core.clone()
		}
	}
	if err := SampleTrainingInto(dst, p, frames, valid, channels, recipe, limits); err == nil {
		t.Fatal("accepted malformed random admission")
	}
	if !reflect.DeepEqual(mask, append([]bool(nil), dst.Mask...)) {
		t.Fatal("admission mutated mask")
	}
	for i, values := range sampleFloatOutputs(dst) {
		if !reflect.DeepEqual(before[i], sampleBits(values)) {
			t.Fatal("admission mutated destination")
		}
	}
	if p != nil && (!reflect.DeepEqual(nextBefore, p.next) || (p.parent != nil && !reflect.DeepEqual(parentBefore, p.parent.core))) {
		t.Fatal("admission mutated proposal/parent")
	}
}

func TestTrainingRandomAdmission(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                    string
		frames, valid, channels int
		limits                  TrainingRandomLimits
	}{
		{"frames_zero", 0, 1, 2, randomLimitsFixture()},
		{"frames_negative", -1, 1, 2, randomLimitsFixture()},
		{"valid_zero", 3, 0, 2, randomLimitsFixture()},
		{"valid_negative", 3, -1, 2, randomLimitsFixture()},
		{"valid_exceeds", 3, 4, 2, randomLimitsFixture()},
		{"channel_negative", 3, 2, -1, randomLimitsFixture()},
		{"channel_zero", 3, 2, 0, randomLimitsFixture()},
		{"channel_one", 3, 2, 1, randomLimitsFixture()},
		{"frames_limit_negative", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{-1, 12}, 64}},
		{"noise_limit_negative", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{4, -1}, 64}},
		{"frames_limit_zero", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{0, 12}, 64}},
		{"noise_limit_zero", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{4, 0}, 64}},
		{"raw_limit_zero", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{4, 12}, 0}},
		{"raw_limit_negative", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{4, 12}, -1}},
		{"frames_limit_exceeded", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{2, 12}, 64}},
		{"noise_limit_exceeded", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{4, 5}, 64}},
		{"raw_limit_exceeded", 3, 2, 2, TrainingRandomLimits{TrainingSampleLimits{4, 12}, 19}},
		{"element_overflow", maxInt, 1, 2, TrainingRandomLimits{TrainingSampleLimits{maxInt, maxInt}, maxInt}},
		{"byte_overflow", maxInt/8 + 1, 1, 2, TrainingRandomLimits{TrainingSampleLimits{maxInt, maxInt}, maxInt}},
		{"total_word_overflow", maxInt / 8, maxInt / 8, 2, TrainingRandomLimits{TrainingSampleLimits{maxInt, maxInt}, maxInt}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := randomProposalFixture(t, newRunFixture(t, 3))
			rejectRandomUnchanged(t, sampleBuffers(3, 2), p, tc.frames, tc.valid, tc.channels, randomRecipeFixture(), tc.limits)
		})
	}
	for _, tc := range []struct {
		name   string
		mutate func(*TrainingRandomRecipe)
	}{
		{"algorithm", func(r *TrainingRandomRecipe) { r.Algorithm += "-other" }},
		{"digest_format", func(r *TrainingRandomRecipe) { r.OptionsSHA256 = "bad" }},
		{"digest_mismatch", func(r *TrainingRandomRecipe) { r.OptionsSHA256 = strings.Repeat("a", 64) }},
		{"upstream_skip", func(r *TrainingRandomRecipe) { r.DistillProbability = .25 }},
		{"skip_nan", func(r *TrainingRandomRecipe) { r.DistillProbability = float32(math.NaN()) }},
		{"text_dropout", func(r *TrainingRandomRecipe) { r.TextDropout = .2 }},
		{"voice_dropout", func(r *TrainingRandomRecipe) { r.VoiceDropout = .2 }},
		{"multiplier", func(r *TrainingRandomRecipe) { r.FlowBatchMultiplier = 2 }},
		{"classic", func(r *TrainingRandomRecipe) { r.StopGradient = "classic" }},
		{"normalize", func(r *TrainingRandomRecipe) { r.Normalize = false }},
		{"p_equal", func(r *TrainingRandomRecipe) { r.PEqual = 1 }},
		{"eos_weight", func(r *TrainingRandomRecipe) { r.EOSLossWeight = 0 }},
		{"logit_mean", func(r *TrainingRandomRecipe) { r.LogitMean = 0 }},
		{"logit_std", func(r *TrainingRandomRecipe) { r.LogitStd = 2 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := randomRecipeFixture()
			tc.mutate(&r)
			rejectRandomUnchanged(t, sampleBuffers(3, 2), randomProposalFixture(t, newRunFixture(t, 3)), 3, 2, 2, r, randomLimitsFixture())
		})
	}
	for field := 0; field < 4; field++ {
		for _, offset := range []int{-1, 1} {
			t.Run(fmt.Sprintf("shape_%d_%+d", field, offset), func(t *testing.T) {
				dst := sampleBuffers(3, 2)
				setSampleOutput(&dst, field, make([]float32, len(sampleFloatOutputs(dst)[field])+offset))
				rejectRandomUnchanged(t, dst, randomProposalFixture(t, newRunFixture(t, 3)), 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
			})
		}
	}
	t.Run("mask_shape", func(t *testing.T) {
		dst := sampleBuffers(3, 2)
		dst.Mask = nil
		rejectRandomUnchanged(t, dst, randomProposalFixture(t, newRunFixture(t, 3)), 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
	})
	for a := 0; a < 4; a++ {
		for b := a + 1; b < 4; b++ {
			for shift := 0; shift < 2; shift++ {
				t.Run(fmt.Sprintf("alias_%d_%d_shift_%d", a, b, shift), func(t *testing.T) {
					dst, storage := sampleBuffers(3, 2), make([]float32, 8)
					la, lb := len(sampleFloatOutputs(dst)[a]), len(sampleFloatOutputs(dst)[b])
					setSampleOutput(&dst, a, storage[:la])
					setSampleOutput(&dst, b, storage[shift:shift+lb])
					rejectRandomUnchanged(t, dst, randomProposalFixture(t, newRunFixture(t, 3)), 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
				})
			}
		}
	}
}

func TestTrainingRandomProposalAdmission(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*TrainingRowProposal)
	}{
		{"parent_nil", func(p *TrainingRowProposal) { p.parent = nil }},
		{"committed", func(p *TrainingRowProposal) { p.committed = true }},
		{"parent_core", func(p *TrainingRowProposal) { p.parent.core.order[0] = 99 }},
		{"parent_rng", func(p *TrainingRowProposal) { p.parent.core.timeRNG[0] = 0 }},
		{"parent_step", func(p *TrainingRowProposal) { p.parentStep++ }},
		{"identity", func(p *TrainingRowProposal) { p.next.identity.OptionsSHA256 = strings.Repeat("a", 64) }},
		{"rows", func(p *TrainingRowProposal) { p.next.rows++ }},
		{"step", func(p *TrainingRowProposal) { p.next.step++ }},
		{"cursor", func(p *TrainingRowProposal) { p.next.cursor = 0 }},
		{"epoch", func(p *TrainingRowProposal) { p.next.epoch++ }},
		{"row", func(p *TrainingRowProposal) { p.row = 99 }},
		{"row_valid_but_wrong", func(p *TrainingRowProposal) { p.row = (p.row + 1) % 3 }},
		{"duplicate_order", func(p *TrainingRowProposal) { p.next.order[1] = p.next.order[0] }},
		{"next_rng_encoding", func(p *TrainingRowProposal) { p.next.orderRNG[0] = 0 }},
		{"order", func(p *TrainingRowProposal) { p.next.order[0], p.next.order[1] = p.next.order[1], p.next.order[0] }},
		{"order_rng", func(p *TrainingRowProposal) { p.next.orderRNG[5] ^= 1 }},
		{"time_consumed", func(p *TrainingRowProposal) { _, _ = p.TimeUint64() }},
		{"noise_consumed", func(p *TrainingRowProposal) { _, _ = p.NoiseUint64() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := randomProposalFixture(t, newRunFixture(t, 3))
			tc.mutate(p)
			rejectRandomUnchanged(t, sampleBuffers(3, 2), p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
		})
	}
	t.Run("nil_and_zero_value", func(t *testing.T) {
		for _, p := range []*TrainingRowProposal{nil, {}} {
			rejectRandomUnchanged(t, sampleBuffers(3, 2), p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
		}
	})
	t.Run("stale_after_peer_commit", func(t *testing.T) {
		s := newRunFixture(t, 3)
		p, q := randomProposalFixture(t, s), randomProposalFixture(t, s)
		if err := s.CommitRow(q, 1); err != nil {
			t.Fatal(err)
		}
		rejectRandomUnchanged(t, sampleBuffers(3, 2), p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
	})
	t.Run("second_use", func(t *testing.T) {
		p, dst := randomProposalFixture(t, newRunFixture(t, 3)), sampleBuffers(3, 2)
		if err := SampleTrainingInto(dst, p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture()); err != nil {
			t.Fatal(err)
		}
		rejectRandomUnchanged(t, dst, p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
	})
	t.Run("rollover_fabrication", func(t *testing.T) {
		s := newRunFixture(t, 3)
		for step := uint64(1); step <= 3; step++ {
			if err := s.CommitRow(randomProposalFixture(t, s), step); err != nil {
				t.Fatal(err)
			}
		}
		p := randomProposalFixture(t, s)
		p.next.orderRNG[5] ^= 1
		rejectRandomUnchanged(t, sampleBuffers(3, 2), p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
	})
}

func TestTrainingRandomUniformEndpoints(t *testing.T) {
	for _, tc := range []struct {
		name string
		a, b uint64
		u, v float64
	}{
		{"zero", 0, 0, 0x1p-53, 0},
		{"max", ^uint64(0), ^uint64(0), 1 - 0x1p-53, 1 - 0x1p-53},
		{"quarter_angle", 0, uint64(1) << 62, 0x1p-53, .25},
		{"half_angle", 0, uint64(1) << 63, 0x1p-53, .5},
		{"last_low_bits", 4095, 2047, 0x1p-53, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			u, v := trainingUniforms(tc.a, tc.b)
			if u != tc.u || v != tc.v || !(u > 0 && u < 1) || !(v >= 0 && v < 1) {
				t.Fatalf("discrete uniform mapping %g %g", u, v)
			}
		})
	}
	// Exact dyadic mapping only. No transcendental expected values are derived
	// from the candidate; BOTH Box-Muller and sigmoid require independent oracles.
}

func TestTrainingRandomDrawCountsOwnership(t *testing.T) {
	for _, tc := range []struct{ frames, valid, channels int }{{1, 1, 2}, {4, 1, 3}, {4, 3, 3}, {3, 3, 2}} {
		t.Run(fmt.Sprintf("%d_%d_%d", tc.frames, tc.valid, tc.channels), func(t *testing.T) {
			s, dst := newRunFixture(t, 3), sampleBuffers(tc.frames, tc.channels)
			p, before := randomProposalFixture(t, s), snapshotRunFixture(t, s)
			var phaseLog []string
			if err := sampleTrainingRandomInto(dst, p, tc.frames, tc.valid, tc.channels, randomRecipeFixture(), randomLimitsFixture(), func(phase string, index int) error {
				phaseLog = append(phaseLog, fmt.Sprintf("%s:%d", phase, index))
				return nil
			}); err != nil {
				t.Fatal(err)
			}
			noiseRef, timeRef := rand.NewPCG(5, 6), rand.NewPCG(3, 4)
			for i := 0; i < 2*tc.valid*tc.channels; i++ {
				noiseRef.Uint64()
			}
			for i := 0; i < 6*tc.valid; i++ {
				timeRef.Uint64()
			}
			if p.next.noiseRNG != pcgState(noiseRef) || p.next.timeRNG != pcgState(timeRef) || !bytes.Equal(before, snapshotRunFixture(t, s)) {
				t.Fatal("raw phase counts/parent changed")
			}
			var expected []string
			for _, phase := range []string{"noise", "diagonal", "pair"} {
				n := tc.valid
				if phase == "noise" {
					n *= tc.channels
				}
				for i := 0; i < n; i++ {
					expected = append(expected, fmt.Sprintf("%s:%d", phase, i))
				}
			}
			if !reflect.DeepEqual(phaseLog, expected) {
				t.Fatal("phase scheduling changed")
			}
			for row, valid := range dst.Mask {
				if valid != (row < tc.valid) {
					t.Fatal("prefix mask changed")
				}
			}
			for i, output := range sampleFloatOutputs(dst) {
				n := tc.valid
				if i == 0 {
					n *= tc.channels
				}
				for _, x := range output[:n] {
					if !isFinite(x) || (i > 0 && (x < 0 || x > 1)) {
						t.Fatal("invalid transform domain")
					}
				}
				for _, x := range output[n:] {
					if math.Float32bits(x) != 0 {
						t.Fatal("padding is not positive zero")
					}
				}
			}
			for row := 0; row < tc.valid; row++ {
				if dst.DistillS[row] > dst.DistillT[row] {
					t.Fatal("pair not ordered")
				}
			}
			// Domain/count checks do not qualify numeric transforms/distributions.
		})
	}
	t.Run("bounded_arena_guards", func(t *testing.T) {
		dst := sampleBuffers(3, 2)
		var arena [4][]float32
		for i, values := range sampleFloatOutputs(dst) {
			arena[i] = make([]float32, len(values)+2)
			for j := range arena[i] {
				arena[i][j] = -9
			}
			setSampleOutput(&dst, i, arena[i][1:len(arena[i])-1])
		}
		mask := []bool{true, false, false, false, true}
		dst.Mask = mask[1:4]
		if err := SampleTrainingInto(dst, randomProposalFixture(t, newRunFixture(t, 3)), 3, 2, 2, randomRecipeFixture(), randomLimitsFixture()); err != nil {
			t.Fatal(err)
		}
		for _, storage := range arena {
			if storage[0] != -9 || storage[len(storage)-1] != -9 {
				t.Fatal("float guard changed")
			}
		}
		if !mask[0] || !mask[4] {
			t.Fatal("mask guard changed")
		}
	})
}

func TestTrainingRandomAbortResume(t *testing.T) {
	injected := errors.New("injected postdraw failure")
	for _, stop := range []string{"noise", "diagonal", "pair"} {
		t.Run(stop, func(t *testing.T) {
			s := newRunFixture(t, 3)
			before, p, dst := snapshotRunFixture(t, s), randomProposalFixture(t, s), sampleBuffers(3, 2)
			err := sampleTrainingRandomInto(dst, p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture(), func(phase string, index int) error {
				if phase == stop && index == 0 {
					return injected
				}
				return nil
			})
			if !errors.Is(err, injected) || !bytes.Equal(before, snapshotRunFixture(t, s)) || p.next.noiseRNG == s.core.noiseRNG {
				t.Fatal("postdraw failure did not isolate parent")
			}
			rejectRandomUnchanged(t, dst, p, 3, 2, 2, randomRecipeFixture(), randomLimitsFixture())
			fresh, peer := randomProposalFixture(t, s), randomProposalFixture(t, newRunFixture(t, 3))
			if !reflect.DeepEqual(fresh.next, peer.next) {
				t.Fatal("abandon changed fresh proposal")
			}
		})
	}
	t.Run("metadata_resume_and_rollover", func(t *testing.T) {
		a, b := newRunFixture(t, 3), newRunFixture(t, 3)
		for step := uint64(1); step <= 8; step++ {
			valid := int(step%3) + 1
			for _, state := range []*TrainingRunState{a, b} {
				p := randomProposalFixture(t, state)
				if err := SampleTrainingInto(sampleBuffers(4, 3), p, 4, valid, 3, randomRecipeFixture(), randomLimitsFixture()); err != nil {
					t.Fatal(err)
				}
				// Declared completed metadata only; no optimiser/model is run.
				if err := state.CommitRow(p, step); err != nil {
					t.Fatal(err)
				}
			}
			if !bytes.Equal(snapshotRunFixture(t, a), snapshotRunFixture(t, b)) {
				t.Fatal("raw resume boundary differs")
			}
			var err error
			b, err = RestoreTrainingRunState(snapshotRunFixture(t, b), runIdentityFixture(), runBindingFixture(step))
			if err != nil {
				t.Fatal(err)
			}
		}
	})
}

func TestTrainingRandomTransformDomain(t *testing.T) {
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := trainingTimeFromLogit(bad); err == nil {
			t.Fatal("accepted nonfinite explicit logit")
		}
		if _, err := trainingTimeFromNormal(bad); err == nil {
			t.Fatal("accepted nonfinite normal")
		}
		if _, _, err := trainingPairFromNormals(0, bad); err == nil {
			t.Fatal("accepted nonfinite pair")
		}
	}
	// No transcendental numeric goldens/tolerance or statistical diagnostic is
	// supplied: independent high-precision BOTH BM/time fixtures are prerequisite.
	// Domain and symmetry assertions do not establish numerical acceptance.
	for _, pair := range [][2]float32{{0, 0}, {-1, 1}, {1, -1}, {-math.MaxFloat32, math.MaxFloat32}} {
		s, end, err := trainingPairFromNormals(pair[0], pair[1])
		if err != nil || s < 0 || s > end || end > 1 {
			t.Fatal("pair domain/order invalid")
		}
		if pair[0] == pair[1] && s != end {
			t.Fatal("equal pair split")
		}
	}
}
