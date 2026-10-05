package pockettts

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"testing"
)

func sampleBuffers(frames, channels int) TrainingStepSamples {
	dst := TrainingStepSamples{Mask: make([]bool, frames), Noise: make([]float32, frames*channels), DiagonalTime: make([]float32, frames), DistillS: make([]float32, frames), DistillT: make([]float32, frames)}
	for i := range dst.Mask {
		dst.Mask[i] = i%2 == 0
	}
	for _, values := range sampleFloatOutputs(dst) {
		for i := range values {
			values[i] = -9
		}
	}
	return dst
}
func sampleDraws(valid, channels int) TrainingStepDraws {
	draws := TrainingStepDraws{Noise: make([]float32, valid*channels), DiagonalTime: make([]float32, valid), DistillS: make([]float32, valid), DistillT: make([]float32, valid)}
	for i := range draws.Noise {
		draws.Noise[i] = float32(i+1) / 8
	}
	for i := range draws.DiagonalTime {
		draws.DiagonalTime[i], draws.DistillS[i], draws.DistillT[i] = .5, .25, .75
	}
	return draws
}
func sampleFloatOutputs(dst TrainingStepSamples) [4][]float32 {
	return [4][]float32{dst.Noise, dst.DiagonalTime, dst.DistillS, dst.DistillT}
}
func sampleFloatInputs(draws TrainingStepDraws) [4][]float32 {
	return [4][]float32{draws.Noise, draws.DiagonalTime, draws.DistillS, draws.DistillT}
}
func setSampleOutput(dst *TrainingStepSamples, index int, values []float32) {
	switch index {
	case 0:
		dst.Noise = values
	case 1:
		dst.DiagonalTime = values
	case 2:
		dst.DistillS = values
	case 3:
		dst.DistillT = values
	}
}
func setSampleInput(draws *TrainingStepDraws, index int, values []float32) {
	switch index {
	case 0:
		draws.Noise = values
	case 1:
		draws.DiagonalTime = values
	case 2:
		draws.DistillS = values
	case 3:
		draws.DistillT = values
	}
}
func sampleBits(values []float32) []uint32 {
	bits := make([]uint32, len(values))
	for i, value := range values {
		bits[i] = math.Float32bits(value)
	}
	return bits
}
func requireSampleBits(t *testing.T, got, want []float32) {
	t.Helper()
	// Exact copying, signed-zero preservation and positive-zero padding are
	// the contract here. This is not generated speech/numerical-loss parity.
	if !reflect.DeepEqual(sampleBits(got), sampleBits(want)) {
		t.Fatalf("copy/padding differs: got %v want %v", got, want)
	}
}
func rejectSamplesUnchanged(t *testing.T, dst TrainingStepSamples, frames, valid, channels int, draws TrainingStepDraws, limits TrainingSampleLimits) {
	t.Helper()
	mask := append([]bool(nil), dst.Mask...)
	var before, inputBefore [4][]uint32
	for i, values := range sampleFloatOutputs(dst) {
		before[i] = sampleBits(values)
	}
	for i, values := range sampleFloatInputs(draws) {
		inputBefore[i] = sampleBits(values)
	}
	if err := PrepareTrainingSamplesInto(dst, frames, valid, channels, draws, limits); err == nil {
		t.Fatal("accepted malformed samples")
	}
	if !reflect.DeepEqual(mask, append([]bool(nil), dst.Mask...)) {
		t.Fatal("rejection mutated mask")
	}
	for i, values := range sampleFloatOutputs(dst) {
		if !reflect.DeepEqual(before[i], sampleBits(values)) {
			t.Fatalf("rejection mutated output %d", i)
		}
	}
	for i, values := range sampleFloatInputs(draws) {
		if !reflect.DeepEqual(inputBefore[i], sampleBits(values)) {
			t.Fatalf("rejection mutated input %d", i)
		}
	}
}

func TestTrainingSamplesExplicitFixture(t *testing.T) {
	// Local fixture/generator commit 961c645958cdaeb690cfbbef74186b80698d6dd5.
	// The hash is fixture-input provenance, not generated-output acceptance.
	file, err := os.Open("testdata/training_step_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	data, readErr := io.ReadAll(io.LimitReader(file, (128<<10)+1))
	closeErr := file.Close()
	if readErr != nil || closeErr != nil || len(data) != 84996 {
		t.Fatalf("bounded fixture read: bytes=%d read=%v close=%v", len(data), readErr, closeErr)
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != "2774178aba94bfcfea47932d557c53c5aaf2a43ba32a12fbe1d7a463070d22ed" {
		t.Fatal("fixture-input provenance changed")
	}
	var fixture struct {
		Schema           int       `json:"schema"`
		UpstreamRevision string    `json:"upstream_revision"`
		Generator        string    `json:"generator"`
		Frames           int       `json:"frames"`
		Channels         int       `json:"latent_dim"`
		Mask             []bool    `json:"mask"`
		Noise            []float32 `json:"noise"`
		Diagonal         []float32 `json:"diagonal_time"`
		S                []float32 `json:"distill_s"`
		T                []float32 `json:"distill_t"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.UpstreamRevision != "0acce6b2f390150267557770d2098c5caa9a18ac" || fixture.Generator != "pocket-tts .venv/bin/python (torch 2.13.0+cu130)" || fixture.Frames != 3 || fixture.Channels != 2 || !reflect.DeepEqual(fixture.Mask, []bool{true, true, false}) || len(fixture.Noise) != 6 || len(fixture.Diagonal) != 3 || len(fixture.S) != 3 || len(fixture.T) != 3 {
		t.Fatal("unexpected fixture schema/provenance/shape")
	}
	draws := TrainingStepDraws{fixture.Noise[:4], fixture.Diagonal[:2], fixture.S[:2], fixture.T[:2]}
	dst := sampleBuffers(3, 2)
	if err := PrepareTrainingSamplesInto(dst, 3, 2, 2, draws, TrainingSampleLimits{3, 6}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(dst.Mask, fixture.Mask) {
		t.Fatal("fixture mask differs")
	}
	for i, want := range [4][]float32{fixture.Noise, fixture.Diagonal, fixture.S, fixture.T} {
		requireSampleBits(t, sampleFloatOutputs(dst)[i], want)
	}
}

func TestTrainingSamplesShapeAdmission(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                    string
		frames, valid, channels int
		limits                  TrainingSampleLimits
	}{
		{"frames_zero", 0, 1, 2, TrainingSampleLimits{3, 6}},
		{"frames_negative", -1, 1, 2, TrainingSampleLimits{3, 6}},
		{"valid_zero", 3, 0, 2, TrainingSampleLimits{3, 6}},
		{"valid_negative", 3, -1, 2, TrainingSampleLimits{3, 6}},
		{"valid_exceeds_frames", 3, 4, 2, TrainingSampleLimits{3, 6}},
		{"channels_zero", 3, 2, 0, TrainingSampleLimits{3, 6}},
		{"channels_negative", 3, 2, -1, TrainingSampleLimits{3, 6}},
		{"frame_limit_zero", 3, 2, 2, TrainingSampleLimits{0, 6}},
		{"frame_limit_negative", 3, 2, 2, TrainingSampleLimits{-1, 6}},
		{"noise_limit_zero", 3, 2, 2, TrainingSampleLimits{3, 0}},
		{"noise_limit_negative", 3, 2, 2, TrainingSampleLimits{3, -1}},
		{"frame_limit_exceeded", 3, 2, 2, TrainingSampleLimits{2, 6}},
		{"noise_limit_exceeded", 3, 2, 2, TrainingSampleLimits{3, 5}},
		{"element_overflow", maxInt, 1, 2, TrainingSampleLimits{maxInt, maxInt}},
		{"byte_overflow", maxInt/4 + 1, 1, 1, TrainingSampleLimits{maxInt, maxInt}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rejectSamplesUnchanged(t, sampleBuffers(3, 2), tc.frames, tc.valid, tc.channels, sampleDraws(2, 2), tc.limits)
		})
	}
	for _, size := range []int{0, 2, 4} {
		t.Run(fmt.Sprintf("mask_length_%d", size), func(t *testing.T) {
			dst := sampleBuffers(3, 2)
			dst.Mask = make([]bool, size)
			rejectSamplesUnchanged(t, dst, 3, 2, 2, sampleDraws(2, 2), TrainingSampleLimits{3, 6})
		})
	}
	for field := 0; field < 4; field++ {
		t.Run(fmt.Sprintf("output_%d_nil", field), func(t *testing.T) {
			dst := sampleBuffers(3, 2)
			setSampleOutput(&dst, field, nil)
			rejectSamplesUnchanged(t, dst, 3, 2, 2, sampleDraws(2, 2), TrainingSampleLimits{3, 6})
		})
		t.Run(fmt.Sprintf("input_%d_nil", field), func(t *testing.T) {
			draws := sampleDraws(2, 2)
			setSampleInput(&draws, field, nil)
			rejectSamplesUnchanged(t, sampleBuffers(3, 2), 3, 2, 2, draws, TrainingSampleLimits{3, 6})
		})
		for _, offset := range []int{-1, 1} {
			t.Run(fmt.Sprintf("output_%d_length_%+d", field, offset), func(t *testing.T) {
				dst := sampleBuffers(3, 2)
				setSampleOutput(&dst, field, make([]float32, len(sampleFloatOutputs(dst)[field])+offset))
				rejectSamplesUnchanged(t, dst, 3, 2, 2, sampleDraws(2, 2), TrainingSampleLimits{3, 6})
			})
			t.Run(fmt.Sprintf("input_%d_length_%+d", field, offset), func(t *testing.T) {
				draws := sampleDraws(2, 2)
				setSampleInput(&draws, field, make([]float32, len(sampleFloatInputs(draws)[field])+offset))
				rejectSamplesUnchanged(t, sampleBuffers(3, 2), 3, 2, 2, draws, TrainingSampleLimits{3, 6})
			})
		}
	}
}

func TestTrainingSamplesValues(t *testing.T) {
	for field := 0; field < 4; field++ {
		for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			t.Run(fmt.Sprintf("nonfinite_field_%d_bits_%x", field, math.Float32bits(value)), func(t *testing.T) {
				draws := sampleDraws(2, 2)
				values := sampleFloatInputs(draws)[field]
				values[len(values)-1] = value // last element catches late validation
				rejectSamplesUnchanged(t, sampleBuffers(3, 2), 3, 2, 2, draws, TrainingSampleLimits{3, 6})
			})
		}
	}
	for field := 1; field < 4; field++ {
		for _, value := range []float32{-.01, 1.01} {
			t.Run(fmt.Sprintf("range_field_%d_value_%g", field, value), func(t *testing.T) {
				draws := sampleDraws(2, 2)
				sampleFloatInputs(draws)[field][1] = value
				rejectSamplesUnchanged(t, sampleBuffers(3, 2), 3, 2, 2, draws, TrainingSampleLimits{3, 6})
			})
		}
	}
	t.Run("reversed_pair", func(t *testing.T) {
		draws := sampleDraws(2, 2)
		draws.DistillS[1], draws.DistillT[1] = .8, .2
		rejectSamplesUnchanged(t, sampleBuffers(3, 2), 3, 2, 2, draws, TrainingSampleLimits{3, 6})
	})
	t.Run("finite_endpoints_and_signed_zero", func(t *testing.T) {
		negativeZero := math.Float32frombits(0x80000000)
		draws := TrainingStepDraws{[]float32{negativeZero, math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32}, []float32{negativeZero, 1}, []float32{negativeZero, 1}, []float32{0, 1}}
		dst := sampleBuffers(3, 2)
		if err := PrepareTrainingSamplesInto(dst, 3, 2, 2, draws, TrainingSampleLimits{3, 6}); err != nil {
			t.Fatal(err)
		}
		for i, input := range sampleFloatInputs(draws) {
			output := sampleFloatOutputs(dst)[i]
			requireSampleBits(t, output[:len(input)], input)
			for _, value := range output[len(input):] {
				if math.Float32bits(value) != 0 {
					t.Fatal("sample padding is not positive zero")
				}
			}
		}
	})
}

func TestTrainingSamplesAliasing(t *testing.T) {
	// channels=1 gives equal-length views; cover every writable pair and every
	// output/input pair, including exact and shifted overlaps.
	for shift := 0; shift <= 2; shift++ {
		for a := 0; a < 4; a++ {
			for b := a + 1; b < 4; b++ {
				t.Run(fmt.Sprintf("outputs_%d_%d_shift_%d", a, b, shift), func(t *testing.T) {
					dst := sampleBuffers(3, 1)
					storage := []float32{.2, .3, .4, .5, .6}
					setSampleOutput(&dst, a, storage[:3])
					setSampleOutput(&dst, b, storage[shift:shift+3])
					rejectSamplesUnchanged(t, dst, 3, 2, 1, sampleDraws(2, 1), TrainingSampleLimits{3, 3})
				})
			}
			for b := 0; b < 4; b++ {
				t.Run(fmt.Sprintf("output_%d_input_%d_shift_%d", a, b, shift), func(t *testing.T) {
					dst, draws := sampleBuffers(3, 1), sampleDraws(2, 1)
					storage := []float32{.2, .3, .4, .5}
					setSampleOutput(&dst, a, storage[:3])
					setSampleInput(&draws, b, storage[shift:shift+2])
					rejectSamplesUnchanged(t, dst, 3, 2, 1, draws, TrainingSampleLimits{3, 3})
				})
			}
		}
	}
	t.Run("output_starts_inside_input", func(t *testing.T) {
		dst, draws := sampleBuffers(3, 1), sampleDraws(2, 1)
		storage := []float32{.2, .3, .4, .5}
		draws.DistillT, dst.Noise = storage[:2], storage[1:4]
		rejectSamplesUnchanged(t, dst, 3, 2, 1, draws, TrainingSampleLimits{3, 3})
	})
	t.Run("read_only_inputs_may_alias", func(t *testing.T) {
		values := []float32{.25, .5}
		draws := TrainingStepDraws{values, values, values, values}
		dst := sampleBuffers(3, 1)
		if err := PrepareTrainingSamplesInto(dst, 3, 2, 1, draws, TrainingSampleLimits{3, 3}); err != nil {
			t.Fatal(err)
		}
		for _, output := range sampleFloatOutputs(dst) {
			requireSampleBits(t, output, []float32{.25, .5, 0})
		}
		requireSampleBits(t, values, []float32{.25, .5})
	})
	t.Run("adjacent_disjoint_views", func(t *testing.T) {
		arena := make([]float32, 20)
		dst := TrainingStepSamples{make([]bool, 3), arena[:3], arena[3:6], arena[6:9], arena[9:12]}
		draws := TrainingStepDraws{arena[12:14], arena[14:16], arena[16:18], arena[18:20]}
		for _, values := range sampleFloatInputs(draws) {
			copy(values, []float32{.25, .5})
		}
		if err := PrepareTrainingSamplesInto(dst, 3, 2, 1, draws, TrainingSampleLimits{3, 3}); err != nil {
			t.Fatal(err)
		}
	})
}

func TestTrainingSamplesPaddingOwnership(t *testing.T) {
	for _, tc := range []struct{ frames, valid, channels int }{{1, 1, 1}, {3, 1, 2}, {3, 2, 2}, {3, 3, 2}} {
		t.Run(fmt.Sprintf("frames_%d_valid_%d_channels_%d", tc.frames, tc.valid, tc.channels), func(t *testing.T) {
			maskArena := make([]bool, tc.frames+2)
			for i := range maskArena {
				maskArena[i] = true
			}
			dst := TrainingStepSamples{Mask: maskArena[1 : tc.frames+1]}
			var arenas [4][]float32
			for i := range arenas {
				length := tc.frames
				if i == 0 {
					length *= tc.channels
				}
				arenas[i] = make([]float32, length+2)
				for j := range arenas[i] {
					arenas[i][j] = -9
				}
				setSampleOutput(&dst, i, arenas[i][1:length+1])
			}
			draws := sampleDraws(tc.valid, tc.channels)
			var expected [4][]float32
			for i, input := range sampleFloatInputs(draws) {
				expected[i] = make([]float32, len(sampleFloatOutputs(dst)[i]))
				copy(expected[i], input) // independent copy/zero layout oracle
			}
			if err := PrepareTrainingSamplesInto(dst, tc.frames, tc.valid, tc.channels, draws, TrainingSampleLimits{3, 6}); err != nil {
				t.Fatal(err)
			}
			for row, active := range dst.Mask {
				if active != (row < tc.valid) {
					t.Fatal("invalid prefix mask")
				}
			}
			if !maskArena[0] || !maskArena[len(maskArena)-1] {
				t.Fatal("mask guard overwritten")
			}
			for i, output := range sampleFloatOutputs(dst) {
				requireSampleBits(t, output, expected[i])
				requireSampleBits(t, sampleFloatInputs(draws)[i], expected[i][:len(sampleFloatInputs(draws)[i])])
				if arenas[i][0] != -9 || arenas[i][len(arenas[i])-1] != -9 {
					t.Fatal("output guard overwritten")
				}
			}
			// An EOS boundary would be the first existing invalid frame. The
			// helper never appends one, trims rows or evaluates EOS loss.
			if len(dst.Mask) != tc.frames || (tc.valid < tc.frames && dst.Mask[tc.valid]) {
				t.Fatal("physical/EOS boundary changed")
			}
			for _, input := range sampleFloatInputs(draws) {
				clear(input)
			}
			for i, output := range sampleFloatOutputs(dst) {
				requireSampleBits(t, output, expected[i])
			}
		})
	}
}

func TestTrainingSamplesNoDraws(t *testing.T) {
	// These are metadata-only fixtures, with dummy hash bindings and no model
	// update, checkpoint file, loss/gradient computation or trainer invocation.
	state, peer := newRunFixture(t, 3), newRunFixture(t, 3)
	proposal, err := state.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	parentBefore, proposalBefore := snapshotRunFixture(t, state), proposal.next.clone()
	dst, draws, limits := sampleBuffers(3, 2), sampleDraws(2, 2), TrainingSampleLimits{3, 6}
	if err := PrepareTrainingSamplesInto(dst, 3, 2, 2, draws, limits); err != nil {
		t.Fatal(err)
	}
	rejectSamplesUnchanged(t, dst, 3, 0, 2, draws, limits)
	if !bytes.Equal(parentBefore, snapshotRunFixture(t, state)) || !reflect.DeepEqual(proposalBefore, proposal.next) {
		t.Fatal("preparation changed row/RNG metadata")
	}
	// Abandon the isolated proposal. Restoring the untouched parent's boundary
	// must preserve its next raw PCG draws, without assuming normal equivalence.
	restored, err := RestoreTrainingRunState(parentBefore, runIdentityFixture(), runBindingFixture(0))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(parentBefore, snapshotRunFixture(t, restored)) {
		t.Fatal("restore changed metadata")
	}
	next, err := state.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	peerNext, err := peer.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	restoredNext, err := restored.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []*TrainingRowProposal{peerNext, restoredNext} {
		if p.row != next.row {
			t.Fatal("abandon/restore changed next row")
		}
	}
	for i := 0; i < 7; i++ {
		timeDraw, err := next.TimeUint64()
		if err != nil {
			t.Fatal(err)
		}
		noiseDraw, err := next.NoiseUint64()
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range []*TrainingRowProposal{peerNext, restoredNext} {
			gotTime, timeErr := p.TimeUint64()
			gotNoise, noiseErr := p.NoiseUint64()
			if timeErr != nil || noiseErr != nil || gotTime != timeDraw || gotNoise != noiseDraw {
				t.Fatal("abandon/restore changed raw draws")
			}
		}
	}
	// AllocsPerRun measures the helper alone with already-owned tiny buffers.
	// Its warm-up and 100-call average do not qualify every shape/error path,
	// whole training allocations, or the historical inference outlier.
	var helperErr error
	allocs := testing.AllocsPerRun(100, func() {
		helperErr = PrepareTrainingSamplesInto(dst, 3, 2, 2, draws, limits)
	})
	if helperErr != nil || allocs != 0 {
		t.Fatalf("helper-only average allocations=%g err=%v", allocs, helperErr)
	}
}
