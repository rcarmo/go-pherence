package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedRNNTDecoderCachedStepsPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	var state DecoderState
	var first []float32
	var firstValue float32
	for index, token := range []int{rnntBlank, 3, 7, rnntBlank, 11} {
		got, err := decoder.Step(token, &state)
		if err != nil {
			t.Fatal(err)
		}
		if index == 0 {
			first, firstValue = got, got[0]
		} else if &got[0] == &first[0] || first[0] != firstValue {
			t.Fatal("Step output aliases a prior call")
		}
		output, hidden, cell, initialized := state.Snapshot()
		if !initialized || len(output) != rnntHidden || len(hidden) != 2*rnntHidden || len(cell) != 2*rnntHidden {
			t.Fatalf("step=%d cache shape", index)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"output", got}, {"hidden", hidden}, {"cell", cell}} {
			ref := readStemFixture(t, fmt.Sprintf("rnnt_decoder_step%d_%s", index, item.name), len(item.got))
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			t.Logf("step=%d %s max_abs=%g mean_abs=%g outside=%d", index, item.name, maxAbs, sumAbs/float64(len(item.got)), outside)
			if outside != 0 {
				t.Fatalf("step=%d %s differs from PyTorch", index, item.name)
			}
		}
	}
	if first[0] != firstValue || &first[0] == &state.output[0] {
		t.Fatal("Step output aliases state after decoding")
	}
}

func TestRNNTDecoderRejectsMalformedState(t *testing.T) {
	if _, err := LoadRNNTDecoder(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, err := (*RNNTDecoder)(nil).Step(0, &DecoderState{}); err == nil {
		t.Fatal("accepted nil decoder")
	}
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	var state DecoderState
	if _, err := decoder.Step(3, &state); err != nil {
		t.Fatal(err)
	}
	out, hidden, cell, initialized := state.Snapshot()
	for _, token := range []int{-1, rnntVocabulary} {
		if _, err := decoder.Step(token, &state); err == nil {
			t.Fatalf("accepted token=%d", token)
		}
	}
	out2, hidden2, cell2, initialized2 := state.Snapshot()
	if !reflect.DeepEqual(out, out2) || !reflect.DeepEqual(hidden, hidden2) || !reflect.DeepEqual(cell, cell2) || initialized != initialized2 {
		t.Fatal("rejected token changed cache")
	}
	state.hidden[0] = float32(math.NaN())
	if _, err := decoder.Step(rnntBlank, &state); err == nil {
		t.Fatal("accepted non-finite cached state on blank")
	}
	state.hidden[0] = hidden[0]
	state.hidden = state.hidden[:rnntHidden]
	if _, err := decoder.Step(7, &state); err == nil {
		t.Fatal("accepted truncated cached state")
	}
}
