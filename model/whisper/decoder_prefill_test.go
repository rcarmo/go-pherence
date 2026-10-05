package whisper

import (
	"math"
	"math/rand"
	"sync"
	"testing"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Batched prompt prefill must leave the KV cache, position and every later
// logit bit-identical to feeding the same tokens one at a time.
func TestAdvanceTokensMatchesSequential(t *testing.T) {
	old := linearWorkers
	defer func() { linearWorkers = old }()
	for _, cfg := range []Config{
		{EncoderDModel: 64, DecoderDModel: 64, DecoderLayers: 2, DecoderHeads: 2, EncoderHeads: 2, HeadDim: 32, DecoderFFNDim: 128, EncoderFFNDim: 128, MaxDecoderLength: 40, VocabSize: 96},
		{EncoderDModel: 64, DecoderDModel: 64, DecoderLayers: 3, DecoderHeads: 4, EncoderHeads: 4, HeadDim: 16, DecoderFFNDim: 256, EncoderFFNDim: 256, MaxDecoderLength: 40, VocabSize: 96},
	} {
		frames := 11
		dec, layers, head, enc := newTestQ5Decoder(cfg, 21, frames)
		dec.q5Layers, dec.lmHeadQ5 = layers, head
		for _, workers := range []int{1, 3, 4} {
			linearWorkers = workers
			for _, compat := range []bool{false, true} {
				for _, n := range []int{2, 5, 23} {
					prompt := make([]int, n)
					for i := range prompt {
						prompt[i] = (i*37 + 5) % cfg.VocabSize
					}
					state := func() *DecoderState {
						s := NewDecoderState(cfg, enc, frames, dec)
						if compat {
							s.applyOriginalDecoderCompatibility(frames)
						}
						return s
					}
					seq, bat := state(), state()
					for _, tok := range prompt {
						dec.AdvanceToken(tok, seq)
					}
					if !dec.batchPrefillEligible(bat) {
						t.Fatal("batch prefill not exercised")
					}
					dec.AdvanceTokens(prompt, bat)
					if seq.Pos != bat.Pos || seq.LastToken != bat.LastToken {
						t.Fatal("position", seq.Pos, bat.Pos, seq.LastToken, bat.LastToken)
					}
					for l := range seq.SelfKCache {
						bitsEqual(t, "selfK", seq.SelfKCache[l], bat.SelfKCache[l])
						bitsEqual(t, "selfV", seq.SelfVCache[l], bat.SelfVCache[l])
					}
					for _, tok := range []int{7, 60, 3} {
						bitsEqual(t, "logits", dec.ForwardToken(tok, seq), dec.ForwardToken(tok, bat))
					}
				}
			}
		}
	}
}

func bitsEqual(t *testing.T, what string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length %d vs %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("%s[%d]: %v vs %v", what, i, got[i], want[i])
		}
	}
}

// The batched prefill must show a cross-attention observer exactly the rows
// the sequential loop shows, for every (layer, head, position), even when heads
// run concurrently.
func TestAdvanceTokensObserverMatchesSequential(t *testing.T) {
	old := linearWorkers
	defer func() { linearWorkers = old }()
	cfg := Config{EncoderDModel: 64, DecoderDModel: 64, DecoderLayers: 3, DecoderHeads: 4, EncoderHeads: 4, HeadDim: 16, DecoderFFNDim: 256, EncoderFFNDim: 256, MaxDecoderLength: 40, VocabSize: 96}
	frames := 11
	dec, layers, head, enc := newTestQ5Decoder(cfg, 21, frames)
	dec.q5Layers, dec.lmHeadQ5 = layers, head
	type key struct{ layer, head, position int }
	record := func(state *DecoderState) (map[key][]float32, *sync.Mutex) {
		rows, mu := map[key][]float32{}, &sync.Mutex{}
		state.CrossAttentionObserver = func(layer, head, position int, row []float32) {
			mu.Lock()
			defer mu.Unlock()
			k := key{layer, head, position}
			if _, dup := rows[k]; dup {
				t.Errorf("duplicate observation %v", k)
			}
			rows[k] = append([]float32(nil), row...)
		}
		return rows, mu
	}
	for _, workers := range []int{1, 4} {
		linearWorkers = workers
		for _, compat := range []bool{false, true} {
			prompt := []int{1, 9, 17, 33, 2, 60, 61, 5, 44, 70, 3}
			state := func() *DecoderState {
				s := NewDecoderState(cfg, enc, frames, dec)
				if compat {
					s.applyOriginalDecoderCompatibility(frames)
				}
				return s
			}
			seq, bat := state(), state()
			want, _ := record(seq)
			got, _ := record(bat)
			for _, tok := range prompt {
				dec.AdvanceToken(tok, seq)
			}
			if !dec.batchPrefillEligible(bat) {
				t.Fatal("batch prefill not exercised with observer")
			}
			dec.AdvanceTokens(prompt, bat)
			if len(want) != cfg.DecoderLayers*cfg.DecoderHeads*len(prompt) || len(got) != len(want) {
				t.Fatalf("observations %d vs %d", len(got), len(want))
			}
			for k, row := range want {
				bitsEqual(t, "observed row", row, got[k])
			}
		}
	}
}

// linearQ5Batch must equal SdotQ5_0 + bias per element for every tile and
// remainder shape.
func TestLinearQ5BatchMatchesPerElement(t *testing.T) {
	old := linearWorkers
	defer func() { linearWorkers = old }()
	r := rand.New(rand.NewSource(3))
	const inDim, outDim = 96, 1100
	rowBytes := inDim / 32 * simdrt.Q5_0BlockBytes
	raw := make([]byte, outDim*rowBytes)
	for b := 0; b < outDim*inDim/32; b++ {
		block := raw[b*simdrt.Q5_0BlockBytes:][:simdrt.Q5_0BlockBytes]
		block[0], block[1] = byte(r.Intn(256)), byte(0x20+r.Intn(16))
		r.Read(block[2:])
	}
	bias := make([]float32, outDim)
	for i := range bias {
		bias[i] = float32(r.NormFloat64())
	}
	for _, workers := range []int{1, 3, 4} {
		linearWorkers = workers
		for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 33} {
			x := make([]float32, n*inDim)
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
			out := make([]float32, n*outDim)
			linearQ5Batch(out, x, raw, bias, inDim, outDim, n)
			for i := 0; i < n; i++ {
				for o := 0; o < outDim; o++ {
					want := simdrt.SdotQ5_0(x[i*inDim:(i+1)*inDim], raw[o*rowBytes:(o+1)*rowBytes]) + bias[o]
					if math.Float32bits(out[i*outDim+o]) != math.Float32bits(want) {
						t.Fatalf("workers=%d n=%d i=%d o=%d", workers, n, i, o)
					}
				}
			}
		}
	}
}
