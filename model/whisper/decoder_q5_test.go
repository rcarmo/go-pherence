package whisper

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Packed Q5 decoding must leave every logit bit-identical to the widened F32
// decoder (default and original-compat arithmetic, single and parallel rows).
func TestDecoderQ5MatchesF32(t *testing.T) {
	cfg := Config{EncoderDModel: 64, DecoderDModel: 64, DecoderLayers: 2, DecoderHeads: 2, EncoderHeads: 2, HeadDim: 32, DecoderFFNDim: 128, EncoderFFNDim: 128, MaxDecoderLength: 8, VocabSize: 96}
	r := rand.New(rand.NewSource(7))
	q5 := func(out, in int) ([]byte, []float32) {
		raw := make([]byte, out*in/32*simdrt.Q5_0BlockBytes)
		for b := 0; b < len(raw)/simdrt.Q5_0BlockBytes; b++ {
			blk := raw[b*simdrt.Q5_0BlockBytes:][:simdrt.Q5_0BlockBytes]
			binary.LittleEndian.PutUint16(blk, 0x2000|uint16(r.Intn(0x0c00))|uint16(r.Intn(2))<<15) // finite small scales
			r.Read(blk[2:])
		}
		w := make([]float32, out*in)
		simdrt.DequantQ5_0Into(w, raw)
		return raw, w
	}
	vec := func(n int, s float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(r.NormFloat64() * s)
		}
		return v
	}
	dec := NewDecoder(cfg)
	d, f := cfg.DecoderDModel, cfg.DecoderFFNDim
	layers := make([]decoderLayerQ5, cfg.DecoderLayers)
	dec.PosEmbed = vec(cfg.MaxDecoderLength*d, 0.3)
	dec.FinalLNWeight, dec.FinalLNBias = vec(d, 1), vec(d, 0.1)
	var head []byte
	head, dec.TokenEmbed = q5(cfg.VocabSize, d)
	for l := range dec.Layers {
		L, P := &dec.Layers[l], &layers[l]
		L.SelfAttnLNWeight, L.SelfAttnLNBias, L.CrossAttnLNWeight, L.CrossAttnLNBias, L.MLPLNWeight, L.MLPLNBias = vec(d, 1), vec(d, .1), vec(d, 1), vec(d, .1), vec(d, 1), vec(d, .1)
		P.selfQ, L.SelfQWeight = q5(d, d)
		P.selfK, L.SelfKWeight = q5(d, d)
		P.selfV, L.SelfVWeight = q5(d, d)
		P.selfO, L.SelfOWeight = q5(d, d)
		P.crossQ, L.CrossQWeight = q5(d, d)
		P.crossO, L.CrossOWeight = q5(d, d)
		P.fc1, L.FC1Weight = q5(f, d)
		P.fc2, L.FC2Weight = q5(d, f)
		_, L.CrossKWeight = q5(d, d)
		_, L.CrossVWeight = q5(d, d)
		L.SelfQBias, L.SelfVBias, L.SelfOBias, L.CrossQBias, L.CrossOBias, L.CrossVBias = vec(d, .1), vec(d, .1), vec(d, .1), vec(d, .1), vec(d, .1), vec(d, .1)
		L.FC1Bias, L.FC2Bias = vec(f, .1), vec(d, .1)
	}
	frames := 7
	enc := vec(frames*d, 1)
	old := linearWorkers
	defer func() { linearWorkers = old }()
	for _, workers := range []int{1, 4} {
		linearWorkers = workers
		for _, compat := range []bool{false, true} {
			run := func(packed bool) [][]float32 {
				dec.q5Layers, dec.lmHeadQ5 = nil, nil
				if packed {
					dec.q5Layers, dec.lmHeadQ5 = layers, head
				}
				s := NewDecoderState(cfg, enc, frames, dec)
				if compat {
					s.applyOriginalDecoderCompatibility(frames)
				}
				var out [][]float32
				for _, tok := range []int{3, 50, 95, 0, 17} {
					out = append(out, dec.ForwardToken(tok, s))
				}
				return out
			}
			want, got := run(false), run(true)
			for i := range want {
				for j := range want[i] {
					if math.Float32bits(want[i][j]) != math.Float32bits(got[i][j]) {
						t.Fatalf("workers=%d compat=%v token %d logit %d: %v vs %v", workers, compat, i, j, got[i][j], want[i][j])
					}
				}
			}
		}
	}
	dec.q5Layers, dec.lmHeadQ5 = nil, nil
}

func TestLinearQ5IntoMatchesLinearInto(t *testing.T) {
	r := rand.New(rand.NewSource(9))
	old := linearWorkers
	defer func() { linearWorkers = old }()
	for _, shape := range [][2]int{{1280, 1280}, {5120, 1280}, {1280, 5120}, {1000, 96}} {
		out, in := shape[0], shape[1]
		raw := make([]byte, out*in/32*simdrt.Q5_0BlockBytes)
		for b := 0; b < len(raw)/simdrt.Q5_0BlockBytes; b++ {
			blk := raw[b*simdrt.Q5_0BlockBytes:][:simdrt.Q5_0BlockBytes]
			binary.LittleEndian.PutUint16(blk, uint16(r.Intn(0x7c00))|uint16(r.Intn(2))<<15)
			r.Read(blk[2:])
		}
		w := make([]float32, out*in)
		simdrt.DequantQ5_0Into(w, raw)
		x, bias := make([]float32, in), make([]float32, out)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		for i := range bias {
			bias[i] = float32(r.NormFloat64())
		}
		for _, workers := range []int{1, 4} {
			linearWorkers = workers
			for _, b := range [][]float32{nil, bias} {
				want, got := make([]float32, out), make([]float32, out)
				linearInto(want, x, w, b, in, out)
				linearQ5Into(got, x, raw, b, in, out)
				for i := range want {
					if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
						t.Fatalf("%v workers=%d bias=%v row %d", shape, workers, b != nil, i)
					}
				}
			}
		}
	}
}

func TestCrossAttentionParallelHeadsExact(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	heads, hd, seq := 20, 64, 1500
	q, k, v := make([]float32, heads*hd), make([]float32, heads*seq*hd), make([]float32, heads*seq*hd)
	for _, s := range [][]float32{q, k, v} {
		for i := range s {
			s[i] = float32(r.NormFloat64())
		}
	}
	for _, pad := range []int{0, 36} {
		want, got := make([]float32, heads*hd), make([]float32, heads*hd)
		crossAttentionHeadMajorPadded(want, q, k, v, seq, heads, hd, make([]float32, seq), 0, 0, nil, pad)
		for _, workers := range []int{2, 3, 4, 7} {
			crossAttentionHeadMajorParallel(got, q, k, v, seq, heads, hd, make([]float32, workers*seq), 0, 0, nil, pad, workers)
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
					t.Fatalf("pad=%d workers=%d out %d", pad, workers, i)
				}
			}
		}
	}
}
