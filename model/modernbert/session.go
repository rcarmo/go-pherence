package modernbert

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// Session owns reusable single-request scratch and is not concurrent-safe.
// Model weights remain immutable and may be shared by many sessions.
type Session struct {
	m                                                                    *Model
	maxSeq                                                               int
	hidden, next, normed, qkv, q, k, v, attn, proj, wi, mid, mlp, scores []float32
}

func (m *Model) NewSession(maxSeq int) (*Session, error) {
	if m == nil {
		return nil, fmt.Errorf("modernbert: nil model")
	}
	if maxSeq < 1 || maxSeq > m.Config.MaxPositions {
		return nil, fmt.Errorf("modernbert: session capacity must be 1..%d", m.Config.MaxPositions)
	}
	h, inter := m.Config.HiddenSize, m.Config.IntermediateSize
	hidden, okH := checked.MulInt(maxSeq, h)
	qkv, okQ := checked.MulInt(hidden, 3)
	wi, okW := checked.MulInt(maxSeq, 2*inter)
	mid, okM := checked.MulInt(maxSeq, inter)
	scores, okS := checked.MulInt(maxSeq, maxSeq)
	if !okH || !okQ || !okW || !okM || !okS {
		return nil, fmt.Errorf("modernbert: session workspace overflow")
	}
	return &Session{m: m, maxSeq: maxSeq, hidden: make([]float32, hidden), next: make([]float32, hidden), normed: make([]float32, hidden), qkv: make([]float32, qkv), q: make([]float32, hidden), k: make([]float32, hidden), v: make([]float32, hidden), attn: make([]float32, hidden), proj: make([]float32, hidden), wi: make([]float32, wi), mid: make([]float32, mid), mlp: make([]float32, hidden), scores: make([]float32, scores)}, nil
}

// ForwardInto writes [len(ids),hidden] to dst with no warm-path heap allocation.
// dst may not alias model/session storage. The session is reusable after errors.
func (s *Session) ForwardInto(dst []float32, ids []int, mask []bool) error {
	if s == nil || s.m == nil {
		return fmt.Errorf("modernbert: nil session")
	}
	c := s.m.Config
	n, h, inter := len(ids), c.HiddenSize, c.IntermediateSize
	if n < 1 || n > s.maxSeq || len(mask) != n || len(dst) != n*h {
		return fmt.Errorf("modernbert: invalid input/output shape")
	}
	any := false
	for _, keep := range mask {
		any = any || keep
	}
	if !any {
		return fmt.Errorf("modernbert: attention mask has no visible tokens")
	}
	hidden, next, normed := s.hidden[:n*h], s.next[:n*h], s.normed[:n*h]
	clear(hidden)
	emb := s.m.weights.embedding
	for i, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return fmt.Errorf("modernbert: token out of range")
		}
		copy(hidden[i*h:], emb[id*h:(id+1)*h])
	}
	layerNorm(normed, hidden, s.m.weights.embeddingNorm, n, h, float32(c.NormEps))
	hidden, normed = normed, hidden
	hd := h / c.Heads
	for l, w := range s.m.weights.layers {
		input := hidden
		if l > 0 {
			layerNorm(normed, hidden, w.attnNorm, n, h, float32(c.NormEps))
			input = normed
		}
		qkv := s.qkv[:n*3*h]
		clear(qkv)
		if !simd.DenseNTTo(qkv, input, w.qkv, n, 3*h, h, 1, h, h, 3*h) {
			return fmt.Errorf("modernbert: qkv rejected")
		}
		q, k, v := s.q[:n*h], s.k[:n*h], s.v[:n*h]
		for i := 0; i < n; i++ {
			copy(q[i*h:], qkv[i*3*h:i*3*h+h])
			copy(k[i*h:], qkv[i*3*h+h:i*3*h+2*h])
			copy(v[i*h:], qkv[i*3*h+2*h:i*3*h+3*h])
		}
		kind := c.LayerTypes[l]
		theta := c.RopeParameters[kind].Theta
		rope(q, n, c.Heads, hd, theta)
		rope(k, n, c.Heads, hd, theta)
		attn := s.attn[:n*h]
		clear(attn)
		scores := s.scores[:n*n]
		window := c.LocalAttention / 2
		scale := float32(1 / math.Sqrt(float64(hd)))
		for head := 0; head < c.Heads; head++ {
			for i := 0; i < n; i++ {
				row := scores[i*n : (i+1)*n]
				for j := 0; j < n; j++ {
					allowed := mask[j]
					if kind == "sliding_attention" && (j < i-window || j > i+window) {
						allowed = false
					}
					if !allowed {
						row[j] = float32(math.Inf(-1))
						continue
					}
					var sum float32
					for d := 0; d < hd; d++ {
						sum += q[i*h+head*hd+d] * k[j*h+head*hd+d]
					}
					row[j] = sum * scale
				}
				if !simd.SoftmaxRowsInPlace(row, 1, n) {
					return fmt.Errorf("modernbert: softmax rejected")
				}
				for d := 0; d < hd; d++ {
					var sum float32
					for j, p := range row {
						sum += p * v[j*h+head*hd+d]
					}
					attn[i*h+head*hd+d] = sum
				}
			}
		}
		proj := s.proj[:n*h]
		clear(proj)
		if !simd.DenseNTTo(proj, attn, w.attnOut, n, h, h, 1, h, h, h) {
			return fmt.Errorf("modernbert: output projection rejected")
		}
		for i := range hidden {
			next[i] = hidden[i] + proj[i]
		}
		hidden, next = next, hidden
		layerNorm(normed, hidden, w.mlpNorm, n, h, float32(c.NormEps))
		wi := s.wi[:n*2*inter]
		clear(wi)
		if !simd.DenseNTTo(wi, normed, w.mlpIn, n, 2*inter, h, 1, h, h, 2*inter) {
			return fmt.Errorf("modernbert: mlp input rejected")
		}
		mid := s.mid[:n*inter]
		for row := 0; row < n; row++ {
			base := row * 2 * inter
			if !simd.GELUExactMulTo(mid[row*inter:(row+1)*inter], wi[base:base+inter], wi[base+inter:base+2*inter]) {
				return fmt.Errorf("modernbert: GeGLU rejected")
			}
		}
		mlp := s.mlp[:n*h]
		clear(mlp)
		if !simd.DenseNTTo(mlp, mid, w.mlpOut, n, h, inter, 1, inter, inter, h) {
			return fmt.Errorf("modernbert: mlp output rejected")
		}
		for i := range hidden {
			next[i] = hidden[i] + mlp[i]
		}
		hidden, next = next, hidden
	}
	layerNorm(dst, hidden, s.m.weights.finalNorm, n, h, float32(c.NormEps))
	for _, v := range dst {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("modernbert: nonfinite output")
		}
	}
	return nil
}
