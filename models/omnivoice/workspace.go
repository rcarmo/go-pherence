package omnivoice

import (
	"fmt"
	"math"
	"strconv"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// Workspace owns reusable scratch for one in-flight block call. It is not safe
// for concurrent use. A workspace may be shared sequentially by decoder layers
// with the same dimensions, epsilon and RoPE theta. No model weights live here.
// ForwardInto supports dst == x; dst must not otherwise overlap x or mask.
type Workspace struct {
	packed                                            []float32
	tokens, hidden, intermediate, heads, kvheads, dim int
	theta                                             float64
	arena                                             []float32
	norm, q, k, v, attended, gate, up, down           []float32
	scores, headout, qhead, khead, vhead              []float32
	cos, sin, rotA, rotB                              []float32
	positions                                         []int
}

// NewWorkspace reserves scratch once. The hot path does not grow it implicitly.
func (b *Block) NewWorkspace(tokens int) (*Workspace, error) {
	c := b.config
	dims := []int{c.HiddenSize, c.NumAttentionHeads * c.HeadDim, c.NumKeyValueHeads * c.HeadDim, c.NumKeyValueHeads * c.HeadDim, c.NumAttentionHeads * c.HeadDim, c.IntermediateSize, c.IntermediateSize, c.HiddenSize, tokens, c.HeadDim, c.HeadDim, c.HeadDim, c.HeadDim, c.HeadDim / 2, c.HeadDim / 2}
	total := 0
	for _, width := range dims {
		size, ok := product(tokens, width)
		if !ok || total > int(^uint(0)>>1)/4-size {
			return nil, fmt.Errorf("omnivoice: workspace size overflow")
		}
		total += size
	}
	if total > int(^uint(0)>>1)/4-c.HeadDim {
		return nil, fmt.Errorf("omnivoice: workspace size overflow")
	}
	total += c.HeadDim
	packedSize, ok := product(max(c.HiddenSize, c.IntermediateSize, c.NumAttentionHeads*c.HeadDim), 16)
	if !ok || total > int(^uint(0)>>1)/4-packedSize {
		return nil, fmt.Errorf("omnivoice: packed scratch overflow")
	}
	total += packedSize
	w := &Workspace{tokens: tokens, hidden: c.HiddenSize, intermediate: c.IntermediateSize, heads: c.NumAttentionHeads, kvheads: c.NumKeyValueHeads, dim: c.HeadDim, theta: c.RopeParameters.RopeTheta, arena: make([]float32, total), positions: make([]int, tokens)}
	offset := 0
	take := func(n int) []float32 { r := w.arena[offset : offset+n : offset+n]; offset += n; return r }
	fields := []*[]float32{&w.norm, &w.q, &w.k, &w.v, &w.attended, &w.gate, &w.up, &w.down, &w.scores, &w.headout, &w.qhead, &w.khead, &w.vhead, &w.cos, &w.sin}
	for i, p := range fields {
		*p = take(tokens * dims[i])
	}
	w.rotA = take(c.HeadDim / 2)
	w.rotB = take(c.HeadDim / 2)
	w.packed = take(packedSize)
	for i := range w.positions {
		w.positions[i] = -1
	}
	w.prepareRoPE(nil)
	return w, nil
}

// ScratchBytes returns float scratch plus position-cache storage, excluding Go headers.
func (w *Workspace) ScratchBytes() int { return len(w.arena)*4 + len(w.positions)*strconv.IntSize/8 }
func (w *Workspace) matches(b *Block, tokens int) bool {
	c := b.config
	return w != nil && w.tokens == tokens && w.hidden == c.HiddenSize && w.intermediate == c.IntermediateSize && w.heads == c.NumAttentionHeads && w.kvheads == c.NumKeyValueHeads && w.dim == c.HeadDim && w.theta == c.RopeParameters.RopeTheta
}
func (w *Workspace) prepareRoPE(positions []int) {
	half := w.dim / 2
	for t := 0; t < w.tokens; t++ {
		p := t
		if positions != nil {
			p = positions[t]
		}
		if w.positions[t] == p {
			continue
		}
		w.positions[t] = p
		for j := 0; j < half; j++ {
			angle := float32(p) * float32(1/math.Pow(w.theta, float64(2*j)/float64(w.dim)))
			w.cos[t*half+j] = float32(math.Cos(float64(angle)))
			w.sin[t*half+j] = float32(math.Sin(float64(angle)))
		}
	}
}

// rotate uses existing vector dispatch. Tables are shared by Q/K and every head,
// avoiding per-head transcendental work. Separate operations preserve float32
// multiply/add rounding rather than introducing an untested FMA approximation.
func (w *Workspace) rotate(x []float32, heads int) {
	half := w.dim / 2
	for t := 0; t < w.tokens; t++ {
		co, si := w.cos[t*half:(t+1)*half], w.sin[t*half:(t+1)*half]
		for h := 0; h < heads; h++ {
			start := (t*heads + h) * w.dim
			a, b := x[start:start+half], x[start+half:start+w.dim]
			simd.VecMul(w.rotA, a, co)
			simd.VecMul(w.rotB, b, si)
			simd.VecMul(a, a, si)
			simd.VecMul(b, b, co)
			simd.VecAdd(b, b, a)
			simd.VecScale(w.rotB, w.rotB, -1)
			simd.VecAdd(a, w.rotA, w.rotB)
		}
	}
}
func (s *Workspace) linearInto(y, x, w []float32, rows, in, out int) {
	clear(y)
	if !simd.SgemmNTPackedTo(y, x, w, s.packed, rows, out, in, 1, in, in, out) {
		panic("omnivoice: internal linear shape error")
	}
}
func normalizeInto(y, x, w []float32, rows, width int, eps float32) {
	copy(y, x)
	for r := 0; r < rows; r++ {
		simd.RMSNorm(y[r*width:(r+1)*width], w, eps)
	}
}
