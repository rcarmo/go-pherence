package gliner2

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TensorSource is implemented by loader/safetensors.File. GetFloat32 returns
// owned decoded weights, so modules remain valid after the file is closed.
type TensorSource interface {
	GetFloat32(string) ([]float32, []int, error)
}

type weightReader struct {
	source TensorSource
	err    error
}

func (r *weightReader) tensor(name string, shape ...int) []float32 {
	if r.err != nil {
		return nil
	}
	// Validate expected dimensions before requesting a potentially expensive decode.
	n := 1
	for _, d := range shape {
		var ok bool
		n, ok = checked.MulInt(n, d)
		if d <= 0 || !ok {
			r.err = fmt.Errorf("%s invalid/overflowing shape %v", name, shape)
			return nil
		}
	}
	x, s, err := r.source.GetFloat32(name)
	if err != nil {
		r.err = fmt.Errorf("%s: %w", name, err)
		return nil
	}
	if len(s) != len(shape) {
		r.err = fmt.Errorf("%s rank mismatch", name)
		return nil
	}
	for i, d := range shape {
		if s[i] != d {
			r.err = fmt.Errorf("%s shape %v want %v", name, s, shape)
			return nil
		}
	}
	if len(x) != n {
		r.err = fmt.Errorf("%s length mismatch", name)
		return nil
	}
	return append([]float32(nil), x...)
}
func (r *weightReader) linear(name string, in, out int) Linear {
	return Linear{InDim: in, OutDim: out, Weight: r.tensor(name+".weight", out, in), Bias: r.tensor(name+".bias", out)}
}
func (r *weightReader) norm(name string, d int) LayerNorm {
	return LayerNorm{Weight: r.tensor(name+".weight", d), Bias: r.tensor(name+".bias", d), Epsilon: 1e-5}
}

// LoadBoundaryModules binds only the encoder and marginal head, not the full
// extraction model. Prefixes match the published GLiNER 2.5 safetensors file.
func LoadBoundaryModules(source TensorSource, hidden int, c BoundaryHeadConfig) (BoundaryEncoder, BoundaryQueryHead, error) {
	var e BoundaryEncoder
	var h BoundaryQueryHead
	if source == nil || hidden <= 0 {
		return e, h, fmt.Errorf("tensor source and hidden width required")
	}
	if err := c.Validate(); err != nil {
		return e, h, err
	}
	d := c.BoundaryDim
	doubled, ok := checked.MulInt(2, d)
	if !ok {
		return e, h, fmt.Errorf("boundary output projection width overflow")
	}
	qkv, ok := checked.MulInt(3, d)
	if c.BoundaryAttentionLayers > 0 && !ok {
		return e, h, fmt.Errorf("boundary QKV width overflow")
	}
	// Check the floating-point product before narrowing; the upper boundary is
	// exclusive because float64(MaxInt) may round upwards. Input projection is 2*ff.
	ff := 1
	if c.BoundaryRefinementLayers > 0 {
		width := float64(d) * c.BoundaryFFNMultiplier
		if math.IsNaN(width) || math.IsInf(width, 0) || width <= 0 || width >= float64(int(^uint(0)>>1)/2) {
			return e, h, fmt.Errorf("boundary FFN width invalid or overflowing")
		}
		ff = max(1, int(width))
	}
	r := weightReader{source: source}
	p := "boundary_head.boundary_encoder"
	e = BoundaryEncoder{HiddenSize: hidden, BoundaryDim: d, LeftProjection: r.linear(p+".left_projection", hidden, d), RightProjection: r.linear(p+".right_projection", hidden, d), OutputProjection: r.linear(p+".output_projection", doubled, d), LayerNorm: r.norm(p+".layer_norm", d), BosState: r.tensor(p+".bos_state", hidden), EosState: r.tensor(p+".eos_state", hidden)}
	for i := 0; i < c.BoundaryAttentionLayers && r.err == nil; i++ {
		n := fmt.Sprintf("%s.attention_blocks.%d", p, i)
		e.AttentionBlocks = append(e.AttentionBlocks, BoundaryAttentionBlock{NumHeads: c.BoundaryAttentionHeads, Window: c.BoundaryAttentionWindow, Norm: r.norm(n+".norm", d), QKVProjection: r.linear(n+".qkv_projection", d, qkv), OutputProjection: r.linear(n+".output_projection", d, d)})
	}
	for i := 0; i < c.BoundaryRefinementLayers && r.err == nil; i++ {
		n := fmt.Sprintf("%s.refinement_blocks.%d", p, i)
		e.RefinementBlocks = append(e.RefinementBlocks, ResidualSwiGLU{Norm: r.norm(n+".norm", d), InputProjection: r.linear(n+".input_projection", d, 2*ff), OutputProjection: r.linear(n+".output_projection", ff, d)})
	}
	p = "boundary_head.boundary_query_head"
	h = BoundaryQueryHead{StartBoundary: r.linear(p+".start_boundary_projection", d, d), StartQuery: r.linear(p+".start_query_projection", hidden, d), EndBoundary: r.linear(p+".end_boundary_projection", d, d), EndQuery: r.linear(p+".end_query_projection", hidden, d), InsideText: r.linear(p+".inside_text_projection", hidden, d), InsideQuery: r.linear(p+".inside_query_projection", hidden, d)}
	if r.err != nil {
		return BoundaryEncoder{}, BoundaryQueryHead{}, r.err
	}
	if err := e.Validate(); err != nil {
		return e, h, err
	}
	return e, h, h.Validate()
}
