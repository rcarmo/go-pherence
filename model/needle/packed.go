package needle

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type packedKey struct {
	name  string
	layer int
}

// PackedBytes reports retained original CQ payloads/codebooks, in addition to
// decoded model storage. This hybrid path does not claim a memory reduction.
func (m *Model) PackedBytes() int64 {
	if m == nil {
		return 0
	}
	var n int64
	for _, p := range m.packed {
		n += p.Bytes()
	}
	return n
}
func (e *execution) packedLinear(x *value, m *simd.CQMatrix) *value {
	// Mul's input transform is padded128; charge its bounded scratch to workspace.
	scratch := int64((x.c+127)/128*128) * 4
	batch := scratch * int64(x.r)
	if batch <= 64<<20 {
		scratch = batch
	}
	e.t.reserve(scratch + 512)
	out := e.t.alloc(x.r, m.Rows())
	if !m.Mul(out.x, x.x, x.r) {
		panic(workLimit{fmt.Errorf("needle: invalid packed matrix input")})
	}
	for _, v := range out.x {
		if !finite(v) {
			panic(workLimit{fmt.Errorf("needle: nonfinite packed output")})
		}
	}
	return out
}
func (e *execution) outputProjection(x *value) *value {
	if e.packedEnabled {
		if p := e.m.packed[packedKey{"embedding/embedding", -1}]; p != nil {
			out := e.packedLinear(e.aq(x), p)
			if out.c != e.m.config.OutVocab {
				return e.t.cols(out, 0, e.m.config.OutVocab)
			}
			return out
		}
	}
	emb := e.param("embedding/embedding", -1)
	var head *value
	if e.parameterViews != nil {
		n := e.m.config.OutVocab * emb.c
		head = e.t.view(emb.x[:n:n], e.m.config.OutVocab, emb.c)
	} else {
		head = e.t.rows(emb, 0, e.m.config.OutVocab)
	}
	return e.t.mm(e.aq(x), head, true)
}
