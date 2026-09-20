package llamagraph

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// Validate checks Go extents and the native signed-int/reshape contract before
// indexing config slices or entering C. It does not admit RSS or prove native
// allocation rollback, tensor dtype support, weight upload or decode lifetimes.
func (c Config) Validate() error {
	for name, n := range map[string]int{"vocab": c.NVocab, "embedding": c.NEmbd, "heads": c.NHeads, "KV heads": c.NHeadsKV, "layers": c.NLayers, "FFN": c.NFF, "context": c.NCtx, "threads": c.NThreads} {
		if n <= 0 || n > math.MaxInt32 {
			return fmt.Errorf("llamagraph invalid %s=%d", name, n)
		}
	}
	if c.NLayers > 128 {
		return fmt.Errorf("llamagraph layers exceed native array limit128")
	}
	if c.NHeadsKV > c.NHeads || c.NHeads%c.NHeadsKV != 0 {
		return fmt.Errorf("llamagraph invalid GQA grouping")
	}
	for _, v := range []float32{c.RopeBase, c.RmsEps} {
		if v <= 0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("llamagraph invalid rope/norm control")
		}
	}
	for _, types := range [][]int{c.WQType, c.WKType, c.WVType, c.WOType, c.FFNGateType, c.FFNUpType, c.FFNDownType} {
		if len(types) < c.NLayers {
			return fmt.Errorf("llamagraph short per-layer dtype array")
		}
		for _, v := range types[:c.NLayers] {
			if v < 0 || v > math.MaxInt32 {
				return fmt.Errorf("llamagraph dtype out of C-int range")
			}
		}
	}
	if c.TokEmbdType < 0 || c.TokEmbdType > math.MaxInt32 || c.OutputType < 0 || c.OutputType > math.MaxInt32 {
		return fmt.Errorf("llamagraph dtype out of C-int range")
	}
	for _, dims := range [][]int{c.WQOut, c.WKOut, c.WVOut, c.WOIn} {
		if dims != nil && len(dims) < c.NLayers {
			return fmt.Errorf("llamagraph short projection dimension array")
		}
		for _, v := range dims {
			if v < 0 || v > math.MaxInt32 {
				return fmt.Errorf("llamagraph invalid projection override")
			}
		}
	}
	dim := func(values []int, i, def int) int {
		if values != nil && values[i] != 0 {
			return values[i]
		}
		return def
	}
	q := dim(c.WQOut, 0, c.NEmbd)
	if q%c.NHeads != 0 {
		return fmt.Errorf("llamagraph Q width not divisible by heads")
	}
	head := q / c.NHeads
	kv, ok := checked.MulInt(head, c.NHeadsKV)
	if !ok || kv > math.MaxInt32 || c.RopeDims <= 0 || c.RopeDims%2 != 0 || c.RopeDims > head {
		return fmt.Errorf("llamagraph invalid head/RoPE dimensions")
	}
	for i := 0; i < c.NLayers; i++ {
		if dim(c.WQOut, i, c.NEmbd) != q || dim(c.WKOut, i, kv) != kv || dim(c.WVOut, i, kv) != kv || dim(c.WOIn, i, c.NEmbd) != q {
			return fmt.Errorf("llamagraph layer%d projection widths disagree with shared head shape", i)
		}
	}
	// Conservative F32-sized extents cover the unquantised KV/weight geometry.
	for _, factors := range [][]int{{c.NEmbd, c.NVocab, 4}, {c.NEmbd, c.NFF, 4}, {c.NEmbd, q, 4}, {c.NEmbd, kv, 4}, {kv, c.NCtx, c.NLayers, 2, 4}} {
		n := 1
		for _, v := range factors {
			n, ok = checked.MulInt(n, v)
			if !ok {
				return fmt.Errorf("llamagraph tensor extent overflow")
			}
		}
	}
	return nil
}
