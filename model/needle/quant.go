package needle

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/half"
)

//go:embed testdata/cq-codebooks.json
var codebookJSON []byte
var cqCodebooks = func() map[string][]float32 {
	var out map[string][]float32
	if err := json.Unmarshal(codebookJSON, &out); err != nil {
		panic(err)
	}
	return out
}()

// Quantization selects Needle3's dequantized CQ-weight/A8/KV8 STE reference.
// Packed .cact execution is a separate format/runtime, not selected here.
type Quantization struct {
	WeightBits     float64
	ActivationBits int
	KVBits         int
}

func (q *Quantization) validate(generation int) error {
	if q == nil {
		return nil
	}
	if generation != 3 {
		return fmt.Errorf("needle: CQ/A8 mode currently requires Needle3")
	}
	if q.WeightBits != 0 && q.WeightBits != 1 && q.WeightBits != 1.58 && q.WeightBits != 2 && q.WeightBits != 4 && q.WeightBits != 8 {
		return fmt.Errorf("needle: unsupported CQ weight bits")
	}
	if q.ActivationBits != 0 && q.ActivationBits != 8 {
		return fmt.Errorf("needle: activation bits must be 0 or 8")
	}
	if q.KVBits != 0 && q.KVBits != 8 {
		return fmt.Errorf("needle: KV bits must be 0 or 8")
	}
	return nil
}
func (t *tape) fakeA8(x *value) *value {
	o := t.alloc(x.r, x.c)
	for r := 0; r < x.r; r++ {
		off := r * x.c
		mx := float32(1e-5)
		for _, v := range x.x[off : off+x.c] {
			a := float32(math.Abs(float64(v)))
			if a > mx {
				mx = a
			}
		}
		scale := mx / 127
		for j := 0; j < x.c; j++ {
			o.x[off+j] = float32(math.RoundToEven(float64(x.x[off+j]/scale))) * scale
		}
	}
	t.record(func() {
		for i, g := range o.g {
			x.g[i] += g
		}
	})
	return o
}
func (e *execution) aq(x *value) *value {
	if e.q != nil && e.q.ActivationBits == 8 {
		return e.t.fakeA8(x)
	}
	return x
}
func (e *execution) kvq(x *value) *value {
	if e.q != nil && e.q.KVBits == 8 {
		return e.t.fakeA8(x)
	}
	return x
}
func isCQ(name string) bool {
	return strings.HasSuffix(name, "/kernel") || strings.HasSuffix(name, "/embedding") || strings.HasPrefix(name, "stack/mhc_phi")
}
func cqSecondLast(name string) bool {
	return strings.HasSuffix(name, "/kernel") || strings.HasPrefix(name, "stack/mhc_phi")
}
func walsh128(x []float32) {
	for stride := 1; stride < 128; stride *= 2 {
		for start := 0; start < 128; start += 2 * stride {
			for j := 0; j < stride; j++ {
				a, b := x[start+j], x[start+j+stride]
				x[start+j], x[start+j+stride] = a+b, a-b
			}
		}
	}
	scale := float32(1 / math.Sqrt(128))
	for i := range x {
		x[i] *= scale
	}
}
func nearestFP16(x float32) float32 { return half.F16ToF32(half.F32ToF16Even(x)) }

// CQ quantizes groups along the reduction axis, after normalized Walsh rotation.
// Norms use IEEE ties-even FP16, unlike legacy half.F32ToF16's finite tie policy.
func cqValues(dst, src []float32, shape []int, second bool, bits float64) {
	key := fmt.Sprint(bits)
	cb := cqCodebooks[key]
	width := shape[len(shape)-1]
	cols := 1
	lead := len(src) / width
	if second {
		cols = width
		width = shape[len(shape)-2]
		lead = len(src) / (width * cols)
	}
	var group [128]float32
	for l := 0; l < lead; l++ {
		for col := 0; col < cols; col++ {
			for start := 0; start < width; start += 128 {
				clear(group[:])
				n := min(128, width-start)
				for j := 0; j < n; j++ {
					idx := l*width + start + j
					if second {
						idx = (l*width+start+j)*cols + col
					}
					group[j] = src[idx]
				}
				walsh128(group[:])
				var norm2 float32
				for _, v := range group {
					norm2 += v * v
				}
				norm := float32(math.Sqrt(float64(norm2)))
				denom := max(norm, float32(1e-12))
				rounded := nearestFP16(norm)
				for i, v := range group {
					unit := v / denom
					right := sort.Search(len(cb), func(i int) bool { return cb[i] >= unit })
					right = max(1, min(right, len(cb)-1))
					left := right - 1
					pick := right
					if math.Abs(float64(unit-cb[left])) <= math.Abs(float64(unit-cb[right])) {
						pick = left
					}
					group[i] = cb[pick] * rounded
				}
				walsh128(group[:])
				for j := 0; j < n; j++ {
					idx := l*width + start + j
					if second {
						idx = (l*width+start+j)*cols + col
					}
					dst[idx] = group[j]
				}
			}
		}
	}
}
func (e *execution) quantParam(name string, p *value) *value {
	if e.q == nil || e.q.WeightBits == 0 || !isCQ(name) || len(e.m.tensors[name].Shape) < 2 {
		return p
	}
	// AB-scaled checkpoints require sandwich derivatives; fail before execution
	// rather than silently applying unscaled CQ.
	for key := range e.m.tensors {
		if strings.HasPrefix(key, "ab_scales/") {
			panic(workLimit{fmt.Errorf("needle: AB-scaled CQ checkpoints not implemented")})
		}
	}
	o := e.t.alloc(p.r, p.c)
	cqValues(o.x, p.x, e.m.tensors[name].Shape, cqSecondLast(name), e.q.WeightBits)
	for _, v := range o.x {
		if !finite(v) {
			panic(workLimit{fmt.Errorf("needle: CQ norm overflow in %s", name)})
		}
	}
	e.t.record(func() {
		for i, g := range o.g {
			p.g[i] += g
		}
	})
	return o
}
