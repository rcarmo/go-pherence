package needle

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// Checkpoint returns owned data; it can be saved with loader/needle.Save.
func (m *Model) Checkpoint() *checkpoint.Checkpoint {
	cp := &checkpoint.Checkpoint{FormatVersion: 2, Config: append(json.RawMessage(nil), m.rawConfig...), Tensors: map[string]checkpoint.Tensor{}}
	for name, t := range m.tensors {
		cp.Tensors[name] = checkpoint.Tensor{Shape: append([]int{}, t.Shape...), Data: append([]float32(nil), t.Data...)}
	}
	return cp
}
func (m *Model) Configuration() Config {
	b, _ := json.Marshal(m.config)
	var c Config
	_ = json.Unmarshal(b, &c)
	return c
}

// Adapter uses upstream LoRA orientation: A[...,input,rank], B[...,rank,output].
// FP32 training is explicit. CQ straight-through/A8 deployment training is not
// silently approximated by this path.
type Adapter struct {
	Rank    int
	Scale   float32
	Weights map[string]LoRA
}
type LoRA struct{ A, B checkpoint.Tensor }

func (m *Model) NewAdapter(rank int, alpha float32, seed uint64) (*Adapter, error) {
	if rank < 1 || rank > 256 || alpha <= 0 || !finite(alpha) {
		return nil, fmt.Errorf("needle: invalid LoRA rank/alpha")
	}
	ad := &Adapter{Rank: rank, Scale: alpha / float32(rank), Weights: map[string]LoRA{}}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	names := make([]string, 0)
	for name, w := range m.tensors {
		if !loraTarget(name) {
			continue
		}
		nonzero := false
		for _, v := range w.Data {
			if math.Abs(float64(v)) > 1e-6 {
				nonzero = true
				break
			}
		}
		if nonzero {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	var bytes int64
	for _, name := range names {
		w := m.tensors[name]
		l, k, n := w.Shape[0], w.Shape[1], w.Shape[2]
		bytes += int64(l) * int64(k+n) * int64(rank) * 4
		if bytes > 512<<20 {
			return nil, fmt.Errorf("needle: adapter exceeds 512 MiB")
		}
	}
	for _, name := range names {
		w := m.tensors[name]
		l, k, n := w.Shape[0], w.Shape[1], w.Shape[2]
		a := checkpoint.Tensor{Shape: []int{l, k, rank}, Data: make([]float32, l*k*rank)}
		b := checkpoint.Tensor{Shape: []int{l, rank, n}, Data: make([]float32, l*rank*n)}
		for i := range a.Data {
			a.Data[i] = float32(rng.NormFloat64() / float64(rank))
		}
		ad.Weights[name] = LoRA{A: a, B: b}
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("needle: no nonzero LoRA targets")
	}
	return ad, nil
}
func loraTarget(name string) bool {
	if !strings.HasPrefix(name, "stack/layers/block/self_attn/") {
		return false
	}
	for _, p := range []string{"q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"} {
		if strings.HasSuffix(name, "/"+p+"/kernel") {
			return true
		}
	}
	return false
}
func finite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
func (m *Model) validateAdapter(a *Adapter) error {
	if a == nil || a.Rank < 1 || a.Rank > 256 || a.Scale <= 0 || !finite(a.Scale) || len(a.Weights) == 0 {
		return fmt.Errorf("needle: invalid adapter")
	}
	for name, w := range a.Weights {
		base, ok := m.tensors[name]
		if !ok || !loraTarget(name) {
			return fmt.Errorf("needle: invalid LoRA target %s", name)
		}
		l, k, n := base.Shape[0], base.Shape[1], base.Shape[2]
		if !slices.Equal(w.A.Shape, []int{l, k, a.Rank}) || !slices.Equal(w.B.Shape, []int{l, a.Rank, n}) || len(w.A.Data) != l*k*a.Rank || len(w.B.Data) != l*a.Rank*n {
			return fmt.Errorf("needle: adapter shape mismatch %s", name)
		}
		for _, data := range [][]float32{w.A.Data, w.B.Data} {
			for _, v := range data {
				if !finite(v) {
					return fmt.Errorf("needle: nonfinite adapter")
				}
			}
		}
	}
	return nil
}
func (m *Model) Merge(a *Adapter) (*Model, error) {
	if err := m.validateAdapter(a); err != nil {
		return nil, err
	}
	cp := m.Checkpoint()
	for name, w := range a.Weights {
		base := cp.Tensors[name]
		l, k, n := base.Shape[0], base.Shape[1], base.Shape[2]
		delta := make([]float32, k*n)
		for layer := 0; layer < l; layer++ {
			if !simd.MatMul(delta, w.A.Data[layer*k*a.Rank:], w.B.Data[layer*a.Rank*n:], k, n, a.Rank, false, false) {
				return nil, fmt.Errorf("needle: invalid LoRA multiplication")
			}
			simd.Saxpy(a.Scale, delta, base.Data[layer*k*n:(layer+1)*k*n])
		}
	}
	return New(cp)
}
func (m *Model) AdapterLossGrad(a *Adapter, ids []int, mask []float32, opts Options) (float64, map[string]checkpoint.Tensor, error) {
	merged, err := m.Merge(a)
	if err != nil {
		return 0, nil, err
	}
	loss, full, err := merged.LossGrad(ids, mask, opts)
	if err != nil {
		return 0, nil, err
	}
	out := map[string]checkpoint.Tensor{}
	for name, w := range a.Weights {
		base := m.tensors[name]
		l, k, n, r := base.Shape[0], base.Shape[1], base.Shape[2], a.Rank
		ga := checkpoint.Tensor{Shape: append([]int(nil), w.A.Shape...), Data: make([]float32, len(w.A.Data))}
		gb := checkpoint.Tensor{Shape: append([]int(nil), w.B.Shape...), Data: make([]float32, len(w.B.Data))}
		for layer := 0; layer < l; layer++ {
			dw := full[name].Data[layer*k*n:]
			simd.MatMul(ga.Data[layer*k*r:], dw, w.B.Data[layer*r*n:], k, r, n, false, true)
			simd.MatMul(gb.Data[layer*r*n:], w.A.Data[layer*k*r:], dw, r, n, k, true, false)
		}
		for i := range ga.Data {
			ga.Data[i] *= a.Scale
		}
		for i := range gb.Data {
			gb.Data[i] *= a.Scale
		}
		out[name+"/A"] = ga
		out[name+"/B"] = gb
	}
	return loss, out, nil
}

// AdamW is session-owned and not concurrent-safe. Step validates all values and
// computes a candidate update before publishing any parameter/moment changes.
type AdamW struct {
	Beta1, Beta2, Epsilon, WeightDecay, ClipNorm float64
	step                                         int
	m, v                                         map[string][]float32
}

func NewAdamW() *AdamW {
	return &AdamW{Beta1: .9, Beta2: .999, Epsilon: 1e-8, WeightDecay: 1e-4, ClipNorm: 1, m: map[string][]float32{}, v: map[string][]float32{}}
}
func (o *AdamW) update(weights map[string]checkpoint.Tensor, grads map[string]checkpoint.Tensor, lr float64) (map[string]checkpoint.Tensor, error) {
	if o == nil || o.Beta1 < 0 || o.Beta1 >= 1 || o.Beta2 < 0 || o.Beta2 >= 1 || o.Epsilon <= 0 || o.WeightDecay < 0 || o.ClipNorm < 0 || lr < 0 {
		return nil, fmt.Errorf("needle: invalid AdamW options")
	}
	for _, v := range []float64{o.Beta1, o.Beta2, o.Epsilon, o.WeightDecay, o.ClipNorm, lr} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("needle: nonfinite AdamW options")
		}
	}
	if len(weights) != len(grads) {
		return nil, fmt.Errorf("needle: gradient key count mismatch")
	}
	var norm float64
	for name, w := range weights {
		g, ok := grads[name]
		if !ok || !slices.Equal(w.Shape, g.Shape) || len(w.Data) != len(g.Data) {
			return nil, fmt.Errorf("needle: gradient shape mismatch %s", name)
		}
		if o.step > 0 && (len(o.m[name]) != len(w.Data) || len(o.v[name]) != len(w.Data)) {
			return nil, fmt.Errorf("needle: optimizer parameter set changed")
		}
		for i, v := range g.Data {
			if !finite(v) || !finite(w.Data[i]) {
				return nil, fmt.Errorf("needle: nonfinite training data")
			}
			norm += float64(v) * float64(v)
		}
	}
	scale := 1.
	norm = math.Sqrt(norm)
	if o.ClipNorm > 0 && norm > o.ClipNorm {
		scale = o.ClipNorm / norm
	}
	next := o.step + 1
	c1, c2 := 1-math.Pow(o.Beta1, float64(next)), 1-math.Pow(o.Beta2, float64(next))
	out := map[string]checkpoint.Tensor{}
	nm, nv := map[string][]float32{}, map[string][]float32{}
	for name, w := range weights {
		result := checkpoint.Tensor{Shape: append([]int{}, w.Shape...), Data: make([]float32, len(w.Data))}
		mm, vv := make([]float32, len(w.Data)), make([]float32, len(w.Data))
		for i, p := range w.Data {
			g := float64(grads[name].Data[i]) * scale
			pm, pv := float64(0), float64(0)
			if o.step > 0 {
				pm = float64(o.m[name][i])
				pv = float64(o.v[name][i])
			}
			mm[i] = float32(o.Beta1*pm + (1-o.Beta1)*g)
			vv[i] = float32(o.Beta2*pv + (1-o.Beta2)*g*g)
			u := float64(p) - lr*(float64(mm[i])/c1/(math.Sqrt(float64(vv[i])/c2)+o.Epsilon)+o.WeightDecay*float64(p))
			result.Data[i] = float32(u)
			if !finite(result.Data[i]) || !finite(mm[i]) || !finite(vv[i]) {
				return nil, fmt.Errorf("needle: nonfinite optimizer update")
			}
		}
		out[name] = result
		nm[name] = mm
		nv[name] = vv
	}
	o.step = next
	o.m, o.v = nm, nv
	return out, nil
}
func (m *Model) TrainStep(opt *AdamW, ids []int, mask []float32, lr float64, opts Options) (*Model, float64, error) {
	loss, g, err := m.LossGrad(ids, mask, opts)
	if err != nil {
		return nil, 0, err
	}
	weights := map[string]checkpoint.Tensor{}
	for name := range g {
		weights[name] = m.tensors[name]
	}
	updated, err := opt.update(weights, g, lr)
	if err != nil {
		return nil, 0, err
	}
	cp := m.Checkpoint()
	for name, w := range updated {
		cp.Tensors[name] = w
	}
	next, err := New(cp)
	return next, loss, err
}
func (m *Model) TrainAdapterStep(a *Adapter, opt *AdamW, ids []int, mask []float32, lr float64, opts Options) (*Adapter, float64, error) {
	loss, g, err := m.AdapterLossGrad(a, ids, mask, opts)
	if err != nil {
		return nil, 0, err
	}
	weights := map[string]checkpoint.Tensor{}
	for name, w := range a.Weights {
		weights[name+"/A"] = w.A
		weights[name+"/B"] = w.B
	}
	updated, err := opt.update(weights, g, lr)
	if err != nil {
		return nil, 0, err
	}
	out := &Adapter{Rank: a.Rank, Scale: a.Scale, Weights: map[string]LoRA{}}
	for name := range a.Weights {
		out.Weights[name] = LoRA{A: updated[name+"/A"], B: updated[name+"/B"]}
	}
	return out, loss, nil
}

// WarmupCosine is the schedule used by upstream local fine-tuning. It accepts
// an explicit warmup so the one-step edge case does not divide by zero.
func WarmupCosine(step, total, warmup int, peak float64) float64 {
	if total < 1 || step < 0 || step >= total || warmup < 0 || warmup >= total || peak < 0 || math.IsNaN(peak) || math.IsInf(peak, 0) {
		return 0
	}
	if warmup > 0 && step < warmup {
		return peak * float64(step) / float64(warmup)
	}
	return peak * .5 * (1 + math.Cos(math.Pi*float64(step-warmup)/float64(total-warmup)))
}
