package needle

// A deliberately small reverse-mode tape for Needle's FP32 CPU path. Matrix
// products dispatch through the same checked SIMD primitive in both directions.
import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

type value struct {
	x, g []float32
	r, c int
}
type tape struct {
	train       bool
	used, limit int64
	backward    []func()
	arena       *inferenceArena
}
type workLimit struct{ error }

func (t *tape) alloc(r, c int) *value {
	if r <= 0 || c <= 0 || int64(r) > t.limit/4/int64(c) {
		panic(workLimit{fmt.Errorf("needle: invalid/excessive workspace shape %dx%d", r, c)})
	}
	bytes := int64(r) * int64(c) * 4
	if t.train {
		bytes *= 2
	}
	bytes += 128 // Tensor object and backing-slice bookkeeping.
	if bytes > t.limit-t.used {
		panic(workLimit{fmt.Errorf("needle: workspace budget exceeded (%d bytes)", t.limit)})
	}
	t.used += bytes
	if t.arena != nil && !t.train {
		v := t.arena.alloc(r * c)
		v.r, v.c = r, c
		return v
	}
	v := &value{r: r, c: c}
	if t.train {
		storage := make([]float32, 2*r*c)
		v.x, v.g = storage[:r*c:r*c], storage[r*c:]
	} else {
		v.x = make([]float32, r*c)
	}
	return v
}
func (t *tape) reserve(n int64) {
	if n < 0 || n > t.limit-t.used {
		panic(workLimit{fmt.Errorf("needle: workspace budget exceeded (%d bytes)", t.limit)})
	}
	t.used += n
}
func (t *tape) ints(n int) []int {
	t.reserve(int64(n) * 8)
	if t.arena != nil && !t.train {
		return t.arena.ints(n)
	}
	return make([]int, n)
}
func (t *tape) pointers(n int) []*value {
	if t.arena != nil && !t.train {
		return t.arena.pointers(n)
	}
	return make([]*value, n)
}
func (t *tape) record(f func()) {
	if t.train {
		t.reserve(256) // Conservative closure/slice bookkeeping charge, not RSS.
		t.backward = append(t.backward, f)
	}
}
func (t *tape) view(x []float32, r, c int) *value {
	t.reserve(128)
	var v *value
	if t.arena != nil {
		v = t.arena.node()
	} else {
		v = &value{}
	}
	v.x, v.r, v.c = x, r, c
	return v
}
func (t *tape) leaf(x []float32) *value { v := t.alloc(1, len(x)); copy(v.x, x); return v }
func (t *tape) back() {
	for i := len(t.backward) - 1; i >= 0; i-- {
		t.backward[i]()
	}
}
func (t *tape) gather(a *value, r, c int, indices []int) *value {
	if t.train {
		t.reserve(int64(len(indices)) * 8)
		indices = append([]int(nil), indices...)
	}
	o := t.alloc(r, c)
	for i, j := range indices {
		if j >= 0 {
			o.x[i] = a.x[j]
		}
	}
	if t.train {
		t.record(func() {
			for i, j := range indices {
				if j >= 0 {
					a.g[j] += o.g[i]
				}
			}
		})
	}
	return o
}
func (t *tape) slice(a *value, start, r, c int) *value {
	if !t.train {
		return t.view(a.x[start:start+r*c:start+r*c], r, c)
	}
	o := t.alloc(r, c)
	copy(o.x, a.x[start:start+r*c])
	t.record(func() { simd.Saxpy(1, o.g, a.g[start:start+r*c]) })
	return o
}
func (t *tape) rows(a *value, start, n int) *value { return t.slice(a, start*a.c, n, a.c) }
func (t *tape) cols(a *value, start, n int) *value {
	if !t.train {
		if a.r == 1 {
			return t.slice(a, start, 1, n)
		}
		o := t.alloc(a.r, n)
		for row := 0; row < a.r; row++ {
			copy(o.x[row*n:], a.x[row*a.c+start:row*a.c+start+n])
		}
		return o
	}
	o := t.alloc(a.r, n)
	for row := 0; row < a.r; row++ {
		copy(o.x[row*n:], a.x[row*a.c+start:row*a.c+start+n])
	}
	t.record(func() {
		for row := 0; row < a.r; row++ {
			simd.Saxpy(1, o.g[row*n:(row+1)*n], a.g[row*a.c+start:row*a.c+start+n])
		}
	})
	return o
}
func (t *tape) transpose(a *value) *value {
	if !t.train {
		o := t.alloc(a.c, a.r)
		for i := 0; i < a.r; i++ {
			for j := 0; j < a.c; j++ {
				o.x[j*a.r+i] = a.x[i*a.c+j]
			}
		}
		return o
	}
	o := t.alloc(a.c, a.r)
	for i := 0; i < a.r; i++ {
		for j := 0; j < a.c; j++ {
			o.x[j*a.r+i] = a.x[i*a.c+j]
		}
	}
	t.record(func() {
		for i := 0; i < a.r; i++ {
			for j := 0; j < a.c; j++ {
				a.g[i*a.c+j] += o.g[j*a.r+i]
			}
		}
	})
	return o
}
func (t *tape) broadcast(a *value, r int) *value {
	if !t.train {
		if r == 1 {
			return t.slice(a, 0, 1, len(a.x))
		}
		o := t.alloc(r, len(a.x))
		for row := 0; row < r; row++ {
			copy(o.x[row*len(a.x):], a.x)
		}
		return o
	}
	o := t.alloc(r, len(a.x))
	for row := 0; row < r; row++ {
		copy(o.x[row*len(a.x):], a.x)
	}
	t.record(func() {
		for row := 0; row < r; row++ {
			simd.Saxpy(1, o.g[row*len(a.x):(row+1)*len(a.x)], a.g)
		}
	})
	return o
}
func (t *tape) add(a, b *value) *value {
	o := t.alloc(a.r, a.c)
	simd.VecAdd(o.x, a.x, b.x)
	if t.train {
		t.record(func() { simd.Saxpy(1, o.g, a.g); simd.Saxpy(1, o.g, b.g) })
	}
	return o
}
func (t *tape) scale(a *value, s float32) *value {
	o := t.alloc(a.r, a.c)
	simd.VecScale(o.x, a.x, s)
	if t.train {
		t.record(func() { simd.Saxpy(s, o.g, a.g) })
	}
	return o
}
func (t *tape) mul(a, b *value) *value {
	o := t.alloc(a.r, a.c)
	simd.VecMul(o.x, a.x, b.x)
	if t.train {
		t.record(func() {
			for i, g := range o.g {
				a.g[i] += g * b.x[i]
				b.g[i] += g * a.x[i]
			}
		})
	}
	return o
}
func (t *tape) constant(r, c int, v float32) *value {
	o := t.alloc(r, c)
	for i := range o.x {
		o.x[i] = v
	}
	return o
}
func (t *tape) unary(a *value, kind string) *value {
	o := t.alloc(a.r, a.c)
	for i, x := range a.x {
		switch kind {
		case "sigmoid":
			o.x[i] = sigmoid(x)
		case "silu":
			o.x[i] = x * sigmoid(x)
		case "exp":
			o.x[i] = float32(math.Exp(float64(x)))
		}
	}
	if t.train {
		t.record(func() {
			for i, x := range a.x {
				var d float32
				switch kind {
				case "sigmoid":
					d = o.x[i] * (1 - o.x[i])
				case "silu":
					s := sigmoid(x)
					d = s * (1 + x*(1-s))
				case "exp":
					d = o.x[i]
				}
				a.g[i] += o.g[i] * d
			}
		})
	}
	return o
}
func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }
func (t *tape) mm(a, b *value, transB bool) *value {
	n, k := b.c, a.c
	if transB {
		n = b.r
	}
	o := t.alloc(a.r, n)
	if !simd.MatMul(o.x, a.x, b.x, a.r, n, k, false, transB) {
		panic("needle: internal matrix shape mismatch")
	}
	if t.train {
		t.reserve(int64(len(a.x)+len(b.x)) * 4)
	}
	if t.train {
		t.record(func() {
			da := make([]float32, len(a.x))
			db := make([]float32, len(b.x))
			if !simd.MatMul(da, o.g, b.x, a.r, k, n, false, !transB) {
				panic("needle: invalid input gradient")
			}
			if transB {
				if !simd.MatMul(db, o.g, a.x, n, k, a.r, true, false) {
					panic("needle: invalid weight gradient")
				}
			} else {
				if !simd.MatMul(db, a.x, o.g, k, n, a.r, true, false) {
					panic("needle: invalid weight gradient")
				}
			}
			simd.Saxpy(1, da, a.g)
			simd.Saxpy(1, db, b.g)
		})
	}
	return o
}
func (t *tape) rms(a *value) *value {
	o := t.alloc(a.r, a.c)
	t.reserve(int64(a.r) * 4)
	inv := make([]float32, a.r)
	if t.arena != nil && !t.train {
		inv = t.arena.alloc(a.r).x
	}
	for r := 0; r < a.r; r++ {
		x := a.x[r*a.c : (r+1)*a.c]
		ss := simd.Sdot(x, x) / float32(a.c)
		inv[r] = float32(1 / math.Sqrt(float64(ss+1e-6)))
		for j, v := range x {
			o.x[r*a.c+j] = v * inv[r]
		}
	}
	if t.train {
		t.record(func() {
			for r := 0; r < a.r; r++ {
				off := r * a.c
				d := simd.Sdot(o.g[off:off+a.c], a.x[off:off+a.c]) / float32(a.c)
				iv := inv[r]
				for j := 0; j < a.c; j++ {
					a.g[off+j] += iv * (o.g[off+j] - a.x[off+j]*d*iv*iv)
				}
			}
		})
	}
	return o
}
func (t *tape) norm(a, s *value) *value {
	if !t.train {
		o := t.rms(a)
		for i := range o.x {
			o.x[i] *= 1 + s.x[i%a.c]
		}
		return o
	}
	return t.mul(t.rms(a), t.broadcast(t.add(s, t.constant(s.r, s.c, 1)), a.r))
}

// window=0 is full causal attention; causal=false is unrestricted row softmax.
func (t *tape) softmax(a *value, causal bool, window int) *value {
	o := t.alloc(a.r, a.c)
	for r := 0; r < a.r; r++ {
		lo, hi := 0, a.c
		if causal {
			hi = r + 1
			if window > 0 && hi > window {
				lo = hi - window
			}
		}
		max := float32(math.Inf(-1))
		for c := lo; c < hi; c++ {
			if a.x[r*a.c+c] > max {
				max = a.x[r*a.c+c]
			}
		}
		var sum float32
		for c := lo; c < hi; c++ {
			v := float32(math.Exp(float64(a.x[r*a.c+c] - max)))
			o.x[r*a.c+c] = v
			sum += v
		}
		for c := lo; c < hi; c++ {
			o.x[r*a.c+c] /= sum
		}
	}
	if t.train {
		t.record(func() {
			for r := 0; r < a.r; r++ {
				off := r * a.c
				d := simd.Sdot(o.g[off:off+a.c], o.x[off:off+a.c])
				for c := 0; c < a.c; c++ {
					a.g[off+c] += o.x[off+c] * (o.g[off+c] - d)
				}
			}
		})
	}
	return o
}
func (t *tape) concat(parts []*value) *value {
	r, c := parts[0].r, 0
	for _, p := range parts {
		c += p.c
	}
	o := t.alloc(r, c)
	base := 0
	for _, p := range parts {
		for row := 0; row < r; row++ {
			copy(o.x[row*c+base:], p.x[row*p.c:(row+1)*p.c])
		}
		base += p.c
	}
	if t.train {
		t.record(func() {
			base := 0
			for _, p := range parts {
				for row := 0; row < r; row++ {
					simd.Saxpy(1, o.g[row*c+base:row*c+base+p.c], p.g[row*p.c:(row+1)*p.c])
				}
				base += p.c
			}
		})
	}
	return o
}
func (t *tape) shifted(a *value, offset, window int) *value {
	idx := t.ints(len(a.x))
	for r := 0; r < a.r; r++ {
		for c := 0; c < a.c; c++ {
			j := -1
			if r >= offset && (window == 0 || offset < window) {
				j = (r-offset)*a.c + c
			}
			idx[r*a.c+c] = j
		}
	}
	return t.gather(a, a.r, a.c, idx)
}
func (t *tape) conv(a, taps *value, dilation, window int) *value {
	o := t.constant(a.r, a.c, 0)
	for j := 0; j < taps.r; j++ {
		o = t.add(o, t.mul(t.shifted(a, j*dilation, window), t.broadcast(t.rows(taps, j, 1), a.r)))
	}
	return o
}

// Sinkhorn uses exactly the upstream twenty row/column log-normalization steps.
func (t *tape) logsoftmax(a *value) *value {
	o := t.alloc(a.r, a.c)
	for r := 0; r < a.r; r++ {
		off := r * a.c
		max := a.x[off]
		for _, v := range a.x[off : off+a.c] {
			if v > max {
				max = v
			}
		}
		var sum float64
		for _, v := range a.x[off : off+a.c] {
			sum += math.Exp(float64(v - max))
		}
		lse := max + float32(math.Log(sum))
		for c := 0; c < a.c; c++ {
			o.x[off+c] = a.x[off+c] - lse
		}
	}
	if t.train {
		t.record(func() {
			for r := 0; r < a.r; r++ {
				off := r * a.c
				var sum float32
				for _, v := range o.g[off : off+a.c] {
					sum += v
				}
				for c := 0; c < a.c; c++ {
					a.g[off+c] += o.g[off+c] - float32(math.Exp(float64(o.x[off+c])))*sum
				}
			}
		})
	}
	return o
}
