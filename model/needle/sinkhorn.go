package needle

import "math"

// sinkhorn retains the training graph, but evaluates inference in one owned
// matrix instead of allocating four transpose/logsoftmax nodes per iteration.
// Arithmetic order and twenty row/column log-normalization passes are unchanged.
func (t *tape) sinkhorn(a *value) *value {
	if t.train {
		for iter := 0; iter < 20; iter++ {
			a = t.transpose(t.logsoftmax(t.transpose(t.logsoftmax(a))))
		}
		return t.unary(a, "exp")
	}
	o := t.alloc(a.r, a.c)
	copy(o.x, a.x)
	for iter := 0; iter < 20; iter++ {
		for r := 0; r < o.r; r++ {
			off := r * o.c
			mx := o.x[off]
			for j := 0; j < o.c; j++ {
				if o.x[off+j] > mx {
					mx = o.x[off+j]
				}
			}
			var sum float64
			for j := 0; j < o.c; j++ {
				sum += math.Exp(float64(o.x[off+j] - mx))
			}
			lse := mx + float32(math.Log(sum))
			for j := 0; j < o.c; j++ {
				o.x[off+j] -= lse
			}
		}
		for c := 0; c < o.c; c++ {
			mx := o.x[c]
			for i := 0; i < o.r; i++ {
				if o.x[i*o.c+c] > mx {
					mx = o.x[i*o.c+c]
				}
			}
			var sum float64
			for i := 0; i < o.r; i++ {
				sum += math.Exp(float64(o.x[i*o.c+c] - mx))
			}
			lse := mx + float32(math.Log(sum))
			for i := 0; i < o.r; i++ {
				o.x[i*o.c+c] -= lse
			}
		}
	}
	for i, v := range o.x {
		o.x[i] = float32(math.Exp(float64(v)))
	}
	return o
}
