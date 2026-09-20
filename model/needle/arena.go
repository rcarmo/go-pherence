package needle

import "fmt"

// inferenceArena groups short-lived objects without pooling across requests.
// Blocks never move, so read-only tensor views stay valid for the whole step.
// A separate physical allocation counter includes block capacity/metadata; the
// tape's existing logical budget still accounts operations, indices and scratch.
type inferenceArena struct {
	limit, bytes int64
	floats       []float32
	values       []value
}

func (a *inferenceArena) charge(n int64) {
	if n < 0 || n > a.limit-a.bytes {
		panic(workLimit{fmt.Errorf("needle: inference arena budget exceeded")})
	}
	a.bytes += n
}
func (a *inferenceArena) alloc(n int) *value {
	if len(a.floats) < n {
		size := max(n, 1024)
		remaining := (a.limit - a.bytes - 128) / 4
		if remaining < int64(size) {
			size = n
		}
		a.charge(int64(size)*4 + 128)
		a.floats = make([]float32, size)
	}
	v := a.node()
	v.x = a.floats[:n:n]
	a.floats = a.floats[n:]
	return v
}
func (a *inferenceArena) node() *value {
	if len(a.values) == 0 {
		size := 32
		if int64(size)*64+128 > a.limit-a.bytes {
			size = 1
		}
		a.charge(int64(size)*64 + 128)
		a.values = make([]value, size)
	}
	v := &a.values[0]
	a.values = a.values[1:]
	return v
}
