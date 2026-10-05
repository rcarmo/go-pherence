package pockettts

// flowShapeArena belongs to one graph builder and must not be shallow-copied
// when both builders will append. Returned slices occupy disjoint regions and
// cap=len isolates append. Retaining one shape pins at most 256 ints for valid
// rank<=3 shapes; invalid ranks keep the old owned-copy path for admission.
type flowShapeArena struct{ remaining []int }

func (a *flowShapeArena) copy(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	if len(shape) > 3 {
		return append([]int(nil), shape...)
	}
	if len(a.remaining) < len(shape) {
		a.remaining = make([]int, 256)
	}
	out := a.remaining[:len(shape):len(shape)]
	a.remaining = a.remaining[len(shape):]
	copy(out, shape)
	return out
}
