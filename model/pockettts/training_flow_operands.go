package pockettts

// Each builder owns its operand arena. Full-slice expressions isolate writes
// and appends between commands; retained slices pin at most 256 integers for
// supported command arities. Larger invalid lists keep independent storage.
type flowOperandArena struct{ remaining []int }

func (a *flowOperandArena) copy(inputs []int) []int {
	if len(inputs) == 0 {
		return nil
	}
	if len(inputs) > 8 {
		out := append([]int(nil), inputs...)
		return out[:len(out):len(out)]
	}
	if len(a.remaining) < len(inputs) {
		a.remaining = make([]int, 256)
	}
	out := a.remaining[:len(inputs):len(inputs)]
	a.remaining = a.remaining[len(inputs):]
	copy(out, inputs)
	return out
}
