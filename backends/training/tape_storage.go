package training

import "math"

// TapeStoragePlan assigns tensor/reshape families to equal-sized allocations.
// Constants, unwritten tensors and declared output families are retained.
// A slot is reusable only strictly after the preceding family's last consumer;
// no dispatch writes into its inputs. This is intra-tape, not cross-tape reuse.
type TapeStoragePlan struct {
	Slots    []int
	Elements []int
	Bytes    int64
}

func PlanTapeStorage(t Tape, maxBytes int64) (TapeStoragePlan, error) {
	return planTapeStorage(t, maxBytes, nil, false)
}

// PlanZeroTapeStorage additionally shares immutable positive-zero constants.
// Protected IDs are resident parameter bindings, including zero-valued ones.
// Source validation/byte admission is unchanged; zero slots remain pinned.
func PlanZeroTapeStorage(t Tape, maxBytes int64, protected map[int]bool) (TapeStoragePlan, error) {
	return planTapeStorage(t, maxBytes, protected, true)
}
func planTapeStorage(t Tape, maxBytes int64, protected map[int]bool, shareZero bool) (TapeStoragePlan, error) {
	var p TapeStoragePlan
	if err := t.Validate(maxBytes); err != nil {
		return p, err
	}
	n := len(t.Tensors)
	root := make([]int, n)
	birth, last := make([]int, n), make([]int, n)
	for i := range root {
		root[i] = i
		birth[i] = -1
		last[i] = -1
	}
	for index, c := range t.Commands {
		if c.Op == "reshape" {
			root[c.Output] = root[c.Inputs[0]]
		} else {
			birth[c.Output] = index
			last[c.Output] = index
		}
		for _, id := range c.Inputs {
			r := root[id]
			if index > last[r] {
				last[r] = index
			}
		}
	}
	for _, id := range t.Outputs {
		last[root[id]] = len(t.Commands)
	}
	p.Slots = make([]int, n)
	for i := range p.Slots {
		p.Slots[i] = -1
	}
	ends := []int{}
	allocate := func(r int, pinned bool) {
		extent, _ := t.Tensors[r].Elements()
		slot := -1
		if !pinned {
			for i, x := range p.Elements {
				if x == extent && ends[i] < birth[r] {
					slot = i
					break
				}
			}
		}
		if slot < 0 {
			slot = len(p.Elements)
			p.Elements = append(p.Elements, extent)
			ends = append(ends, 0)
			p.Bytes += int64(extent) * 4
		}
		p.Slots[r] = slot
		if pinned {
			ends[slot] = len(t.Commands)
		} else {
			ends[slot] = last[r]
		}
	}
	zeroSlots := map[int]int{}
	// Initial values upload before execution, so constants cannot reuse even if
	// their last consumer is early. Unwritten/dead tensors remain conservatively
	// allocated, matching source admission instead of implicitly pruning here.
	for r := range root {
		if root[r] == r && birth[r] < 0 {
			if shareZero && !protected[r] && positiveZeroConstant(t.Tensors[r].Values) {
				n, _ := t.Tensors[r].Elements()
				if slot, ok := zeroSlots[n]; ok {
					p.Slots[r] = slot
				} else {
					allocate(r, true)
					zeroSlots[n] = p.Slots[r]
				}
			} else {
				allocate(r, true)
			}
		}
	}
	for _, c := range t.Commands {
		if c.Op != "reshape" {
			allocate(c.Output, false)
		}
	}
	for i := range root {
		p.Slots[i] = p.Slots[root[i]]
	}
	return p, nil
}

// DeclaredTapeOutput restricts reusable storage exposure to terminal roots.
func DeclaredTapeOutput(outputs []int, id int) bool {
	for _, out := range outputs {
		if out == id {
			return true
		}
	}
	return false
}

func positiveZeroConstant(x []float32) bool {
	if len(x) == 0 {
		return false
	}
	for _, v := range x {
		if v != 0 || math.Signbit(float64(v)) {
			return false
		}
	}
	return true
}
