package needle

import "fmt"

// inferenceArena groups request/session-local scratch. Blocks never move, so
// read-only views remain valid for one execution. Reset retains bounded decoder
// blocks and clears used storage before the next transactional step.
type inferenceArena struct {
	limit, bytes             int64
	floatBlocks              [][]float32
	intBlocks                [][]int
	valueBlocks              [][]value
	pointerBlocks            [][]*value
	floatBlock, floatOff     int
	intBlock, intOff         int
	valueBlock, valueOff     int
	pointerBlock, pointerOff int
}

func (a *inferenceArena) charge(n int64) {
	if n < 0 || n > a.limit-a.bytes {
		panic(workLimit{fmt.Errorf("needle: inference arena budget exceeded")})
	}
	a.bytes += n
}
func (a *inferenceArena) alloc(n int) *value {
	for a.floatBlock < len(a.floatBlocks) && len(a.floatBlocks[a.floatBlock])-a.floatOff < n {
		a.floatBlock++
		a.floatOff = 0
	}
	if a.floatBlock == len(a.floatBlocks) {
		size := max(n, 1024)
		a.charge(int64(size)*4 + 128)
		a.floatBlocks = append(a.floatBlocks, make([]float32, size))
	}
	block := a.floatBlocks[a.floatBlock]
	x := block[a.floatOff : a.floatOff+n : a.floatOff+n]
	a.floatOff += n
	v := a.node()
	v.x = x
	return v
}
func (a *inferenceArena) ints(n int) []int {
	for a.intBlock < len(a.intBlocks) && len(a.intBlocks[a.intBlock])-a.intOff < n {
		a.intBlock++
		a.intOff = 0
	}
	if a.intBlock == len(a.intBlocks) {
		size := max(n, 1024)
		a.charge(int64(size)*8 + 128)
		a.intBlocks = append(a.intBlocks, make([]int, size))
	}
	block := a.intBlocks[a.intBlock]
	x := block[a.intOff : a.intOff+n : a.intOff+n]
	a.intOff += n
	return x
}
func (a *inferenceArena) pointers(n int) []*value {
	for a.pointerBlock < len(a.pointerBlocks) && len(a.pointerBlocks[a.pointerBlock])-a.pointerOff < n {
		a.pointerBlock++
		a.pointerOff = 0
	}
	if a.pointerBlock == len(a.pointerBlocks) {
		size := max(n, 128)
		a.charge(int64(size)*8 + 128)
		a.pointerBlocks = append(a.pointerBlocks, make([]*value, size))
	}
	block := a.pointerBlocks[a.pointerBlock]
	out := block[a.pointerOff : a.pointerOff+n : a.pointerOff+n]
	a.pointerOff += n
	return out
}
func (a *inferenceArena) node() *value {
	for a.valueBlock < len(a.valueBlocks) && a.valueOff == len(a.valueBlocks[a.valueBlock]) {
		a.valueBlock++
		a.valueOff = 0
	}
	if a.valueBlock == len(a.valueBlocks) {
		size := 64
		if int64(size)*64+128 > a.limit-a.bytes {
			size = 1
		}
		a.charge(int64(size)*64 + 128)
		a.valueBlocks = append(a.valueBlocks, make([]value, size))
	}
	v := &a.valueBlocks[a.valueBlock][a.valueOff]
	a.valueOff++
	*v = value{}
	return v
}
func (a *inferenceArena) reset() {
	for bi := 0; bi <= a.floatBlock && bi < len(a.floatBlocks); bi++ {
		end := len(a.floatBlocks[bi])
		if bi == a.floatBlock {
			end = a.floatOff
		}
		clear(a.floatBlocks[bi][:end])
	}
	for bi := 0; bi <= a.intBlock && bi < len(a.intBlocks); bi++ {
		end := len(a.intBlocks[bi])
		if bi == a.intBlock {
			end = a.intOff
		}
		clear(a.intBlocks[bi][:end])
	}
	for bi := 0; bi <= a.pointerBlock && bi < len(a.pointerBlocks); bi++ {
		end := len(a.pointerBlocks[bi])
		if bi == a.pointerBlock {
			end = a.pointerOff
		}
		clear(a.pointerBlocks[bi][:end])
	}
	for bi := 0; bi <= a.valueBlock && bi < len(a.valueBlocks); bi++ {
		end := len(a.valueBlocks[bi])
		if bi == a.valueBlock {
			end = a.valueOff
		}
		clear(a.valueBlocks[bi][:end])
	}
	a.floatBlock, a.floatOff, a.intBlock, a.intOff, a.valueBlock, a.valueOff, a.pointerBlock, a.pointerOff = 0, 0, 0, 0, 0, 0, 0, 0
}
