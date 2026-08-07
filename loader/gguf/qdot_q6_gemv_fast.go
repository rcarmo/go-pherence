package gguf

import (
	"encoding/binary"

	"github.com/rcarmo/go-pherence/half"
)

func dotQ6KQ8KGemvFast(raw []byte, y []q8KBlock, blocks int) float32 {
	var sum float32
	for bi := 0; bi < blocks; bi++ {
		blk := raw[bi*210 : (bi+1)*210]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:])) * y[bi].d
		sum += d * float32(q6KBlockDot((*[210]byte)(blk), &y[bi].qs))
	}
	return sum
}
