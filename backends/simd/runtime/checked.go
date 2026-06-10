package simd

import "github.com/rcarmo/go-pherence/internal/checked"

const int32Max = int32(^uint32(0) >> 1)

func checkedFloat32ByteOffset(index int) (uintptr, bool) {
	off, ok := checked.MulInt(index, 4)
	if !ok {
		return 0, false
	}
	return uintptr(off), true
}
