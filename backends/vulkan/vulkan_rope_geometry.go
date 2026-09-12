package vulkan

import "fmt"

// Raw32bit fields exactly match the pair-owned shader's push block.
type vkRoPEPush struct{ Pos, Heads, HeadDim, RotHalf uint32 }

func vkRoPEGeometry(pos, heads, headDim, rotHalf int) (vkRoPEPush, uint32, int, int, error) {
	invalid := func() (vkRoPEPush, uint32, int, int, error) {
		return vkRoPEPush{}, 0, 0, 0, fmt.Errorf("invalid Vulkan RoPE geometry pos=%d heads=%d headDim=%d rotHalf=%d", pos, heads, headDim, rotHalf)
	}
	const maxU32 = uint64(1<<32 - 1)
	if pos < 0 || heads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 {
		return invalid()
	}
	for _, n := range []int{pos, heads, headDim, rotHalf} {
		if uint64(n) > maxU32 {
			return invalid()
		}
	}
	// Widen before pos+1/products. Operand bounds above also prevent uint64
	// overflow; divide before multiply for the doubled frequency extent.
	total := uint64(heads) * uint64(headDim)
	pairs := uint64(heads) * uint64(rotHalf)
	frequencyRows := (uint64(pos) + 1) * uint64(rotHalf)
	maxInt := uint64(int(^uint(0) >> 1))
	if total > maxU32 || pairs > maxU32 || frequencyRows > maxU32/2 || total > maxInt || frequencyRows > maxInt/2 {
		return invalid()
	}
	return vkRoPEPush{uint32(pos), uint32(heads), uint32(headDim), uint32(rotHalf)}, uint32((pairs + 255) / 256), int(total), int(frequencyRows * 2), nil
}
