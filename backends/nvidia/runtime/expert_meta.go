package nvidia

import (
	"fmt"
	"unsafe"
)

var fnExpertMetaReduce CUfunction

func ExpertMetaReduce(offsets, weights, counts, sums *Buffer, groups int) error {
	if groups <= 0 {
		return nil
	}
	if offsets == nil || weights == nil || counts == nil || sums == nil || offsets.Ptr == 0 || weights.Ptr == 0 || counts.Ptr == 0 || sums.Ptr == 0 || offsets.Size < (groups+1)*4 || counts.Size < groups*4 || sums.Size < groups*4 {
		return fmt.Errorf("invalid expert meta reduce buffers groups=%d", groups)
	}
	if fnExpertMetaReduce == 0 {
		return fmt.Errorf("expert meta reduce kernel not loaded")
	}
	g := uint32(groups)
	return LaunchKernel(fnExpertMetaReduce, uint32(groups), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&offsets.Ptr), unsafe.Pointer(&weights.Ptr), unsafe.Pointer(&counts.Ptr), unsafe.Pointer(&sums.Ptr), unsafe.Pointer(&g))
}
