package nvidia

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestJITErrorLogOptionsABI(t *testing.T) {
	for _, size := range []int{0, 1, 8192} {
		buf := make([]byte, size)
		vals := jitErrorLogOptions(buf)
		if vals.buffer != unsafe.Pointer(unsafe.SliceData(buf)) || vals.size != uintptr(size) {
			t.Fatalf("size=%d values=%v", size, vals)
		}
		if unsafe.Sizeof(vals) != 2*unsafe.Sizeof(uintptr(0)) || unsafe.Offsetof(vals.size) != unsafe.Sizeof(uintptr(0)) {
			t.Fatal("JIT options are not pointer-sized slots")
		}
		runtime.KeepAlive(buf)
	}
}
