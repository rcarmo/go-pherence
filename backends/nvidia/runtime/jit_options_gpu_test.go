package nvidia

import (
	"os"
	"runtime"
	"strings"
	"testing"
	"unsafe"
)

func TestJITModuleOptionsGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_CUDA_JIT") != "1" || os.Getenv("GO_PHERENCE_DISABLE_NVIDIA") != "" {
		t.Skip("requires explicit CUDA JIT validation")
	}
	if !Available() {
		t.Fatal("CUDA JIT requested but NVIDIA runtime unavailable")
	}
	if cuModuleLoadDataEx == nil {
		t.Fatal("cuModuleLoadDataEx unavailable")
	}
	t.Cleanup(Shutdown)
	// Valid PTX also exercises retained module lifetime and teardown.
	_, err := LoadPTX(".version 7.0\n.target sm_52\n.address_size 64\n.visible .entry layout_jit_probe() { ret; }\n", "layout_jit_probe")
	if err != nil {
		t.Fatal(err)
	}

	// Verify CUDA interprets the same option slots as a buffer pointer and a
	// by-value byte count, and really writes the diagnostic into that buffer.
	image := append([]byte(".version 7.0\n.target sm_52\n.address_size 64\nTHIS_IS_INVALID_PTX\n"), 0)
	log := make([]byte, 8192)
	values := jitErrorLogOptions(log)
	opts := [2]uint32{5, 6} // CU_JIT_ERROR_LOG_BUFFER / _SIZE_BYTES
	var mod CUmodule
	runtime.LockOSThread()
	cudaMu.Lock()
	ensureContextLocked()
	result := cuModuleLoadDataEx(&mod, unsafe.Pointer(&image[0]), 2, unsafe.Pointer(&opts[0]), unsafe.Pointer(&values))
	cudaMu.Unlock()
	runtime.UnlockOSThread()
	runtime.KeepAlive(image)
	runtime.KeepAlive(log)
	if result == CUDA_SUCCESS {
		t.Fatal("invalid PTX was accepted")
	}
	if strings.Trim(string(log), "\x00\n\r ") == "" {
		t.Fatalf("CUDA rejected PTX (%d) but did not fill JIT error log", result)
	}
}
