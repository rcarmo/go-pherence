package nvidia

import (
	"strings"
	"testing"
	"unsafe"
)

func TestDiagnosticLaunchCheckOptInAndCaptureExclusion(t *testing.T) {
	oldLaunch, oldSync, oldOK, oldCapture := cuLaunchKernel, cuCtxSynchronize, gpuOK, captureLaunchStream
	defer func() {
		cuLaunchKernel, cuCtxSynchronize, gpuOK, captureLaunchStream = oldLaunch, oldSync, oldOK, oldCapture
	}()
	gpuOK = false
	captureLaunchStream = 0
	syncs := 0
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult { syncs++; return 719 }
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "")
	if err := LaunchKernel(1, 1, 1, 1, 16, 16, 1, 0); err != nil || syncs != 0 {
		t.Fatal("default altered", err, syncs)
	}
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "1")
	if err := LaunchKernel(1, 1, 1, 1, 16, 16, 1, 0); err == nil || !strings.Contains(err.Error(), "post-launch sync") || syncs != 1 {
		t.Fatal("missing launch attribution", err, syncs)
	}
	captureLaunchStream = 1
	if err := LaunchKernel(1, 1, 1, 1, 16, 16, 1, 0); err != nil || syncs != 1 {
		t.Fatal("graph capture synchronised", err, syncs)
	}
}
