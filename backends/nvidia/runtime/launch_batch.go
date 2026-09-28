package nvidia

import (
	"fmt"
	"os"
	"runtime"
	"unsafe"
)

// KernelLaunch holds one launch and its owned scalar argument storage. Args are
// the bit patterns for at most eight by-value arguments, each at most 64 bits.
// Pointers use CUdeviceptr values; float32 uses math.Float32bits. This matches
// the little-endian CUDA driver ABI of the supported Linux amd64/arm64 hosts.
// Do not copy or mutate a launch while LaunchBatch is running. No device memory
// is owned by this structure; callers keep referenced buffers/modules alive.
type KernelLaunch struct {
	Function    CUfunction
	Grid, Block [3]uint32
	SharedMem   uint32
	ArgCount    int
	Args        [8]uint64
	pointers    [8]unsafe.Pointer
}

// LaunchBatch validates every command before launching any, then enqueues in
// order on the current capture/default stream under one pinned driver scope.
// A driver error may follow earlier launches; callers must synchronise before
// freeing resources. An empty batch is a no-op. This does not synchronise on
// success unless the explicit diagnostic launch-check mode is enabled.
func LaunchBatch(commands []KernelLaunch) error {
	if len(commands) == 0 {
		return nil
	}
	if cuLaunchKernel == nil {
		return fmt.Errorf("cuLaunchKernel unavailable")
	}
	for i := range commands {
		c := &commands[i]
		if c.Function == 0 || c.ArgCount < 0 || c.ArgCount > len(c.Args) || c.Grid[0] == 0 || c.Grid[1] == 0 || c.Grid[2] == 0 || c.Block[0] == 0 || c.Block[1] == 0 || c.Block[2] == 0 {
			return fmt.Errorf("invalid CUDA batch launch %d", i)
		}
		for j := 0; j < c.ArgCount; j++ {
			c.pointers[j] = unsafe.Pointer(&c.Args[j])
		}
	}
	release := lockDriver()
	defer release()
	stream := uintptr(captureLaunchStream)
	check := stream == 0 && os.Getenv("GO_PHERENCE_CUDA_LAUNCH_CHECK") == "1"
	stats := gpuStatsEnabled.Load()
	for i := range commands {
		c := &commands[i]
		var ptr unsafe.Pointer
		if c.ArgCount > 0 {
			ptr = unsafe.Pointer(&c.pointers[0])
		}
		if r := cuLaunchKernel(c.Function, c.Grid[0], c.Grid[1], c.Grid[2], c.Block[0], c.Block[1], c.Block[2], c.SharedMem, stream, ptr, nil); r != CUDA_SUCCESS {
			return fmt.Errorf("CUDA batch launch %d: %d", i, r)
		}
		if stats {
			gpuStatsKernelLaunches.Add(1)
		}
		if check {
			if r := cuCtxSynchronize(); r != CUDA_SUCCESS {
				return fmt.Errorf("CUDA batch launch %d sync: %d", i, r)
			}
		}
	}
	runtime.KeepAlive(commands)
	return nil
}
