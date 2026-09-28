//go:build linux && (amd64 || arm64)

package nvidia

import (
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/ebitengine/purego"
)

func TestFixedCUDAKernelLauncherGPU(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_CUDA_LAUNCH") != "1" {
		t.Skip("set GO_PHERENCE_TEST_CUDA_LAUNCH=1 for native CUDA launch qualification")
	}
	if !Init() {
		t.Fatal("CUDA requested but unavailable")
	}
	lib, err := purego.Dlopen("libcuda.so.1", purego.RTLD_LAZY)
	if err != nil {
		t.Fatal(err)
	}
	defer purego.Dlclose(lib)
	m, err := LoadPTXFunctions(".version 7.0\n.target sm_52\n.address_size 64\n.visible .entry launch_binding_probe() { ret; }\n", []string{"launch_binding_probe"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := m.Close(); err != nil {
			t.Error(err)
		}
	}()
	fixed := cuLaunchKernel
	defer func() { cuLaunchKernel = fixed }()
	typed := fixed
	purego.RegisterLibFunc(&typed, lib, "cuLaunchKernel")
	cmds := make([]KernelLaunch, 256)
	for i := range cmds {
		cmds[i] = KernelLaunch{Function: m.Function("launch_binding_probe"), Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{1, 1, 1}}
	}
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "")
	allocations := func() float64 {
		return testing.AllocsPerRun(5, func() {
			if e := LaunchBatch(cmds); e != nil {
				panic(e)
			}
		})
	}
	cuLaunchKernel = typed
	before := allocations()
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	cuLaunchKernel = fixed
	after := allocations()
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	t.Logf("256-launch batch allocations typed=%g fixed=%g", before, after)
	// Context setup is still reflective. Normal execution permits four objects
	// per batch. The race runtime randomly discards sync.Pool entries, including
	// purego's syscall frames; test the relative reduction in that mode instead.
	if (!launchBindingRace && after > 4) || after > before/4 {
		t.Fatal("fixed launch allocation regression", before, after)
	}
	runtime.GC()
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 5 {
				local := []KernelLaunch{{Function: m.Function("launch_binding_probe"), Grid: [3]uint32{1, 1, 1}, Block: [3]uint32{1, 1, 1}}}
				if e := LaunchBatch(local); e != nil {
					t.Error(e)
				}
			}
		}()
	}
	wg.Wait()
	if err := SyncErr(); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GO_PHERENCE_CUDA_LAUNCH_CHECK", "1")
	if err := LaunchBatch(cmds[:1]); err != nil {
		t.Fatal(err)
	}
}
