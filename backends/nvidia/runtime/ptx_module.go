package nvidia

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// PTXModule owns a separately loaded module. Close must follow completion of
// its launches; the caller must not use returned functions afterwards. Global
// Shutdown also requires all such owners to have closed first.
type PTXModule struct {
	mu        sync.Mutex
	handle    CUmodule
	functions map[string]CUfunction
}

// LoadPTXFunctions loads one owned module for a fixed, nonempty function set.
// Failed construction unloads the module; no handle enters the global cache.
func LoadPTXFunctions(ptx string, names []string) (*PTXModule, error) {
	if ptx == "" || len(names) == 0 {
		return nil, fmt.Errorf("PTX and function names required")
	}
	seen := map[string]bool{}
	for _, name := range names {
		if name == "" || seen[name] {
			return nil, fmt.Errorf("empty/duplicate PTX function")
		}
		seen[name] = true
	}
	if !Init() {
		return nil, fmt.Errorf("NVIDIA unavailable")
	}
	release := lockDriver()
	defer release()
	mod, fn, err := loadPTXModuleLocked(ptx, names[0])
	if err != nil {
		return nil, err
	}
	result := &PTXModule{handle: mod, functions: map[string]CUfunction{names[0]: fn}}
	for _, name := range names[1:] {
		bytes := append([]byte(name), 0)
		var f CUfunction
		r := cuModuleGetFunction(&f, mod, unsafe.Pointer(&bytes[0]))
		runtime.KeepAlive(bytes)
		if r != CUDA_SUCCESS {
			cuModuleUnload(mod)
			return nil, fmt.Errorf("cuModuleGetFunction(%s): %d", name, r)
		}
		result.functions[name] = f
	}
	return result, nil
}
func (m *PTXModule) Function(name string) CUfunction {
	if m == nil {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle == 0 {
		return 0
	}
	return m.functions[name]
}
func (m *PTXModule) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle == 0 {
		return nil
	}
	release := lockDriver()
	defer release()
	if r := cuCtxSynchronize(); r != CUDA_SUCCESS {
		return fmt.Errorf("PTX module sync: %d", r)
	}
	if r := cuModuleUnload(m.handle); r != CUDA_SUCCESS {
		return fmt.Errorf("PTX module unload: %d", r)
	}
	m.handle = 0
	m.functions = nil
	return nil
}
