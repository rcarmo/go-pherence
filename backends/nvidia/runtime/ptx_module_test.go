package nvidia

import (
	"strings"
	"testing"
)

func TestOwnedPTXModuleCloseRetries(t *testing.T) {
	oldCtx, oldSet, oldSync, oldUnload := gpuCtx, cuCtxSetCurrent, cuCtxSynchronize, cuModuleUnload
	defer func() { gpuCtx, cuCtxSetCurrent, cuCtxSynchronize, cuModuleUnload = oldCtx, oldSet, oldSync, oldUnload }()
	gpuCtx = 0
	syncFail, unloadFail := true, false
	unloads := 0
	check := func() {
		t.Helper()
		if cudaMu.TryLock() {
			cudaMu.Unlock()
			t.Fatal("driver operation outside lock")
		}
	}
	cuCtxSynchronize = func() CUresult {
		check()
		if syncFail {
			return 1
		}
		return CUDA_SUCCESS
	}
	cuModuleUnload = func(CUmodule) CUresult {
		check()
		unloads++
		if unloadFail {
			return 2
		}
		return CUDA_SUCCESS
	}
	m := &PTXModule{handle: 9, functions: map[string]CUfunction{"entry": 10}}
	if err := m.Close(); err == nil || !strings.Contains(err.Error(), "sync") || unloads != 0 || m.Function("entry") != 10 {
		t.Fatal("sync failure lost ownership", err)
	}
	syncFail, unloadFail = false, true
	if err := m.Close(); err == nil || !strings.Contains(err.Error(), "unload") || unloads != 1 || m.Function("entry") != 10 {
		t.Fatal("unload failure lost ownership", err)
	}
	unloadFail = false
	if err := m.Close(); err != nil || m.Function("entry") != 0 || unloads != 2 {
		t.Fatal("retry failed", err)
	}
	if err := m.Close(); err != nil || unloads != 2 {
		t.Fatal("not idempotent", err)
	}
}

func TestOwnedPTXModuleRejectsInvalid(t *testing.T) {
	for _, tc := range []struct {
		ptx   string
		names []string
	}{{"", []string{"a"}}, {"x", nil}, {"x", []string{""}}, {"x", []string{"a", "a"}}} {
		if m, e := LoadPTXFunctions(tc.ptx, tc.names); e == nil || m != nil {
			t.Fatal("invalid module accepted")
		}
	}
	var m *PTXModule
	if m.Function("a") != 0 {
		t.Fatal("nil handle")
	}
	if e := m.Close(); e != nil {
		t.Fatal(e)
	}
	m = &PTXModule{}
	if m.Function("a") != 0 {
		t.Fatal("empty handle")
	}
	if e := m.Close(); e != nil {
		t.Fatal(e)
	}
}
