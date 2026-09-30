//go:build linux

package nvidia

import (
	"strings"
	"testing"
)

// No Init or device call: test CUDA's error and ownership boundary with a
// fake driver. The GPU can be unavailable while this regression runs.
func TestSyncErrDoesNotCertifyFailedContextSelection(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync := cuCtxSetCurrent, cuCtxSynchronize
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize = oldSet, oldSync
	}()
	gpuOK, gpuCtx = true, 123
	calls := 0
	cuCtxSetCurrent = func(CUcontext) CUresult { return 201 }
	cuCtxSynchronize = func() CUresult { calls++; return CUDA_SUCCESS }
	if err := SyncErr(); err == nil || calls != 0 {
		t.Fatalf("sync certified wrong context: calls=%d err=%v", calls, err)
	}
}

func TestBufferFreeCheckedRetainsOwnerOnFailure(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldFree := cuCtxSetCurrent, cuMemFree
	oldStats := SetStatsEnabled(true)
	before := StatsSnapshot()
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuMemFree = oldSet, oldFree
		SetStatsEnabled(oldStats)
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(ctx CUcontext) CUresult {
		if ctx != 123 {
			t.Errorf("wrong CUDA context: %d", ctx)
		}
		return CUDA_SUCCESS
	}
	calls := 0
	cuMemFree = func(p CUdeviceptr) CUresult {
		calls++
		if p != 77 {
			t.Errorf("wrong CUDA pointer: %d", p)
		}
		if cudaMu.TryLock() {
			cudaMu.Unlock()
			t.Error("free ran outside driver lock")
		}
		if calls == 1 {
			return 201
		}
		return CUDA_SUCCESS
	}
	b := &Buffer{Ptr: 77, Size: 4096}
	if err := b.FreeChecked(); err == nil || !strings.Contains(err.Error(), "cuMemFree: error 201") || b.Ptr != 77 || calls != 1 {
		t.Fatalf("failed free lost ownership: buffer=%+v calls=%d err=%v", b, calls, err)
	}
	if after := StatsSnapshot(); after.Frees != before.Frees || after.FreeBytes != before.FreeBytes {
		t.Fatalf("failed free incremented stats: before=%+v after=%+v", before, after)
	}
	// The caller may retry only after separately establishing context health.
	if err := b.FreeChecked(); err != nil || b.Ptr != 0 || calls != 2 {
		t.Fatalf("successful free not recorded: buffer=%+v calls=%d err=%v", b, calls, err)
	}
	if after := StatsSnapshot(); after.Frees != before.Frees+1 || after.FreeBytes != before.FreeBytes+4096 {
		t.Fatalf("successful free not counted: before=%+v after=%+v", before, after)
	}
	if err := b.FreeChecked(); err != nil || calls != 2 {
		t.Fatalf("repeated free called driver: calls=%d err=%v", calls, err)
	}
	b.Ptr = 88
	cuCtxSetCurrent = func(CUcontext) CUresult { return 201 }
	if err := b.FreeChecked(); err == nil || !strings.Contains(err.Error(), "cuCtxSetCurrent") || b.Ptr != 88 || calls != 2 {
		t.Fatalf("failed context selection called free or lost owner: buffer=%+v calls=%d err=%v", b, calls, err)
	}
}
