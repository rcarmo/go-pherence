package nemotrondiarization

import ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"

// Narrow seams for model-free driver-failure tests. Production calls always use
// the checked runtime functions; tests restore these globals and never run in
// parallel with other PTX tests.
var ptxCleanupSync = ptx.SyncErr
var ptxCleanupFree = (*ptx.Buffer).FreeChecked
var ptxCleanupMalloc = ptx.Malloc
