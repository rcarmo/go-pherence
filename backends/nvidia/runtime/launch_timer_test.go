//go:build linux

package nvidia

import (
	"math"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

type launchTimerTestState struct {
	gpuOK                bool
	gpuCtx               CUcontext
	captureLaunchStream  CUstream
	cuCtxSetCurrent      func(CUcontext) CUresult
	cuEventCreate        func(*CUevent, uint32) CUresult
	cuEventDestroy       func(CUevent) CUresult
	cuEventRecord        func(CUevent, CUstream) CUresult
	cuEventSynchronize   func(CUevent) CUresult
	cuEventElapsedTime   func(*float32, CUevent, CUevent) CUresult
	cuCtxSynchronize     func() CUresult
	cuLaunchKernel       func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult
	gpuStats             bool
	gpuStatsKernelLaunch uint64
}

func saveLaunchTimerTestState() launchTimerTestState {
	return launchTimerTestState{
		gpuOK:                gpuOK,
		gpuCtx:               gpuCtx,
		captureLaunchStream:  captureLaunchStream,
		cuCtxSetCurrent:      cuCtxSetCurrent,
		cuEventCreate:        cuEventCreate,
		cuEventDestroy:       cuEventDestroy,
		cuEventRecord:        cuEventRecord,
		cuEventSynchronize:   cuEventSynchronize,
		cuEventElapsedTime:   cuEventElapsedTime,
		cuCtxSynchronize:     cuCtxSynchronize,
		cuLaunchKernel:       cuLaunchKernel,
		gpuStats:             gpuStatsEnabled.Load(),
		gpuStatsKernelLaunch: gpuStatsKernelLaunches.Load(),
	}
}

func (s launchTimerTestState) restore() {
	gpuOK = s.gpuOK
	gpuCtx = s.gpuCtx
	captureLaunchStream = s.captureLaunchStream
	cuCtxSetCurrent = s.cuCtxSetCurrent
	cuEventCreate = s.cuEventCreate
	cuEventDestroy = s.cuEventDestroy
	cuEventRecord = s.cuEventRecord
	cuEventSynchronize = s.cuEventSynchronize
	cuEventElapsedTime = s.cuEventElapsedTime
	cuCtxSynchronize = s.cuCtxSynchronize
	cuLaunchKernel = s.cuLaunchKernel
	gpuStatsEnabled.Store(s.gpuStats)
	gpuStatsKernelLaunches.Store(s.gpuStatsKernelLaunch)
}

func primeFakeLaunchTimerGPU(ok bool) {
	gpuOK = ok
	gpuCtx = 1
}

func checkLaunchTimerDriverLocked(t *testing.T, lt *LaunchTimer, tid int) {
	t.Helper()
	if cudaMu.TryLock() {
		cudaMu.Unlock()
		t.Fatal("driver call outside cuda lock")
	}
	if lt != nil && lt.mu.TryLock() {
		lt.mu.Unlock()
		t.Fatal("driver call outside timer lock")
	}
	if tid != 0 && syscall.Gettid() != tid {
		t.Fatal("driver call on wrong thread")
	}
}

func makeLaunchTimerCommand(fn CUfunction, arg0 uint64, arg1 float32) KernelLaunch {
	return KernelLaunch{
		Function: fn,
		Grid:     [3]uint32{1, 1, 1},
		Block:    [3]uint32{32, 1, 1},
		ArgCount: 2,
		Args:     [8]uint64{arg0, uint64(math.Float32bits(arg1))},
	}
}

func TestLaunchTimerNewValidationAndCreateFailure(t *testing.T) {
	state := saveLaunchTimerTestState()
	defer state.restore()

	for _, capacity := range []int{-1, 0, maxLaunchTimerCommands + 1} {
		if lt, err := newFakeLaunchTimer(capacity); err == nil || lt != nil {
			t.Fatalf("capacity %d accepted", capacity)
		}
	}

	primeFakeLaunchTimerGPU(true)
	tid := 0
	creates, destroys := 0, 0
	cuCtxSetCurrent = func(CUcontext) CUresult {
		tid = syscall.Gettid()
		return CUDA_SUCCESS
	}
	cuEventCreate = func(ev *CUevent, flags uint32) CUresult {
		checkLaunchTimerDriverLocked(t, nil, tid)
		if flags != 0 {
			t.Fatalf("timing disabled: flags=%d", flags)
		}
		creates++
		if creates == 3 {
			return 17
		}
		*ev = CUevent(100 + creates)
		return CUDA_SUCCESS
	}
	cuEventDestroy = func(ev CUevent) CUresult {
		checkLaunchTimerDriverLocked(t, nil, tid)
		destroys++
		if ev == 0 {
			t.Fatal("destroyed zero handle")
		}
		return CUDA_SUCCESS
	}
	if lt, err := newFakeLaunchTimer(3); err == nil || lt != nil || creates != 3 || destroys != 2 || !strings.Contains(err.Error(), "create") {
		t.Fatal("event create failure not cleaned up", lt, err, creates, destroys)
	}
}

func TestLaunchTimerMeasureValidationAndCaptureReject(t *testing.T) {
	state := saveLaunchTimerTestState()
	defer state.restore()
	primeFakeLaunchTimerGPU(true)
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	nextEvent := CUevent(10)
	cuEventCreate = func(ev *CUevent, flags uint32) CUresult {
		if flags != 0 {
			t.Fatalf("flags=%d", flags)
		}
		*ev = nextEvent
		nextEvent++
		return CUDA_SUCCESS
	}
	cuEventDestroy = func(CUevent) CUresult { return CUDA_SUCCESS }
	lt, err := newFakeLaunchTimer(2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
		if err := lt.Close(); err != nil {
			t.Fatal(err)
		}
	}()
	records, launches := 0, 0
	cuEventRecord = func(CUevent, CUstream) CUresult { records++; return CUDA_SUCCESS }
	cuEventSynchronize = func(CUevent) CUresult { return CUDA_SUCCESS }
	cuEventElapsedTime = func(*float32, CUevent, CUevent) CUresult { return CUDA_SUCCESS }
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		launches++
		return CUDA_SUCCESS
	}

	cmds := []KernelLaunch{makeLaunchTimerCommand(1, 7, 0.25), makeLaunchTimerCommand(2, 8, 0.5)}
	invalid := append([]KernelLaunch(nil), cmds...)
	invalid[1].Grid[0] = 0
	if ms, err := lt.Measure(invalid); err == nil || ms != nil || records != 0 || launches != 0 {
		t.Fatal("geometry validation launched work", ms, err, records, launches)
	}
	tooMany := append(cmds, makeLaunchTimerCommand(3, 9, 0.75))
	if ms, err := lt.Measure(tooMany); err == nil || ms != nil || !strings.Contains(err.Error(), "capacity") {
		t.Fatal("capacity overflow accepted", ms, err)
	}
	captureLaunchStream = 7
	if ms, err := lt.Measure(cmds); err == nil || ms != nil || !strings.Contains(err.Error(), "capture") || records != 0 || launches != 0 {
		t.Fatal("capture not rejected", ms, err, records, launches)
	}
}

func TestLaunchTimerMeasureOrderLocksAndTimings(t *testing.T) {
	state := saveLaunchTimerTestState()
	defer state.restore()
	primeFakeLaunchTimerGPU(true)
	tid := 0
	cuCtxSetCurrent = func(CUcontext) CUresult {
		tid = syscall.Gettid()
		return CUDA_SUCCESS
	}
	nextEvent := CUevent(200)
	cuEventCreate = func(ev *CUevent, flags uint32) CUresult {
		if flags != 0 {
			t.Fatalf("flags=%d", flags)
		}
		*ev = nextEvent
		nextEvent++
		return CUDA_SUCCESS
	}
	cuEventDestroy = func(CUevent) CUresult { return CUDA_SUCCESS }
	lt, err := newFakeLaunchTimer(2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
		if err := lt.Close(); err != nil {
			t.Fatal(err)
		}
	}()
	trace := []string{}
	syncs := 0
	cuCtxSynchronize = func() CUresult {
		syncs++
		checkLaunchTimerDriverLocked(t, lt, tid)
		return CUDA_SUCCESS
	}
	cuEventRecord = func(ev CUevent, stream CUstream) CUresult {
		checkLaunchTimerDriverLocked(t, lt, tid)
		trace = append(trace, "record:"+strconv.Itoa(int(ev))+":"+strconv.Itoa(int(stream)))
		return CUDA_SUCCESS
	}
	launches := 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, sm uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		checkLaunchTimerDriverLocked(t, lt, tid)
		if stream != 0 {
			t.Fatalf("stream=%d", stream)
		}
		ptrs := unsafe.Slice((*unsafe.Pointer)(args), 2)
		if *(*uint64)(ptrs[0]) != uint64(1234+launches) {
			t.Fatalf("arg0=%d", *(*uint64)(ptrs[0]))
		}
		want1 := float32(0.125 + float32(launches))
		if got := math.Float32frombits(*(*uint32)(ptrs[1])); got != want1 {
			t.Fatalf("arg1=%g want=%g", got, want1)
		}
		trace = append(trace, "launch:"+strconv.Itoa(launches)+":"+strconv.Itoa(int(fn)))
		launches++
		return CUDA_SUCCESS
	}
	cuEventSynchronize = func(ev CUevent) CUresult {
		checkLaunchTimerDriverLocked(t, lt, tid)
		trace = append(trace, "sync:"+strconv.Itoa(int(ev)))
		return CUDA_SUCCESS
	}
	cuEventElapsedTime = func(ms *float32, start, end CUevent) CUresult {
		checkLaunchTimerDriverLocked(t, lt, tid)
		trace = append(trace, "elapsed:"+strconv.Itoa(int(start))+":"+strconv.Itoa(int(end)))
		switch {
		case start == 200 && end == 201:
			*ms = 1.25
		case start == 201 && end == 202:
			*ms = 2.5
		default:
			t.Fatalf("unexpected pair %d %d", start, end)
		}
		return CUDA_SUCCESS
	}
	gpuStatsEnabled.Store(true)
	gpuStatsKernelLaunches.Store(0)
	cmds := []KernelLaunch{makeLaunchTimerCommand(11, 1234, 0.125), makeLaunchTimerCommand(12, 1235, 1.125)}
	ms, err := lt.Measure(cmds)
	if err != nil {
		t.Fatal(err)
	}
	if len(ms) != 2 || ms[0] != 1.25 || ms[1] != 2.5 {
		t.Fatal("wrong timings", ms)
	}
	want := []string{
		"record:200:0",
		"launch:0:11",
		"record:201:0",
		"launch:1:12",
		"record:202:0",
		"sync:202",
		"elapsed:200:201",
		"elapsed:201:202",
	}
	if len(trace) != len(want) {
		t.Fatal(trace)
	}
	for i := range want {
		if trace[i] != want[i] {
			t.Fatalf("trace[%d]=%q want %q", i, trace[i], want[i])
		}
	}
	if syncs != 0 {
		t.Fatal("unexpected drain", syncs)
	}
	if got := gpuStatsKernelLaunches.Load(); got != 2 {
		t.Fatal("stats", got)
	}
}

func TestLaunchTimerMeasureErrorsDrainAndRejectBadElapsed(t *testing.T) {
	for _, tc := range []struct {
		name         string
		recordFailAt int
		launchFailAt int
		syncFail     bool
		drainFail    bool
		elapsedFail  bool
		elapsedValue float32
		wantDrains   int
	}{
		{name: "start-record", recordFailAt: 1, wantDrains: 1},
		{name: "first-launch", launchFailAt: 1, wantDrains: 1},
		{name: "drain-failure", launchFailAt: 1, drainFail: true, wantDrains: 1},
		{name: "mid-record", recordFailAt: 2, wantDrains: 1},
		{name: "mid-launch", launchFailAt: 2, wantDrains: 1},
		{name: "sync", syncFail: true, wantDrains: 1},
		{name: "elapsed", elapsedFail: true, wantDrains: 0},
		{name: "bad-elapsed", elapsedValue: float32(math.NaN()), wantDrains: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			state := saveLaunchTimerTestState()
			defer state.restore()
			primeFakeLaunchTimerGPU(true)
			cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
			nextEvent := CUevent(300)
			cuEventCreate = func(ev *CUevent, flags uint32) CUresult {
				*ev = nextEvent
				nextEvent++
				return CUDA_SUCCESS
			}
			cuEventDestroy = func(CUevent) CUresult { return CUDA_SUCCESS }
			lt, err := newFakeLaunchTimer(2)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
				if err := lt.Close(); err != nil {
					t.Fatal(err)
				}
			}()
			records, launches, drains, elapsed := 0, 0, 0, 0
			cuEventRecord = func(CUevent, CUstream) CUresult {
				records++
				if tc.recordFailAt != 0 && records == tc.recordFailAt {
					return 21
				}
				return CUDA_SUCCESS
			}
			cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
				launches++
				if tc.launchFailAt != 0 && launches == tc.launchFailAt {
					return 22
				}
				return CUDA_SUCCESS
			}
			cuEventSynchronize = func(CUevent) CUresult {
				if tc.syncFail {
					return 23
				}
				return CUDA_SUCCESS
			}
			cuCtxSynchronize = func() CUresult {
				drains++
				if tc.drainFail {
					return 25
				}
				return CUDA_SUCCESS
			}
			cuEventElapsedTime = func(ms *float32, start, end CUevent) CUresult {
				elapsed++
				if tc.elapsedFail {
					return 24
				}
				if tc.elapsedValue != 0 || math.IsNaN(float64(tc.elapsedValue)) {
					*ms = tc.elapsedValue
				} else {
					*ms = 1
				}
				return CUDA_SUCCESS
			}
			cmds := []KernelLaunch{makeLaunchTimerCommand(1, 1, 0.25), makeLaunchTimerCommand(2, 2, 0.5)}
			ms, err := lt.Measure(cmds)
			if err == nil || ms != nil {
				t.Fatal("error path returned timings", ms, err)
			}
			if tc.drainFail && lt.usable {
				t.Fatal("unsafe reuse after drain failure")
			}
			if drains != tc.wantDrains {
				t.Fatal("drains", tc.name, drains, tc.wantDrains)
			}
			if tc.elapsedFail && elapsed != 1 {
				t.Fatal("elapsed not attempted", elapsed)
			}
		})
	}
}

func TestLaunchTimerCloseDestroyRetry(t *testing.T) {
	state := saveLaunchTimerTestState()
	defer state.restore()
	primeFakeLaunchTimerGPU(true)
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	nextEvent := CUevent(400)
	cuEventCreate = func(ev *CUevent, flags uint32) CUresult {
		*ev = nextEvent
		nextEvent++
		return CUDA_SUCCESS
	}
	failDestroy := CUevent(401)
	destroys := []CUevent{}
	cuEventDestroy = func(ev CUevent) CUresult {
		destroys = append(destroys, ev)
		if ev == failDestroy {
			return 31
		}
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = func() CUresult {
		if cudaMu.TryLock() {
			cudaMu.Unlock()
			t.Fatal("sync outside lock")
		}
		return CUDA_SUCCESS
	}
	lt, err := newFakeLaunchTimer(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := lt.Close(); err == nil || !strings.Contains(err.Error(), "destroy") {
		t.Fatal("destroy failure lost", err)
	}
	if lt.usable {
		t.Fatal("close left timer usable")
	}
	if lt.events[0] != 0 || lt.events[1] != failDestroy || lt.events[2] != 0 {
		t.Fatal("wrong retained handles", lt.events)
	}
	failDestroy = 0
	if err := lt.Close(); err != nil {
		t.Fatal(err)
	}
	if lt.events[0] != 0 || lt.events[1] != 0 || lt.events[2] != 0 {
		t.Fatal("retry did not clear handles", lt.events)
	}
	before := len(destroys)
	if err := lt.Close(); err != nil {
		t.Fatal(err)
	}
	if len(destroys) != before {
		t.Fatal("idempotent close retried destroy", destroys)
	}
	var nilTimer *LaunchTimer
	if err := nilTimer.Close(); err != nil {
		t.Fatal(err)
	}
}

// Test allocation with fake driver state without copying/resetting gpuOnce.
func newFakeLaunchTimer(capacity int) (*LaunchTimer, error) {
	release := lockDriver()
	defer release()
	return newLaunchTimerLocked(capacity)
}

func TestLaunchTimerRemainingFailureContracts(t *testing.T) {
	saved := saveLaunchTimerTestState()
	defer saved.restore()
	primeFakeLaunchTimerGPU(true)
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	for _, cap := range []int{0, 513} {
		if owner, e := NewLaunchTimer(cap); e == nil || owner != nil {
			t.Fatal("public invalid cap")
		}
	}
	var nilTimer *LaunchTimer
	if out, e := nilTimer.Measure(nil); e == nil || out != nil {
		t.Fatal("nil measure")
	}
	if e := (*LaunchTimer)(nil).destroyEventsLocked(false); e != nil {
		t.Fatal(e)
	}
	if e := validateLaunchTimerCommands(make([]KernelLaunch, 513), 514); e == nil {
		t.Fatal("absolute command limit")
	}
	cuEventCreate = nil
	if owner, e := newFakeLaunchTimer(1); e == nil || owner != nil {
		t.Fatal("missing create")
	}
	count := 0
	cuEventCreate = func(p *CUevent, _ uint32) CUresult {
		count++
		if count == 2 {
			return 3
		}
		*p = 7
		return CUDA_SUCCESS
	}
	cuEventDestroy = func(CUevent) CUresult { return 4 }
	owner, e := newFakeLaunchTimer(1)
	if e == nil || owner == nil || owner.usable || owner.events[0] != 7 {
		t.Fatal("constructor cleanup ownership", owner, e)
	}
	cuCtxSynchronize = nil
	if e = owner.Close(); e == nil || owner.events[0] != 7 {
		t.Fatal("missing sync lost owner")
	}
	cuCtxSynchronize = func() CUresult { return 5 }
	if e = owner.Close(); e == nil || owner.events[0] != 7 {
		t.Fatal("sync failure lost owner")
	}
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	cuEventDestroy = nil
	if e = owner.Close(); e == nil || owner.events[0] != 7 {
		t.Fatal("missing destroy lost owner")
	}
	cuEventDestroy = func(CUevent) CUresult { return CUDA_SUCCESS }
	if e = owner.Close(); e != nil {
		t.Fatal(e)
	}
	if out, e := owner.Measure(nil); e == nil || out != nil {
		t.Fatal("closed usable")
	}
	next := CUevent(20)
	cuEventCreate = func(p *CUevent, _ uint32) CUresult { *p = next; next++; return CUDA_SUCCESS }
	owner, e = newFakeLaunchTimer(1)
	if e != nil {
		t.Fatal(e)
	}
	cuEventDestroy = nil
	empty := &LaunchTimer{}
	if e = empty.destroyEventsLocked(false); e != nil {
		t.Fatal(e)
	}
	cuEventDestroy = func(CUevent) CUresult { return CUDA_SUCCESS }
	defer owner.Close()
	if out, e := owner.Measure(nil); e != nil || len(out) != 0 {
		t.Fatal("empty batch", e)
	}
	cmds := []KernelLaunch{makeLaunchTimerCommand(1, 2, 0.25)}
	gpuOK = false
	if out, e := owner.Measure(cmds); e == nil || out != nil {
		t.Fatal("unavailable GPU")
	}
	if e = owner.Close(); e == nil {
		t.Fatal("unavailable close")
	}
	gpuOK = true
	owner.usable = true // Re-arm only this fake owner to exercise later failures.
	cuEventRecord = nil
	if out, e := owner.Measure(cmds); e == nil || out != nil {
		t.Fatal("missing event API")
	}
	cuEventRecord = func(CUevent, CUstream) CUresult { return 8 }
	cuEventSynchronize = func(CUevent) CUresult { return CUDA_SUCCESS }
	cuEventElapsedTime = func(*float32, CUevent, CUevent) CUresult { return CUDA_SUCCESS }
	cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
		return CUDA_SUCCESS
	}
	cuCtxSynchronize = nil
	if out, e := owner.Measure(cmds); e == nil || out != nil || owner.usable {
		t.Fatal("missing drain did not poison")
	}
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
}
