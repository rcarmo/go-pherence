package nvidia

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"
)

const maxLaunchTimerCommands = 512

// LaunchTimer owns a fixed CUDA event ring for bounded diagnostic timing on the
// default stream. It is not safe to copy. Close owners before global Shutdown.
// Events perturb short launches; results are diagnostic GPU timeline intervals,
// not a replacement for uninstrumented whole-request latency measurements.
type LaunchTimer struct {
	mu       sync.Mutex
	capacity int
	events   []CUevent
	usable   bool
}

// NewLaunchTimer allocates one timing-enabled CUDA event per launch boundary.
// Capacity is diagnostic-only and bounded to keep event ownership simple.
func NewLaunchTimer(capacity int) (*LaunchTimer, error) {
	if err := validateLaunchTimerCapacity(capacity); err != nil {
		return nil, err
	}
	if !Init() {
		return nil, fmt.Errorf("CUDA device unavailable")
	}
	release := lockDriver()
	defer release()
	return newLaunchTimerLocked(capacity)
}

// newLaunchTimerLocked requires lockDriver and a successfully initialised CUDA
// runtime. On partial cleanup failure it returns the owner plus error so Close
// can be retried.
func newLaunchTimerLocked(capacity int) (*LaunchTimer, error) {
	if err := validateLaunchTimerCapacity(capacity); err != nil {
		return nil, err
	}
	if cuEventCreate == nil || cuEventDestroy == nil {
		return nil, fmt.Errorf("CUDA event timing unavailable")
	}
	lt := &LaunchTimer{capacity: capacity, events: make([]CUevent, capacity+1)}
	for i := range lt.events {
		if r := cuEventCreate(&lt.events[i], 0); r != CUDA_SUCCESS {
			createErr := fmt.Errorf("launch timer event %d create: %d", i, r)
			if cleanupErr := lt.destroyEventsLocked(false); cleanupErr != nil {
				return lt, errors.Join(createErr, cleanupErr)
			}
			return nil, createErr
		}
	}
	lt.usable = true
	return lt, nil
}

func validateLaunchTimerCapacity(capacity int) error {
	if capacity < 1 || capacity > maxLaunchTimerCommands {
		return fmt.Errorf("launch timer capacity must be 1..%d", maxLaunchTimerCommands)
	}
	return nil
}

func validateLaunchTimerCommands(commands []KernelLaunch, capacity int) error {
	if len(commands) > capacity {
		return fmt.Errorf("launch timer capacity %d exceeded by %d commands", capacity, len(commands))
	}
	if len(commands) > maxLaunchTimerCommands {
		return fmt.Errorf("launch timer supports at most %d commands", maxLaunchTimerCommands)
	}
	for i := range commands {
		c := &commands[i]
		if c.Function == 0 || c.ArgCount < 0 || c.ArgCount > len(c.Args) || c.Grid[0] == 0 || c.Grid[1] == 0 || c.Grid[2] == 0 || c.Block[0] == 0 || c.Block[1] == 0 || c.Block[2] == 0 {
			return fmt.Errorf("invalid CUDA launch timer command %d", i)
		}
		for j := 0; j < c.ArgCount; j++ {
			c.pointers[j] = unsafe.Pointer(&c.Args[j])
		}
	}
	return nil
}

// Measure launches the commands on the default stream and returns one elapsed
// time in milliseconds per command.
func (lt *LaunchTimer) Measure(commands []KernelLaunch) ([]float32, error) {
	if lt == nil {
		return nil, fmt.Errorf("nil launch timer")
	}
	lt.mu.Lock()
	defer lt.mu.Unlock()
	if err := validateLaunchTimerCommands(commands, lt.capacity); err != nil {
		return nil, err
	}
	if !lt.usable || len(lt.events) != lt.capacity+1 {
		return nil, fmt.Errorf("launch timer closed")
	}
	if len(commands) == 0 {
		return []float32{}, nil
	}
	if !gpuOK {
		return nil, fmt.Errorf("CUDA device unavailable")
	}
	if cuLaunchKernel == nil || cuEventRecord == nil || cuEventSynchronize == nil || cuEventElapsedTime == nil {
		return nil, fmt.Errorf("CUDA event timing unavailable")
	}
	defer runtime.KeepAlive(commands)
	release := lockDriver()
	defer release()
	if captureLaunchStream != 0 {
		return nil, fmt.Errorf("CUDA graph capture active")
	}
	launched := false
	synced := false
	fail := func(err error) ([]float32, error) {
		if launched && !synced {
			if cuCtxSynchronize == nil {
				lt.usable = false
				return nil, errors.Join(err, fmt.Errorf("launch timer drain unavailable"))
			}
			if r := cuCtxSynchronize(); r != CUDA_SUCCESS {
				lt.usable = false
				return nil, errors.Join(err, fmt.Errorf("launch timer drain: %d", r))
			}
		}
		return nil, err
	}
	launched = true // Even an event record may enqueue work before surfacing an error.
	if r := cuEventRecord(lt.events[0], 0); r != CUDA_SUCCESS {
		return fail(fmt.Errorf("launch timer start record: %d", r))
	}
	stats := gpuStatsEnabled.Load()
	for i := range commands {
		c := &commands[i]
		var ptr unsafe.Pointer
		if c.ArgCount > 0 {
			ptr = unsafe.Pointer(&c.pointers[0])
		}
		if r := cuLaunchKernel(c.Function, c.Grid[0], c.Grid[1], c.Grid[2], c.Block[0], c.Block[1], c.Block[2], c.SharedMem, 0, ptr, nil); r != CUDA_SUCCESS {
			return fail(fmt.Errorf("launch timer command %d launch: %d", i, r))
		}
		launched = true
		if stats {
			gpuStatsKernelLaunches.Add(1)
		}
		if r := cuEventRecord(lt.events[i+1], 0); r != CUDA_SUCCESS {
			return fail(fmt.Errorf("launch timer boundary %d record: %d", i+1, r))
		}
	}
	if r := cuEventSynchronize(lt.events[len(commands)]); r != CUDA_SUCCESS {
		return fail(fmt.Errorf("launch timer sync: %d", r))
	}
	synced = true
	ms := make([]float32, len(commands))
	for i := range ms {
		if r := cuEventElapsedTime(&ms[i], lt.events[i], lt.events[i+1]); r != CUDA_SUCCESS {
			return nil, fmt.Errorf("launch timer elapsed %d: %d", i, r)
		}
		v := float64(ms[i])
		if math.IsNaN(v) || math.IsInf(v, 0) || ms[i] < 0 {
			return nil, fmt.Errorf("launch timer elapsed %d invalid: %g", i, ms[i])
		}
	}
	return ms, nil
}

// Close synchronizes outstanding work, then destroys any remaining events.
// Destroy failures retain the surviving handles for a later retry.
func (lt *LaunchTimer) Close() error {
	if lt == nil {
		return nil
	}
	lt.mu.Lock()
	defer lt.mu.Unlock()
	lt.usable = false
	live := false
	for _, ev := range lt.events {
		if ev != 0 {
			live = true
			break
		}
	}
	if !live {
		return nil
	}
	if !gpuOK {
		return fmt.Errorf("CUDA device unavailable")
	}
	release := lockDriver()
	defer release()
	return lt.destroyEventsLocked(true)
}

func (lt *LaunchTimer) destroyEventsLocked(syncBefore bool) error {
	if lt == nil {
		return nil
	}
	if syncBefore {
		if cuCtxSynchronize == nil {
			return fmt.Errorf("launch timer sync unavailable")
		}
		if r := cuCtxSynchronize(); r != CUDA_SUCCESS {
			return fmt.Errorf("launch timer sync: %d", r)
		}
	}
	if cuEventDestroy == nil {
		for _, ev := range lt.events {
			if ev != 0 {
				return fmt.Errorf("cuEventDestroy unavailable")
			}
		}
		return nil
	}
	var err error
	for i, ev := range lt.events {
		if ev == 0 {
			continue
		}
		if r := cuEventDestroy(ev); r != CUDA_SUCCESS {
			err = errors.Join(err, fmt.Errorf("launch timer event %d destroy: %d", i, r))
			continue
		}
		lt.events[i] = 0
	}
	return err
}
