package nemotrondiarization

import (
	"context"
	"errors"
	"sync"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// No CUDA Init or allocation: these tests substitute the cleanup boundary.
// Do not run these global-seam tests in parallel with any other PTX test.
func fakePTXCleanup(t *testing.T, sync func() error, free func(*ptx.Buffer) error) {
	t.Helper()
	oldSync, oldFree := ptxCleanupSync, ptxCleanupFree
	ptxCleanupSync, ptxCleanupFree = sync, free
	t.Cleanup(func() { ptxCleanupSync, ptxCleanupFree = oldSync, oldFree })
}

func TestPTXTowerFailedSyncRetainsOwnersAndRejectsWork(t *testing.T) {
	failure := errors.New("sync unavailable")
	frees, syncs := 0, 0
	fakePTXCleanup(t, func() error { syncs++; return failure }, func(*ptx.Buffer) error { frees++; return nil })
	b := &ptx.Buffer{Ptr: 31, Size: 16}
	tower := &PTXAudioTower{weight: b}
	if err := tower.Close(); !errors.Is(err, failure) || tower.weight != b || frees != 0 {
		t.Fatalf("failed close lost owner: weight=%p frees=%d err=%v", tower.weight, frees, err)
	}
	if _, err := tower.ForwardRows(context.Background(), nil, 1); !errors.Is(err, failure) {
		t.Fatalf("tower accepted new work: %v", err)
	}
	if err := tower.Close(); !errors.Is(err, failure) || syncs != 1 || frees != 0 {
		t.Fatalf("repeated close retried unsafe cleanup: syncs=%d frees=%d err=%v", syncs, frees, err)
	}
}

func TestPTXTowerPartialFreeRetainsOnlyFailedOwners(t *testing.T) {
	failure := errors.New("free unavailable")
	var freed []ptx.CUdeviceptr
	fakePTXCleanup(t, func() error { return nil }, func(b *ptx.Buffer) error {
		freed = append(freed, b.Ptr)
		if b.Ptr == 2 {
			return failure
		}
		b.Ptr = 0 // fake driver's successful free
		return nil
	})
	tower := &PTXAudioTower{scratch: &ptxLayerScratch{buffers: [9]*ptx.Buffer{
		0: {Ptr: 1}, 8: {Ptr: 2},
	}}, weight: &ptx.Buffer{Ptr: 3}}
	if err := tower.Close(); !errors.Is(err, failure) || tower.scratch.buffers[8].Ptr != 2 || tower.scratch.buffers[0] == nil || tower.weight.Ptr != 3 {
		t.Fatalf("partial free discarded failed/unvisited owners: %+v err=%v", tower, err)
	}
	if err := tower.Close(); !errors.Is(err, failure) || len(freed) != 1 || freed[0] != 2 {
		t.Fatalf("partial free retried: freed=%v err=%v", freed, err)
	}
}

func TestPTXTowerFailedWeightFreeRetainsRemainingLayers(t *testing.T) {
	failure := errors.New("weight free unavailable")
	calls := 0
	fakePTXCleanup(t, func() error { return nil }, func(b *ptx.Buffer) error {
		calls++
		if b.Ptr == 7 {
			return failure
		}
		b.Ptr = 0
		return nil
	})
	layer := &PTXAudioLayer{owned: []*ptx.Buffer{{Ptr: 9}}}
	tower := &PTXAudioTower{bias: &ptx.Buffer{Ptr: 6}, weight: &ptx.Buffer{Ptr: 7}, layers: []*PTXAudioLayer{layer}}
	if err := tower.Close(); !errors.Is(err, failure) || tower.bias != nil || tower.weight.Ptr != 7 || tower.layers[0] != layer || layer.owned[0].Ptr != 9 || calls != 2 {
		t.Fatalf("partial weight free lost ownership or freed downstream: %+v calls=%d err=%v", tower, calls, err)
	}
	if err := tower.Close(); !errors.Is(err, failure) || calls != 2 {
		t.Fatalf("terminal tower retried free: calls=%d err=%v", calls, err)
	}
}

func TestPTXTowerCloseSuccessAndRepeatedClose(t *testing.T) {
	var freed []ptx.CUdeviceptr
	syncs := 0
	fakePTXCleanup(t, func() error { syncs++; return nil }, func(b *ptx.Buffer) error {
		freed = append(freed, b.Ptr)
		b.Ptr = 0
		return nil
	})
	tower := &PTXAudioTower{weight: &ptx.Buffer{Ptr: 1}, bias: &ptx.Buffer{Ptr: 2}, scratch: &ptxLayerScratch{buffers: [9]*ptx.Buffer{0: {Ptr: 3}}}, layers: []*PTXAudioLayer{{owned: []*ptx.Buffer{{Ptr: 4}}}}}
	if err := tower.Close(); err != nil || !tower.closed || tower.weight != nil || tower.bias != nil || tower.scratch != nil || len(tower.layers) != 0 || len(freed) != 4 {
		t.Fatalf("successful close retained owners: tower=%+v freed=%v err=%v", tower, freed, err)
	}
	if err := tower.Close(); err != nil || len(freed) != 4 || syncs != 2 { // tower and layer each sync once
		t.Fatalf("repeated close touched driver: syncs=%d freed=%v err=%v", syncs, freed, err)
	}
}

func TestPTXForwardTransientCleanupFailure(t *testing.T) {
	for _, scenario := range []struct {
		name string
		sync bool
	}{{"sync", true}, {"free", false}} {
		t.Run(scenario.name, func(t *testing.T) {
			failure := errors.New("driver failure")
			var freed []ptx.CUdeviceptr
			fakePTXCleanup(t, func() error {
				if scenario.sync {
					return failure
				}
				return nil
			}, func(b *ptx.Buffer) error {
				freed = append(freed, b.Ptr)
				if b.Ptr == 5 {
					return failure
				}
				b.Ptr = 0
				return nil
			})
			tower := &PTXAudioTower{}
			owners := []*ptx.Buffer{{Ptr: 4}, {Ptr: 5}, {Ptr: 6}}
			if err := tower.releaseForwardOwners(owners); !errors.Is(err, failure) || len(tower.pending) != 3 {
				t.Fatalf("lost transient owners: pending=%v err=%v", tower.pending, err)
			}
			if scenario.sync && len(freed) != 0 || !scenario.sync && (len(freed) != 2 || owners[2].Ptr != 0 || owners[1].Ptr != 5 || owners[0].Ptr != 4) {
				t.Fatalf("incorrect release order/sync barrier: freed=%v owners=%v", freed, owners)
			}
			if _, err := tower.ForwardRows(context.Background(), nil, 1); !errors.Is(err, failure) {
				t.Fatalf("forward did not reject terminal owner: %v", err)
			}
		})
	}
}

func TestPTXForwardPriorFailedSyncCannotBeClearedByLaterSync(t *testing.T) {
	failure := errors.New("first sync failed")
	syncs, frees := 0, 0
	fakePTXCleanup(t, func() error {
		syncs++
		if syncs == 1 {
			return failure
		}
		return nil
	}, func(*ptx.Buffer) error { frees++; return nil })
	tower := &PTXAudioTower{terminal: failure}
	owner := &ptx.Buffer{Ptr: 20}
	if err := tower.releaseForwardOwners([]*ptx.Buffer{owner}); !errors.Is(err, failure) || syncs != 0 || frees != 0 || len(tower.pending) != 1 || owner.Ptr != 20 {
		t.Fatalf("later sync erased earlier failure: syncs=%d frees=%d pending=%v err=%v", syncs, frees, tower.pending, err)
	}
}

func TestPTXScratchConstructorRetainsFailedCleanupOwner(t *testing.T) {
	failure := errors.New("allocation unavailable")
	freeFailure := errors.New("free unavailable")
	oldMalloc := ptxCleanupMalloc
	allocs, frees := 0, 0
	ptxCleanupMalloc = func(int) (*ptx.Buffer, error) {
		allocs++
		if allocs == 3 {
			return nil, failure
		}
		return &ptx.Buffer{Ptr: ptx.CUdeviceptr(allocs), Size: 1 << 20}, nil
	}
	fakePTXCleanup(t, func() error { return nil }, func(b *ptx.Buffer) error {
		frees++
		if b.Ptr == 2 {
			return freeFailure
		}
		b.Ptr = 0
		return nil
	})
	t.Cleanup(func() { ptxCleanupMalloc = oldMalloc })
	scratch, err := newPTXLayerScratch(1)
	if !errors.Is(err, failure) || !errors.Is(err, freeFailure) || scratch == nil || scratch.buffers[1].Ptr != 2 || scratch.buffers[0].Ptr != 1 || frees != 1 {
		t.Fatalf("failed construction discarded owner: scratch=%+v allocs=%d frees=%d err=%v", scratch, allocs, frees, err)
	}
	if again := scratch.close(); !errors.Is(again, freeFailure) || frees != 1 {
		t.Fatalf("constructor cleanup was blindly retried: frees=%d err=%v", frees, again)
	}
}

func TestPTXScratchConstructorFailedSyncSkipsFree(t *testing.T) {
	failure := errors.New("completion unknown")
	allocs, frees := 0, 0
	oldMalloc := ptxCleanupMalloc
	ptxCleanupMalloc = func(int) (*ptx.Buffer, error) {
		allocs++
		if allocs == 2 {
			return nil, errors.New("allocation unavailable")
		}
		return &ptx.Buffer{Ptr: 1}, nil
	}
	t.Cleanup(func() { ptxCleanupMalloc = oldMalloc })
	fakePTXCleanup(t, func() error { return failure }, func(*ptx.Buffer) error { frees++; return nil })
	scratch, err := newPTXLayerScratch(1)
	if !errors.Is(err, failure) || scratch == nil || scratch.buffers[0] == nil || frees != 0 {
		t.Fatalf("constructor freed after failed sync: scratch=%+v frees=%d err=%v", scratch, frees, err)
	}
	if again := scratch.close(); !errors.Is(again, failure) || frees != 0 {
		t.Fatalf("terminal scratch retried: frees=%d err=%v", frees, again)
	}
}

func TestPTXLayerCloseSerializesAgainstForwardSync(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	freeCalls := 0
	fakePTXCleanup(t, func() error {
		once.Do(func() { close(entered); <-release })
		return errors.New("completion unknown")
	}, func(*ptx.Buffer) error { freeCalls++; return nil })
	layer := &PTXAudioLayer{owned: []*ptx.Buffer{{Ptr: 11}}}
	layer.callMu.Lock() // model an in-flight ForwardBuffer including its deferred sync
	done := make(chan error, 1)
	go func() { done <- layer.Close() }()
	select {
	case err := <-done:
		t.Fatalf("close overtook forward: %v", err)
	default:
	}
	layer.callMu.Unlock()
	<-entered
	close(release)
	if err := <-done; err == nil || freeCalls != 0 || layer.owned[0].Ptr != 11 {
		t.Fatalf("failed sync allowed free: freeCalls=%d err=%v", freeCalls, err)
	}
}

func TestPTXRequestClosePreservesFailedTower(t *testing.T) {
	failure := errors.New("completion unknown")
	stream := &PTXStreamingTower{tower: &PTXAudioTower{terminal: failure}}
	request := &PCMStreamingRequest{window: &StreamingWindow{PTXTower: stream}}
	if err := request.ClosePTXTower(); !errors.Is(err, failure) || request.window.PTXTower != stream {
		t.Fatalf("request discarded failed tower: stream=%p err=%v", request.window.PTXTower, err)
	}
	if err := request.ClosePTXTower(); !errors.Is(err, failure) || request.window.PTXTower != stream {
		t.Fatalf("request retried or lost terminal owner: stream=%p err=%v", request.window.PTXTower, err)
	}
}

// Inert Vulkan owners and injected close callbacks: no VulkanInit, loader or
// device call. Failed owners remain reachable and repeated Close reports the
// same error without retrying either callback.
func TestProjectorVulkanCloseFailureRetainsOwners(t *testing.T) {
	oldArena, oldOp := projectorVKArenaClose, projectorVKOpClose
	t.Cleanup(func() { projectorVKArenaClose, projectorVKOpClose = oldArena, oldOp })
	arena := new(vulkan.VkTensorArena)
	op := new(vulkan.VkLinearF32)
	arenaFailure, opFailure := errors.New("arena still in flight"), errors.New("operator still in flight")
	arenaCalls, opCalls := 0, 0
	projectorVKArenaClose = func(*vulkan.VkTensorArena) error { arenaCalls++; return arenaFailure }
	projectorVKOpClose = func(*vulkan.VkLinearF32) error { opCalls++; return nil }
	p := &DeviceStackingProjector{Backend: "vulkan", vkArena: arena, vkOp: op}
	if err := p.Close(); !errors.Is(err, arenaFailure) || p.vkArena != arena || p.vkOp != op || !p.closed || arenaCalls != 1 || opCalls != 0 {
		t.Fatalf("failed arena close dropped owners: arena=%p op=%p calls=%d/%d err=%v", p.vkArena, p.vkOp, arenaCalls, opCalls, err)
	}
	if err := p.Close(); !errors.Is(err, arenaFailure) || arenaCalls != 1 || opCalls != 0 {
		t.Fatalf("repeated close lost error or retried: calls=%d/%d err=%v", arenaCalls, opCalls, err)
	}
	projectorVKArenaClose = func(*vulkan.VkTensorArena) error { arenaCalls++; return nil }
	projectorVKOpClose = func(*vulkan.VkLinearF32) error { opCalls++; return opFailure }
	p = &DeviceStackingProjector{Backend: "vulkan", vkArena: arena, vkOp: op}
	if err := p.Close(); !errors.Is(err, opFailure) || p.vkArena != nil || p.vkOp != op || arenaCalls != 2 || opCalls != 1 {
		t.Fatalf("failed operator close lost owner or double-freed arena: arena=%p op=%p calls=%d/%d err=%v", p.vkArena, p.vkOp, arenaCalls, opCalls, err)
	}
	if err := p.Close(); !errors.Is(err, opFailure) || arenaCalls != 2 || opCalls != 1 {
		t.Fatalf("repeated operator close retried: calls=%d/%d err=%v", arenaCalls, opCalls, err)
	}
	if _, err := p.Project(context.Background(), nil, nil, 1); !errors.Is(err, opFailure) {
		t.Fatalf("projection hid terminal close error: %v", err)
	}
}

func TestPTXLayerAndProjectorCleanupFailures(t *testing.T) {
	failure := errors.New("driver failure")
	frees := 0
	fakePTXCleanup(t, func() error { return nil }, func(b *ptx.Buffer) error {
		frees++
		if b.Ptr == 8 {
			return failure
		}
		b.Ptr = 0
		return nil
	})
	layer := &PTXAudioLayer{maxRows: 1, owned: []*ptx.Buffer{{Ptr: 8}, {Ptr: 9}}}
	if err := layer.Close(); !errors.Is(err, failure) || layer.owned[0].Ptr != 8 || layer.owned[1] != nil {
		t.Fatalf("layer discarded failed owner/double free: owned=%v err=%v", layer.owned, err)
	}
	if err := layer.Close(); !errors.Is(err, failure) || frees != 2 {
		t.Fatalf("layer retried driver after terminal error: frees=%d err=%v", frees, err)
	}
	if _, err := layer.Forward(make([]float32, projectedWidth), 1); !errors.Is(err, failure) {
		t.Fatalf("layer accepted work after terminal error: %v", err)
	}
	if err := layer.ForwardBuffer(&ptx.Buffer{Ptr: 40, Size: projectedWidth * 4}, &ptx.Buffer{Ptr: 80, Size: projectedWidth * 4}, 1); !errors.Is(err, failure) {
		t.Fatalf("device forward hid terminal error: %v", err)
	}
	p := &DeviceStackingProjector{Backend: "ptx", ptxX: &ptx.Buffer{Ptr: 8}, ptxW: &ptx.Buffer{Ptr: 10}}
	if err := p.Close(); !errors.Is(err, failure) || p.ptxX.Ptr != 8 || p.ptxW.Ptr != 10 {
		t.Fatalf("projector discarded owners: %+v err=%v", p, err)
	}
	if err := p.Close(); !errors.Is(err, failure) || frees != 3 {
		t.Fatalf("projector retried: frees=%d err=%v", frees, err)
	}
	if _, err := p.Project(context.Background(), nil, nil, 1); !errors.Is(err, failure) {
		t.Fatalf("projector hid terminal error: %v", err)
	}
	p = &DeviceStackingProjector{Backend: "ptx", terminal: failure, ptxX: &ptx.Buffer{Ptr: 12}}
	if err := p.Close(); !errors.Is(err, failure) || p.ptxX != nil {
		t.Fatalf("projector failed to release after confirmed sync: %+v err=%v", p, err)
	}
	p = &DeviceStackingProjector{Backend: "ptx", terminal: failure, uncertain: true, ptxX: &ptx.Buffer{Ptr: 13}}
	before := frees
	if err := p.Close(); !errors.Is(err, failure) || p.ptxX.Ptr != 13 || frees != before {
		t.Fatalf("projector freed after uncertain result: %+v frees=%d err=%v", p, frees, err)
	}
	s := &PTXStreamingTower{tower: &PTXAudioTower{terminal: failure}}
	if err := s.Close(); !errors.Is(err, failure) || s.tower == nil {
		t.Fatalf("streaming owner lost failed tower: %v", err)
	}
	if _, err := s.Forward(context.Background(), nil, 1); !errors.Is(err, failure) {
		t.Fatalf("streaming owner hid terminal error: %v", err)
	}
}
