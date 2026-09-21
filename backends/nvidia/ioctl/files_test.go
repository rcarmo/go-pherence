package nv

import (
	"errors"
	"os"
	"sync"
	"testing"
)

func TestDescriptorCleanupAllFailureStages(t *testing.T) {
	for fail := 0; fail <= 5; fail++ {
		t.Run(string(rune('0'+fail)), func(t *testing.T) {
			d := &NVDevice{}
			opened := 0
			closed := map[int]int{}
			err := d.openDeviceFiles(func(string, int, uint32) (int, error) {
				if opened == fail {
					return -1, errors.New("open")
				}
				fd := opened
				opened++
				return fd, nil
			}, func(fd int) error { closed[fd]++; return nil })
			if fail < 5 && err == nil {
				t.Fatal("expected error")
			}
			if fail == 5 && err != nil {
				t.Fatal(err)
			}
			d.closeDeviceFiles(func(fd int) error { closed[fd]++; return nil })
			d.closeDeviceFiles(func(fd int) error { t.Fatal("double close"); return nil })
			for fd := 0; fd < opened; fd++ {
				if closed[fd] != 1 {
					t.Fatal(fd, closed)
				}
			}
		})
	}
	d := &NVDevice{}
	d.closeDeviceFiles(func(int) error { t.Fatal("zero value closed stdin"); return nil })
}
func TestHostAllocationCountersConcurrent(t *testing.T) {
	d := &NVDevice{handleCounter: 100, vaAllocator: 4096}
	var wg sync.WaitGroup
	ids := make(chan uint32, 128)
	vas := make(chan uint64, 128)
	for i := 0; i < 128; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); ids <- d.nextHandle(); vas <- d.allocVA(1) }()
	}
	wg.Wait()
	close(ids)
	close(vas)
	seen := map[uint32]bool{}
	for id := range ids {
		if id == 0 || seen[id] {
			t.Fatal(id)
		}
		seen[id] = true
	}
	seenVA := map[uint64]bool{}
	for va := range vas {
		if va == 0 || seenVA[va] {
			t.Fatal(va)
		}
		seenVA[va] = true
	}
	d.handleCounter = ^uint32(0)
	if d.nextHandle() != 0 {
		t.Fatal("handle wrap")
	}
}
func TestMappingDescriptorReleasedOnce(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "mapping")
	if err != nil {
		t.Fatal(err)
	}
	b := &NVBuffer{mappingFile: f}
	b.Free()
	b.Free()
	if _, err = f.Stat(); err == nil {
		t.Fatal("mapping descriptor still open")
	}
}
