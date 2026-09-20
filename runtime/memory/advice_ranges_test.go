package memory

import (
	"syscall"
	"testing"
	"time"
)

func mappedAdvisor(t *testing.T) (*MmapAdvisor, int64) {
	t.Helper()
	page := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, page*8, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(data); err != nil {
			t.Error(err)
		}
	})
	return NewMmapAdvisor(data), int64(page)
}
func TestAdviceOverlapUnionAndPartialEviction(t *testing.T) {
	a, p := mappedAdvisor(t)
	a.Touch(0, 4*p)
	a.Touch(p, 4*p)
	if _, hot, peak := a.Stats(); hot != 5*p || peak != 5*p {
		t.Fatalf("overlap hot/peak=%d/%d want %d", hot, peak, 5*p)
	}
	a.Touch(0, p)
	if _, hot, _ := a.Stats(); hot != 5*p {
		t.Fatal("short touch discarded existing extent", hot)
	}
	if err := a.Evict(0, p); err != nil {
		t.Fatal(err)
	}
	if _, hot, _ := a.Stats(); hot != 5*p {
		t.Fatal("partial eviction cleared whole extent", hot)
	}
	if err := a.Evict(0, 5*p); err != nil {
		t.Fatal(err)
	}
	if _, hot, peak := a.Stats(); hot != 0 || peak != 5*p {
		t.Fatal(hot, peak)
	}
}
func TestColdEvictionProtectsRecentOverlap(t *testing.T) {
	a, p := mappedAdvisor(t)
	a.Touch(0, 4*p)
	a.Touch(p, p)
	cutoff := time.Now().UnixNano()
	a.ranges[0].LastUsed = cutoff - 1
	a.ranges[p].LastUsed = cutoff
	if n, err := a.EvictCold(cutoff); err != nil || n != 0 {
		t.Fatal("evicted recently touched overlapping page", n, err)
	}
	if _, hot, _ := a.Stats(); hot != 4*p {
		t.Fatal(hot)
	}
	a.ranges[p].LastUsed = cutoff - 1
	if n, err := a.EvictCold(cutoff); err != nil || n < 1 {
		t.Fatal(n, err)
	}
	if _, hot, _ := a.Stats(); hot != 0 {
		t.Fatal(hot)
	}
}
