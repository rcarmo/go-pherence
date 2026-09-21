package memory

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
	"sort"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// RangeState tracks the residency state of an mmap'd range.
type RangeState int

const (
	RangeCold        RangeState = iota // not recently used
	RangePrefetching                   // WILLNEED issued
	RangeHot                           // recently accessed
)

// AdvisedRange tracks a page-aligned region of the mmap'd file.
type AdvisedRange struct {
	Offset   int64
	Bytes    int64
	State    RangeState
	Hits     uint64
	Evicts   uint64
	LastUsed int64 // unix nanos
}

// MmapAdvisor manages madvise hints on an mmap'd byte region.
// It tracks per-range residency and provides hit/evict counters
// for budget tuning (inspired by ds4 streaming PR). The owner must call Detach
// before unmapping. Stats are conservative advice bookkeeping, not measured RSS.
type MmapAdvisor struct {
	mu       sync.Mutex
	base     []byte // the mmap'd region
	pageSize int64
	advise   func([]byte, int) error // nil uses syscall.Madvise; injected by host tests

	// Per-range tracking (keyed by page-aligned offset)
	ranges map[int64]*AdvisedRange

	// Global counters
	TotalPrefetches atomic.Uint64
	TotalEvictions  atomic.Uint64
	TotalBytes      atomic.Int64 // currently non-cold bytes
	PeakBytes       int64
}

// NewMmapAdvisor creates an advisor for a mmap'd byte region.
func NewMmapAdvisor(mmapData []byte) *MmapAdvisor {
	ps := int64(syscall.Getpagesize())
	if ps <= 0 {
		ps = 4096
	}
	return &MmapAdvisor{
		base:     mmapData,
		pageSize: ps,
		ranges:   make(map[int64]*AdvisedRange),
	}
}

// Detach waits for in-flight advice and disables future mapping access. It does
// not unmap memory or protect independent raw readers. Retained advisor handles
// become inert; the mapping owner must detach every advisor it created.
func (a *MmapAdvisor) Detach() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.base = nil
	a.ranges = nil
	a.TotalBytes.Store(0)
}
func (a *MmapAdvisor) madviseLocked(off, sz int64, advice int) error {
	if a.advise != nil {
		return a.advise(a.base[off:off+sz], advice)
	}
	return syscall.Madvise(a.base[off:off+sz], advice)
}

// align returns (page-aligned offset, page-aligned size).
func (a *MmapAdvisor) align(offset, bytes int64) (int64, int64) {
	if a == nil || a.pageSize <= 0 {
		return 0, 0
	}
	alignedOff := offset &^ (a.pageSize - 1)
	leading := offset - alignedOff
	alignedBytes := ((leading + bytes) + a.pageSize - 1) &^ (a.pageSize - 1)
	return alignedOff, alignedBytes
}

func (a *MmapAdvisor) boundedRange(offset, bytes int64) (int64, int64, bool) {
	if a == nil {
		return 0, 0, false
	}
	baseLen := int64(len(a.base))
	if baseLen == 0 || bytes <= 0 || offset < 0 || offset >= baseLen {
		return 0, 0, false
	}
	// Clamp the requested byte count before page alignment so huge caller
	// values cannot overflow align's leading+bytes arithmetic.
	if bytes > baseLen-offset {
		bytes = baseLen - offset
	}
	off, sz := a.align(offset, bytes)
	if off < 0 || off >= baseLen {
		return 0, 0, false
	}
	if sz > baseLen-off {
		sz = baseLen - off
	}
	return off, sz, sz > 0
}

// Prefetch issues madvise(MADV_WILLNEED) for a range, pre-faulting pages.
func (a *MmapAdvisor) Prefetch(offset, bytes int64) error {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off, sz, ok := a.boundedRange(offset, bytes)
	if !ok {
		return nil
	}

	if err := a.madviseLocked(off, sz, syscall.MADV_WILLNEED); err != nil {
		return fmt.Errorf("madvise WILLNEED [%d:%d]: %w", off, off+sz, err)
	}

	r, ok := a.ranges[off]
	if !ok {
		r = &AdvisedRange{Offset: off}
		a.ranges[off] = r
	}
	r.Bytes = max(r.Bytes, sz)
	r.State = RangePrefetching
	r.Hits++
	r.LastUsed = time.Now().UnixNano()
	a.recomputeTotalsLocked()

	a.TotalPrefetches.Add(1)
	return nil
}

// Touch marks a range as actively used (hot).
func (a *MmapAdvisor) Touch(offset, bytes int64) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off, sz, ok := a.boundedRange(offset, bytes)
	if !ok {
		return
	}

	r, ok := a.ranges[off]
	if !ok {
		r = &AdvisedRange{Offset: off}
		a.ranges[off] = r
	}
	r.Bytes = max(r.Bytes, sz)
	r.State = RangeHot
	r.Hits++
	r.LastUsed = time.Now().UnixNano()
	a.recomputeTotalsLocked()
}

// Evict issues madvise(MADV_DONTNEED) for a range, releasing pages.
func (a *MmapAdvisor) Evict(offset, bytes int64) error {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	off, sz, ok := a.boundedRange(offset, bytes)
	if !ok {
		return nil
	}
	return a.evictLocked(off, sz)
}
func (a *MmapAdvisor) evictLocked(off, sz int64) error {
	if err := a.madviseLocked(off, sz, syscall.MADV_DONTNEED); err != nil {
		return fmt.Errorf("madvise DONTNEED [%d:%d]: %w", off, off+sz, err)
	}
	// A partial eviction does not make a whole tracked range cold. Keep that
	// range conservatively hot until its complete extent has been evicted.
	for _, r := range a.ranges {
		if r != nil && r.Offset >= off && checked.SaturatingAddInt64(r.Offset, r.Bytes) <= off+sz {
			r.State = RangeCold
			r.Evicts = saturatingAddUint64(r.Evicts, 1)
		}
	}
	a.recomputeTotalsLocked()
	a.TotalEvictions.Add(1)
	return nil
}

// EvictCold serialises eligibility, advice and bookkeeping with Touch/Detach.
// A recent overlapping range protects old aliases of the same pages.
func (a *MmapAdvisor) EvictCold(cutoffNanos int64) (int, error) {
	if a == nil {
		return 0, nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	count := 0
	for _, r := range a.ranges {
		if r == nil || r.State == RangeCold || r.LastUsed >= cutoffNanos {
			continue
		}
		off, sz, ok := a.boundedRange(r.Offset, r.Bytes)
		if !ok {
			continue
		}
		recent := false
		for _, other := range a.ranges {
			if other != nil && other.State != RangeCold && other.LastUsed >= cutoffNanos && other.Offset < off+sz && checked.SaturatingAddInt64(other.Offset, other.Bytes) > off {
				recent = true
				break
			}
		}
		if recent {
			continue
		}
		if err := a.evictLocked(off, sz); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

// MergeRanges coalesces overlapping/adjacent tracked ranges.
// Like ds4 hot_plan_merge.
func (a *MmapAdvisor) MergeRanges() {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(a.ranges) < 2 {
		return
	}

	// Collect and sort
	sorted := make([]*AdvisedRange, 0, len(a.ranges))
	for _, r := range a.ranges {
		sorted = append(sorted, r)
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Offset < sorted[j].Offset
	})

	// Merge overlapping/adjacent
	merged := make(map[int64]*AdvisedRange)
	cur := sanitizeRange(*sorted[0])
	for i := 1; i < len(sorted); i++ {
		rv := sanitizeRange(*sorted[i])
		r := &rv
		curEnd := checked.SaturatingAddInt64(cur.Offset, cur.Bytes)
		if r.Offset <= curEnd && mergeCompatible(cur.State, r.State) {
			// Overlapping or adjacent with equivalent residency — extend
			rEnd := checked.SaturatingAddInt64(r.Offset, r.Bytes)
			if rEnd > curEnd {
				cur.Bytes = saturatingSubInt64(rEnd, cur.Offset)
			}
			cur.Hits = saturatingAddUint64(cur.Hits, r.Hits)
			cur.Evicts = saturatingAddUint64(cur.Evicts, r.Evicts)
			if r.LastUsed > cur.LastUsed {
				cur.LastUsed = r.LastUsed
			}
			if r.State > cur.State {
				cur.State = r.State
			}
		} else {
			// Gap — emit current, start new
			c := cur
			merged[c.Offset] = &c
			cur = *r
		}
	}
	c := cur
	merged[c.Offset] = &c
	a.ranges = merged
	a.recomputeTotalsLocked()
}

func mergeCompatible(a, b RangeState) bool {
	return (a == RangeCold) == (b == RangeCold)
}

func sanitizeRange(r AdvisedRange) AdvisedRange {
	if r.Offset < 0 {
		r.Offset = 0
	}
	if r.Bytes < 0 {
		r.Bytes = 0
	}
	return r
}

func saturatingSubInt64(a, b int64) int64 {
	if a <= b {
		return 0
	}
	return a - b
}

func saturatingAddUint64(a, b uint64) uint64 {
	if a > ^uint64(0)-b {
		return ^uint64(0)
	}
	return a + b
}

func (a *MmapAdvisor) recomputeTotalsLocked() {
	if a == nil {
		return
	}
	// Count the union of tracked non-cold extents, not their overlapping sum.
	ranges := make([]AdvisedRange, 0, len(a.ranges))
	for _, r := range a.ranges {
		if r != nil {
			*r = sanitizeRange(*r)
			if r.State != RangeCold && r.Bytes > 0 {
				ranges = append(ranges, *r)
			}
		}
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].Offset < ranges[j].Offset })
	var total, end int64
	for _, r := range ranges {
		// Corrupted internal extents cannot have a representable union; keep the
		// legacy conservative saturation instead of undercounting a clipped endpoint.
		if r.Offset > checked.MaxInt64()-r.Bytes {
			total = checked.MaxInt64()
			break
		}
		next := checked.SaturatingAddInt64(r.Offset, r.Bytes)
		start := max(end, r.Offset)
		if next > start {
			total = checked.SaturatingAddInt64(total, next-start)
		}
		end = max(end, next)
	}
	a.TotalBytes.Store(total)
	if total > a.PeakBytes {
		a.PeakBytes = total
	}
}

// Stats returns (numRanges, totalHotBytes, peakBytes).
func (a *MmapAdvisor) Stats() (int, int64, int64) {
	if a == nil {
		return 0, 0, 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.ranges), a.TotalBytes.Load(), a.PeakBytes
}

// RangeStats returns a copy of all tracked ranges for reporting.
func (a *MmapAdvisor) RangeStats() []AdvisedRange {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]AdvisedRange, 0, len(a.ranges))
	for _, r := range a.ranges {
		out = append(out, *r)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Offset < out[j].Offset
	})
	return out
}
