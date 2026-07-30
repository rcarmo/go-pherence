package nvidia

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/rcarmo/go-pherence/backends/placement"
)

// ExpertCachePolicy names the eviction policy used by ExpertPool.
type ExpertCachePolicy string

const (
	ExpertCachePolicyLRU ExpertCachePolicy = "lru"
	ExpertCachePolicyLFU ExpertCachePolicy = "lfu"
)

// ParseExpertCachePolicy normalizes a user-supplied expert cache policy name.
// The empty string defaults to LRU for backwards compatibility.
func ParseExpertCachePolicy(name string) (ExpertCachePolicy, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", string(ExpertCachePolicyLRU):
		return ExpertCachePolicyLRU, nil
	case string(ExpertCachePolicyLFU):
		return ExpertCachePolicyLFU, nil
	default:
		return "", fmt.Errorf("unsupported expert cache policy %q", name)
	}
}

func normalizeExpertCachePolicy(policy ExpertCachePolicy) ExpertCachePolicy {
	normalized, err := ParseExpertCachePolicy(string(policy))
	if err != nil {
		return ExpertCachePolicyLRU
	}
	return normalized
}

func (p ExpertCachePolicy) String() string {
	return string(normalizeExpertCachePolicy(p))
}

type expertCacheHistory struct {
	UseCount uint64
	Recency  uint64
}

// ExpertPool manages a fixed number of MoE expert weight sets on GPU.
// Experts are cached with configurable LRU/LFU eviction and hit/miss tracking.
type ExpertPool struct {
	mu      sync.Mutex
	slots   int                        // max experts on GPU simultaneously
	policy  ExpertCachePolicy          // eviction policy for automatic replacement
	cache   map[int]*ExpertEntry       // expert_id → cached entry
	order   []int                      // LRU order (most recent at end)
	history map[int]expertCacheHistory // whole-run use counts and recency by global key
	clock   uint64                     // monotonically increasing recency clock
	budget  *placement.BudgetManager   // optional budget tracking

	// Stats
	Hits   atomic.Uint64
	Misses atomic.Uint64
	Evicts atomic.Uint64
}

// ExpertEntry holds one expert's GPU-resident weights.
type ExpertEntry struct {
	ExpertID  int
	GateW     *GPUMLXWeight // gate projection [hidden → moe_inter]
	UpW       *GPUMLXWeight // up projection [hidden → moe_inter]
	DownW     *GPUMLXWeight // down projection [moe_inter → hidden]
	SizeBytes int64         // total VRAM used
}

// NewExpertPool creates a pool with the given number of GPU slots using the
// historical default LRU eviction policy.
func NewExpertPool(slots int, budget *placement.BudgetManager) *ExpertPool {
	return NewExpertPoolWithPolicy(slots, budget, ExpertCachePolicyLRU)
}

// NewExpertPoolWithPolicy creates a pool with the given number of GPU slots
// and eviction policy. Unsupported policy values fall back to LRU.
func NewExpertPoolWithPolicy(slots int, budget *placement.BudgetManager, policy ExpertCachePolicy) *ExpertPool {
	return &ExpertPool{
		slots:   slots,
		policy:  normalizeExpertCachePolicy(policy),
		cache:   make(map[int]*ExpertEntry),
		history: make(map[int]expertCacheHistory),
		budget:  budget,
	}
}

// Policy returns the pool's configured eviction policy.
func (p *ExpertPool) Policy() ExpertCachePolicy {
	if p == nil {
		return ExpertCachePolicyLRU
	}
	return normalizeExpertCachePolicy(p.policy)
}

// Peek returns the cached expert without changing hit/miss/eviction stats,
// use counts, recency, or LRU order. Use it for planning checks; use Get for
// actual expert use.
func (p *ExpertPool) Peek(expertID int) *ExpertEntry {
	if p == nil || expertID < 0 {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.cache[expertID]
}

// Get returns the cached expert, or nil if not present (miss).
// On hit, the expert is moved to the most-recently-used position and its whole-
// run use count/recency are updated.
func (p *ExpertPool) Get(expertID int) *ExpertEntry {
	if p == nil || expertID < 0 {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	entry, ok := p.cache[expertID]
	if ok {
		p.recordUseLocked(expertID, true)
		p.touchLocked(expertID)
		p.Hits.Add(1)
		if p.budget != nil {
			p.budget.Hit(placement.BudgetExpert)
		}
		return entry
	}
	p.recordUseLocked(expertID, false)
	p.Misses.Add(1)
	return nil
}

// Put adds an expert to the pool, evicting according to the configured policy
// if full. Returns the evicted entry (if any) so the caller can free its GPU
// resources.
func (p *ExpertPool) Put(entry *ExpertEntry) *ExpertEntry {
	if entry == nil {
		return nil
	}
	if p == nil || entry.ExpertID < 0 {
		return entry
	}
	if p.slots <= 0 {
		// A zero-slot pool is disabled. Return the entry so callers that just
		// uploaded GPU resources can release it via FreeExpertEntry.
		return entry
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	// Already cached? Replace the old GPU resources and return them for release.
	if old, ok := p.cache[entry.ExpertID]; ok {
		p.cache[entry.ExpertID] = entry
		p.touchLocked(entry.ExpertID)
		p.touchRecencyLocked(entry.ExpertID)
		if p.budget != nil {
			p.budget.Free(placement.BudgetExpert, old.SizeBytes)
			p.budget.Alloc(placement.BudgetExpert, entry.SizeBytes)
		}
		return old
	}

	var evicted *ExpertEntry
	if len(p.cache) >= p.slots {
		evicted = p.evictPolicyLocked()
	}

	p.cache[entry.ExpertID] = entry
	p.touchLocked(entry.ExpertID)
	p.touchRecencyLocked(entry.ExpertID)

	if p.budget != nil {
		p.budget.Alloc(placement.BudgetExpert, entry.SizeBytes)
	}

	return evicted
}

// EvictLRU explicitly evicts the least-recently-used expert.
// Returns the evicted entry or nil if pool is empty.
func (p *ExpertPool) EvictLRU() *ExpertEntry {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.evictLRULocked()
}

func (p *ExpertPool) evictPolicyLocked() *ExpertEntry {
	if p == nil {
		return nil
	}
	switch normalizeExpertCachePolicy(p.policy) {
	case ExpertCachePolicyLFU:
		return p.evictLFULocked()
	default:
		return p.evictLRULocked()
	}
}

func (p *ExpertPool) evictLRULocked() *ExpertEntry {
	if p == nil || len(p.order) == 0 {
		return nil
	}
	lruID := p.order[0]
	return p.evictByIDLocked(lruID)
}

func (p *ExpertPool) evictLFULocked() *ExpertEntry {
	if p == nil || len(p.cache) == 0 {
		return nil
	}
	var (
		victimID      int
		victimUse     uint64
		victimRecency uint64
		haveVictim    bool
	)
	for expertID := range p.cache {
		h := p.history[expertID]
		if !haveVictim || h.UseCount < victimUse || (h.UseCount == victimUse && h.Recency < victimRecency) {
			victimID = expertID
			victimUse = h.UseCount
			victimRecency = h.Recency
			haveVictim = true
		}
	}
	if !haveVictim {
		return nil
	}
	return p.evictByIDLocked(victimID)
}

func (p *ExpertPool) evictByIDLocked(expertID int) *ExpertEntry {
	if p == nil || expertID < 0 {
		return nil
	}
	entry, ok := p.cache[expertID]
	if !ok {
		p.removeFromOrderLocked(expertID)
		return nil
	}
	delete(p.cache, expertID)
	p.removeFromOrderLocked(expertID)
	p.Evicts.Add(1)
	if p.budget != nil {
		p.budget.Free(placement.BudgetExpert, entry.SizeBytes)
		p.budget.Evict(placement.BudgetExpert)
	}
	return entry
}

func (p *ExpertPool) removeFromOrderLocked(expertID int) {
	if p == nil || expertID < 0 {
		return
	}
	for i, id := range p.order {
		if id == expertID {
			p.order = append(p.order[:i], p.order[i+1:]...)
			return
		}
	}
}

func (p *ExpertPool) touchLocked(expertID int) {
	if p == nil || expertID < 0 {
		return
	}
	p.removeFromOrderLocked(expertID)
	p.order = append(p.order, expertID)
}

func (p *ExpertPool) recordUseLocked(expertID int, resident bool) {
	if p == nil || expertID < 0 {
		return
	}
	h := p.history[expertID]
	h.UseCount++
	if resident {
		p.clock++
		h.Recency = p.clock
	}
	p.history[expertID] = h
}

func (p *ExpertPool) touchRecencyLocked(expertID int) {
	if p == nil || expertID < 0 {
		return
	}
	h := p.history[expertID]
	p.clock++
	h.Recency = p.clock
	p.history[expertID] = h
}

// Size returns the number of currently cached experts.
func (p *ExpertPool) Size() int {
	if p == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.cache)
}

// Slots returns the maximum number of expert slots.
func (p *ExpertPool) Slots() int {
	if p == nil {
		return 0
	}
	return p.slots
}

// Report returns a human-readable summary.
func (p *ExpertPool) Report() string {
	if p == nil {
		return "experts: 0/0 cached  policy=lru hits=0 misses=0 evicts=0"
	}
	p.mu.Lock()
	n := len(p.cache)
	policy := p.policy.String()
	p.mu.Unlock()
	return fmt.Sprintf("experts: %d/%d cached  policy=%s hits=%d misses=%d evicts=%d",
		n, p.slots, policy,
		p.Hits.Load(), p.Misses.Load(), p.Evicts.Load())
}

// FreeExpertEntry releases GPU resources for an evicted expert.
func FreeExpertEntry(e *ExpertEntry) {
	if e == nil {
		return
	}
	if e.GateW != nil {
		e.GateW.Free()
	}
	if e.UpW != nil {
		e.UpW.Free()
	}
	if e.DownW != nil {
		e.DownW.Free()
	}
}
