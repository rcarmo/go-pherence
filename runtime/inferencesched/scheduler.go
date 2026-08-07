package inferencesched

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"
)

var (
	ErrClosed          = errors.New("inferencesched: closed")
	ErrNilRequest      = errors.New("inferencesched: nil request")
	ErrNilWork         = errors.New("inferencesched: nil work")
	ErrEmptyRequestID  = errors.New("inferencesched: empty request id")
	ErrDuplicateID     = errors.New("inferencesched: duplicate request id")
	ErrOverflow        = errors.New("inferencesched: overflow")
	ErrInvalidProgress = errors.New("inferencesched: invalid progress")
)

// Request identifies a resumable session to be scheduled.
type Request interface {
	ID() string
	ArrivalSeq() uint64
}

// Work encapsulates model-neutral incremental inference work.
type Work interface {
	RemainingPrefill() int
	Prefill(maxTokens int) (int, error)
	Decode(maxTokens int) (int, error)
	Finished() bool
	Cancel()
	Close() error
}

// Config controls scheduler limits.
type Config struct {
	MaxActive     int
	DecodeBudget  int
	PrefillBudget int
}

// Validate checks the scheduler configuration.
func (cfg Config) Validate() error {
	if cfg.MaxActive <= 0 {
		return fmt.Errorf("inferencesched: max active must be positive")
	}
	if cfg.DecodeBudget < 0 {
		return fmt.Errorf("inferencesched: decode budget must be non-negative")
	}
	if cfg.PrefillBudget < 0 {
		return fmt.Errorf("inferencesched: prefill budget must be non-negative")
	}
	return nil
}

// StepResult aggregates everything that happened to one request in a Step.
type StepResult struct {
	RequestID      string
	ArrivalSeq     uint64
	TokensConsumed int
	TokensEmitted  int
	Admitted       bool
	Finished       bool
	Canceled       bool
	Err            error
}

// StepStats summarizes one scheduling step.
type StepStats struct {
	DecodeBudget   int
	PrefillBudget  int
	DecodeUsed     int
	PrefillUsed    int
	TokensConsumed int
	TokensEmitted  int
	Admitted       int
	Finished       int
	Canceled       int
	Errors         int
	Waiting        int
	Running        int
}

// StepReport is the result of one Step invocation.
type StepReport struct {
	Results []StepResult
	Stats   StepStats
}

// Scheduler is a minimal model-neutral decode/prefill scheduler.
type Scheduler struct {
	mu            sync.Mutex
	cfg           Config
	closed        bool
	waiting       []*entry
	running       []*entry
	index         map[string]*entry
	addOrder      uint64
	decodeCursor  int
	prefillCursor int
}

type entry struct {
	ctx   context.Context
	req   Request
	work  Work
	order uint64
}

// New constructs a scheduler from a validated config.
func New(cfg Config) (*Scheduler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Scheduler{
		cfg:   cfg,
		index: make(map[string]*entry),
	}, nil
}

// Add enqueues a request for later admission.
func (s *Scheduler) Add(ctx context.Context, req Request, work Work) error {
	if s == nil {
		return fmt.Errorf("inferencesched: nil scheduler")
	}
	if req == nil {
		return ErrNilRequest
	}
	if work == nil {
		return ErrNilWork
	}
	if req.ID() == "" {
		return ErrEmptyRequestID
	}
	if ctx == nil {
		ctx = context.Background()
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrClosed
	}
	if _, exists := s.index[req.ID()]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateID, req.ID())
	}
	if s.addOrder == math.MaxUint64 {
		return ErrOverflow
	}
	ent := &entry{ctx: ctx, req: req, work: work, order: s.addOrder}
	s.addOrder++
	s.index[req.ID()] = ent
	pos := sort.Search(len(s.waiting), func(i int) bool {
		return entryLess(ent, s.waiting[i])
	})
	s.waiting = insertEntry(s.waiting, pos, ent)
	return nil
}

// Step performs one deterministic scheduling step.
func (s *Scheduler) Step(ctx context.Context) (StepReport, error) {
	if s == nil {
		return StepReport{}, fmt.Errorf("inferencesched: nil scheduler")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return StepReport{}, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return StepReport{}, ErrClosed
	}

	builder := stepBuilder{
		byID: make(map[string]*StepResult),
		stats: StepStats{
			DecodeBudget:  s.cfg.DecodeBudget,
			PrefillBudget: s.cfg.PrefillBudget,
		},
	}
	dead := make(map[*entry]struct{})

	s.cancelDoneWaiting(&builder, dead)
	s.cancelDoneRunning(&builder, dead)
	s.compact(dead)

	for len(s.running) < s.cfg.MaxActive && len(s.waiting) > 0 {
		ent := s.waiting[0]
		s.waiting[0] = nil
		s.waiting = s.waiting[1:]
		s.running = append(s.running, ent)
		builder.markAdmitted(ent)
	}

	s.finishCompletedRunning(&builder, dead)

	if s.cfg.DecodeBudget > 0 && len(s.running) > 0 {
		s.runDecode(&builder, dead)
	}
	if s.cfg.PrefillBudget > 0 && len(s.running) > 0 {
		s.runPrefill(&builder, dead)
	}

	s.cancelDoneWaiting(&builder, dead)
	s.finishOrCancelRunning(&builder, dead)
	s.compact(dead)

	builder.stats.Waiting = len(s.waiting)
	builder.stats.Running = len(s.running)
	return builder.report(), nil
}

// Close cancels all queued/running work and releases them exactly once.
func (s *Scheduler) Close() error {
	if s == nil {
		return nil
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	waiting := s.waiting
	running := s.running
	s.waiting = nil
	s.running = nil
	s.index = map[string]*entry{}
	s.mu.Unlock()

	var errs []error
	for _, ent := range append(append([]*entry(nil), waiting...), running...) {
		if ent == nil {
			continue
		}
		ent.work.Cancel()
		if err := ent.work.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (s *Scheduler) cancelDoneWaiting(builder *stepBuilder, dead map[*entry]struct{}) {
	for _, ent := range s.waiting {
		if ent == nil || isDead(dead, ent) {
			continue
		}
		if ent.ctx.Err() == nil {
			continue
		}
		s.cancelEntry(ent, builder, dead)
	}
}

func (s *Scheduler) cancelDoneRunning(builder *stepBuilder, dead map[*entry]struct{}) {
	for _, ent := range s.running {
		if ent == nil || isDead(dead, ent) {
			continue
		}
		if ent.ctx.Err() == nil {
			continue
		}
		s.cancelEntry(ent, builder, dead)
	}
}

func (s *Scheduler) finishCompletedRunning(builder *stepBuilder, dead map[*entry]struct{}) {
	for _, ent := range s.running {
		if ent == nil || isDead(dead, ent) {
			continue
		}
		if ent.work.Finished() {
			s.finishEntry(ent, builder, dead)
		}
	}
}

func (s *Scheduler) finishOrCancelRunning(builder *stepBuilder, dead map[*entry]struct{}) {
	for _, ent := range s.running {
		if ent == nil || isDead(dead, ent) {
			continue
		}
		if ent.work.Finished() {
			s.finishEntry(ent, builder, dead)
			continue
		}
		if ent.ctx.Err() != nil {
			s.cancelEntry(ent, builder, dead)
		}
	}
}

func (s *Scheduler) compact(dead map[*entry]struct{}) {
	s.waiting = compactEntries(s.waiting, dead)
	s.running = compactEntries(s.running, dead)
	if len(s.running) == 0 {
		s.decodeCursor = 0
		s.prefillCursor = 0
		return
	}
	s.decodeCursor %= len(s.running)
	s.prefillCursor %= len(s.running)
}

func (s *Scheduler) runDecode(builder *stepBuilder, dead map[*entry]struct{}) {
	budget := s.cfg.DecodeBudget
	n := len(s.running)
	if budget <= 0 || n == 0 {
		return
	}
	start := normalizeCursor(s.decodeCursor, n)
	idx := start
	visited := 0
	lastUsed := -1
	for visited < n && budget > 0 {
		ent := s.running[idx]
		if ent != nil && !isDead(dead, ent) {
			rem, err := validateRemainingPrefill(ent.work)
			if err != nil {
				s.errorEntry(ent, builder, dead, err)
			} else if rem == 0 && !ent.work.Finished() {
				emitted, callErr := ent.work.Decode(budget)
				if perr := validateProgress(emitted, budget); perr != nil {
					s.errorEntry(ent, builder, dead, perr)
				} else {
					if emitted > 0 {
						builder.addEmitted(ent, emitted)
						budget -= emitted
						lastUsed = idx
					}
					if callErr != nil {
						s.errorEntry(ent, builder, dead, callErr)
					} else if ent.work.Finished() {
						s.finishEntry(ent, builder, dead)
					} else if ent.ctx.Err() != nil {
						s.cancelEntry(ent, builder, dead)
					}
				}
			}
		}
		idx = (idx + 1) % n
		visited++
	}
	if len(s.running) == 0 {
		s.decodeCursor = 0
		return
	}
	if lastUsed >= 0 {
		s.decodeCursor = (lastUsed + 1) % n
		return
	}
	s.decodeCursor = (start + 1) % n
}

func (s *Scheduler) runPrefill(builder *stepBuilder, dead map[*entry]struct{}) {
	budget := s.cfg.PrefillBudget
	n := len(s.running)
	if budget <= 0 || n == 0 {
		return
	}
	start := normalizeCursor(s.prefillCursor, n)
	idx := start
	visited := 0
	lastUsed := -1
	for visited < n && budget > 0 {
		ent := s.running[idx]
		if ent != nil && !isDead(dead, ent) {
			rem, err := validateRemainingPrefill(ent.work)
			if err != nil {
				s.errorEntry(ent, builder, dead, err)
			} else if rem > 0 && !ent.work.Finished() {
				limit := budget
				if rem < limit {
					limit = rem
				}
				consumed, callErr := ent.work.Prefill(limit)
				if perr := validateProgress(consumed, limit); perr != nil {
					s.errorEntry(ent, builder, dead, perr)
				} else {
					if consumed > 0 {
						builder.addConsumed(ent, consumed)
						budget -= consumed
						lastUsed = idx
					}
					if callErr != nil {
						s.errorEntry(ent, builder, dead, callErr)
					} else if ent.work.Finished() {
						s.finishEntry(ent, builder, dead)
					} else if ent.ctx.Err() != nil {
						s.cancelEntry(ent, builder, dead)
					}
				}
			}
		}
		idx = (idx + 1) % n
		visited++
	}
	if len(s.running) == 0 {
		s.prefillCursor = 0
		return
	}
	if lastUsed >= 0 {
		s.prefillCursor = (lastUsed + 1) % n
		return
	}
	s.prefillCursor = (start + 1) % n
}

func (s *Scheduler) cancelEntry(ent *entry, builder *stepBuilder, dead map[*entry]struct{}) {
	if ent == nil || isDead(dead, ent) {
		return
	}
	builder.markCanceled(ent)
	ent.work.Cancel()
	if err := ent.work.Close(); err != nil {
		builder.setError(ent, err)
	}
	delete(s.index, ent.req.ID())
	dead[ent] = struct{}{}
}

func (s *Scheduler) finishEntry(ent *entry, builder *stepBuilder, dead map[*entry]struct{}) {
	if ent == nil || isDead(dead, ent) {
		return
	}
	builder.markFinished(ent)
	if err := ent.work.Close(); err != nil {
		builder.setError(ent, err)
	}
	delete(s.index, ent.req.ID())
	dead[ent] = struct{}{}
}

func (s *Scheduler) errorEntry(ent *entry, builder *stepBuilder, dead map[*entry]struct{}, err error) {
	if ent == nil || isDead(dead, ent) {
		return
	}
	builder.setError(ent, err)
	if closeErr := ent.work.Close(); closeErr != nil {
		builder.setError(ent, closeErr)
	}
	delete(s.index, ent.req.ID())
	dead[ent] = struct{}{}
}

func validateRemainingPrefill(work Work) (int, error) {
	rem := work.RemainingPrefill()
	if rem < 0 {
		return 0, fmt.Errorf("%w: negative remaining prefill", ErrInvalidProgress)
	}
	return rem, nil
}

func validateProgress(progress, limit int) error {
	if progress < 0 {
		return fmt.Errorf("%w: negative progress", ErrInvalidProgress)
	}
	if progress > limit {
		return fmt.Errorf("%w: progress %d exceeds limit %d", ErrInvalidProgress, progress, limit)
	}
	return nil
}

func entryLess(a, b *entry) bool {
	if a.req.ArrivalSeq() != b.req.ArrivalSeq() {
		return a.req.ArrivalSeq() < b.req.ArrivalSeq()
	}
	return a.order < b.order
}

func insertEntry(dst []*entry, pos int, ent *entry) []*entry {
	dst = append(dst, nil)
	copy(dst[pos+1:], dst[pos:])
	dst[pos] = ent
	return dst
}

func compactEntries(src []*entry, dead map[*entry]struct{}) []*entry {
	if len(src) == 0 {
		return nil
	}
	out := src[:0]
	for _, ent := range src {
		if ent == nil || isDead(dead, ent) {
			continue
		}
		out = append(out, ent)
	}
	for i := len(out); i < len(src); i++ {
		src[i] = nil
	}
	return out
}

func normalizeCursor(cursor, n int) int {
	if n == 0 {
		return 0
	}
	if cursor < 0 {
		cursor = 0
	}
	return cursor % n
}

func isDead(dead map[*entry]struct{}, ent *entry) bool {
	_, ok := dead[ent]
	return ok
}

type stepBuilder struct {
	order []*StepResult
	byID  map[string]*StepResult
	stats StepStats
}

func (b *stepBuilder) result(ent *entry) *StepResult {
	id := ent.req.ID()
	if got, ok := b.byID[id]; ok {
		return got
	}
	res := &StepResult{RequestID: id, ArrivalSeq: ent.req.ArrivalSeq()}
	b.byID[id] = res
	b.order = append(b.order, res)
	return res
}

func (b *stepBuilder) markAdmitted(ent *entry) {
	res := b.result(ent)
	if !res.Admitted {
		res.Admitted = true
		b.stats.Admitted++
	}
}

func (b *stepBuilder) markFinished(ent *entry) {
	res := b.result(ent)
	if !res.Finished {
		res.Finished = true
		b.stats.Finished++
	}
}

func (b *stepBuilder) markCanceled(ent *entry) {
	res := b.result(ent)
	if !res.Canceled {
		res.Canceled = true
		b.stats.Canceled++
	}
}

func (b *stepBuilder) setError(ent *entry, err error) {
	if err == nil {
		return
	}
	res := b.result(ent)
	if res.Err == nil {
		res.Err = err
		b.stats.Errors++
		return
	}
	res.Err = errors.Join(res.Err, err)
}

func (b *stepBuilder) addConsumed(ent *entry, tokens int) {
	res := b.result(ent)
	res.TokensConsumed = saturatingAdd(res.TokensConsumed, tokens)
	b.stats.PrefillUsed = saturatingAdd(b.stats.PrefillUsed, tokens)
	b.stats.TokensConsumed = saturatingAdd(b.stats.TokensConsumed, tokens)
}

func (b *stepBuilder) addEmitted(ent *entry, tokens int) {
	res := b.result(ent)
	res.TokensEmitted = saturatingAdd(res.TokensEmitted, tokens)
	b.stats.DecodeUsed = saturatingAdd(b.stats.DecodeUsed, tokens)
	b.stats.TokensEmitted = saturatingAdd(b.stats.TokensEmitted, tokens)
}

func (b *stepBuilder) report() StepReport {
	results := make([]StepResult, len(b.order))
	for i, res := range b.order {
		results[i] = *res
	}
	return StepReport{Results: results, Stats: b.stats}
}

func saturatingAdd(dst, add int) int {
	if add <= 0 {
		return dst
	}
	if dst > math.MaxInt-add {
		return math.MaxInt
	}
	return dst + add
}
