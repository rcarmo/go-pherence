package kv

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// F32KVView exposes a logical oldest-to-newest K/V window without forcing a
// copy. Ring-backed stores can return two segments when the logical view wraps.
type F32KVView struct {
	FirstK     []float32
	FirstV     []float32
	SecondK    []float32
	SecondV    []float32
	StartToken int
}

// F32KVStore stores paired float32 K/V rows of a fixed width.
type F32KVStore interface {
	Append(k, v []float32) error
	Tokens() int
	Dim() int
	Capacity() int
	View() F32KVView
	Materialize() (k, v []float32, startToken int)
	Reset()
	Bytes() int64
	Checkpoint() F32KVCheckpoint
	Restore(F32KVCheckpoint) error
}

// F32KVCheckpoint captures a restorable logical K/V state.
type F32KVCheckpoint struct {
	valid      bool
	dim        int
	capacity   int
	startToken int
	tokens     int
	k          []float32
	v          []float32
}

// LinearF32KV is an unbounded append-only logical K/V store.
type LinearF32KV struct {
	dim        int
	startToken int
	tokens     int
	k          []float32
	v          []float32
}

// RingF32KV is a fixed-capacity K/V ring that overwrites the oldest rows.
type RingF32KV struct {
	dim        int
	capacity   int
	startToken int
	tokens     int
	head       int // physical row index of the logical oldest token
	k          []float32
	v          []float32
}

func NewLinearF32KV(dim int) *LinearF32KV {
	if dim < 0 {
		dim = 0
	}
	return &LinearF32KV{dim: dim}
}

func NewRingF32KV(dim, capacity int) *RingF32KV {
	if dim < 0 {
		dim = 0
	}
	if capacity < 0 {
		capacity = 0
	}
	if dim == 0 || capacity == 0 {
		return &RingF32KV{dim: dim}
	}
	elems, ok := checked.MulInt(dim, capacity)
	if !ok {
		return &RingF32KV{dim: dim}
	}
	return &RingF32KV{
		dim:      dim,
		capacity: capacity,
		k:        make([]float32, elems),
		v:        make([]float32, elems),
	}
}

func (s *LinearF32KV) Append(k, v []float32) error {
	if s == nil {
		return fmt.Errorf("nil LinearF32KV")
	}
	if err := s.validateState(); err != nil {
		return err
	}
	if s.dim <= 0 {
		return fmt.Errorf("linear KV dim=%d must be > 0", s.dim)
	}
	if len(k) != s.dim || len(v) != s.dim {
		return fmt.Errorf("linear KV row len=%d/%d want %d", len(k), len(v), s.dim)
	}
	if _, ok := checked.AddInt(s.tokens, 1); !ok {
		return fmt.Errorf("linear KV token count overflow")
	}
	if _, ok := checked.AddInt(len(s.k), s.dim); !ok {
		return fmt.Errorf("linear KV K length overflow")
	}
	if _, ok := checked.AddInt(len(s.v), s.dim); !ok {
		return fmt.Errorf("linear KV V length overflow")
	}
	s.k = append(s.k, k...)
	s.v = append(s.v, v...)
	s.tokens++
	return nil
}

func (s *LinearF32KV) Tokens() int {
	if s == nil {
		return 0
	}
	return s.tokens
}

func (s *LinearF32KV) Dim() int {
	if s == nil {
		return 0
	}
	return s.dim
}

func (s *LinearF32KV) Capacity() int { return 0 }

func (s *LinearF32KV) View() F32KVView {
	if s == nil {
		return F32KVView{}
	}
	view := F32KVView{StartToken: s.startToken}
	if err := s.validateState(); err != nil || s.tokens == 0 {
		return view
	}
	view.FirstK = s.k
	view.FirstV = s.v
	return view
}

func (s *LinearF32KV) Materialize() (k, v []float32, startToken int) {
	if s == nil {
		return nil, nil, 0
	}
	return materializeView(s.View())
}

func (s *LinearF32KV) Reset() {
	if s == nil {
		return
	}
	s.k = s.k[:0]
	s.v = s.v[:0]
	s.tokens = 0
	s.startToken = 0
}

func (s *LinearF32KV) Bytes() int64 {
	if s == nil {
		return 0
	}
	return logicalF32KVBytes(s.tokens, s.dim)
}

func (s *LinearF32KV) Checkpoint() F32KVCheckpoint {
	if s == nil {
		return F32KVCheckpoint{}
	}
	k, v, startToken := s.Materialize()
	if s.tokens > 0 && (k == nil || v == nil) {
		return F32KVCheckpoint{}
	}
	return F32KVCheckpoint{
		valid:      true,
		dim:        s.dim,
		capacity:   0,
		startToken: startToken,
		tokens:     s.tokens,
		k:          k,
		v:          v,
	}
}

func (s *LinearF32KV) Restore(cp F32KVCheckpoint) error {
	if s == nil {
		return fmt.Errorf("nil LinearF32KV")
	}
	if _, err := validateF32KVCheckpoint(cp); err != nil {
		return err
	}
	if s.dim != cp.dim {
		return fmt.Errorf("linear KV dim=%d checkpoint dim=%d", s.dim, cp.dim)
	}
	s.k = append(s.k[:0], cp.k...)
	s.v = append(s.v[:0], cp.v...)
	s.tokens = cp.tokens
	s.startToken = cp.startToken
	return nil
}

func (s *LinearF32KV) validateState() error {
	if s.dim < 0 || s.tokens < 0 || s.startToken < 0 {
		return fmt.Errorf("linear KV malformed state")
	}
	need, ok := checked.MulInt(s.tokens, s.dim)
	if !ok {
		return fmt.Errorf("linear KV state overflow")
	}
	if len(s.k) != need || len(s.v) != need {
		return fmt.Errorf("linear KV malformed K/V len=%d/%d want %d", len(s.k), len(s.v), need)
	}
	return nil
}

func (s *RingF32KV) Append(k, v []float32) error {
	if s == nil {
		return fmt.Errorf("nil RingF32KV")
	}
	if err := s.validateState(); err != nil {
		return err
	}
	if s.dim <= 0 {
		return fmt.Errorf("ring KV dim=%d must be > 0", s.dim)
	}
	if s.capacity <= 0 {
		return fmt.Errorf("ring KV capacity=%d must be > 0", s.capacity)
	}
	if len(k) != s.dim || len(v) != s.dim {
		return fmt.Errorf("ring KV row len=%d/%d want %d", len(k), len(v), s.dim)
	}
	if s.tokens < s.capacity {
		sum, ok := checked.AddInt(s.head, s.tokens)
		if !ok {
			return fmt.Errorf("ring KV write index overflow")
		}
		idx := sum
		if idx >= s.capacity {
			idx -= s.capacity
		}
		if err := s.copyRow(idx, k, v); err != nil {
			return err
		}
		s.tokens++
		return nil
	}
	if _, ok := checked.AddInt(s.startToken, 1); !ok {
		return fmt.Errorf("ring KV startToken overflow")
	}
	if err := s.copyRow(s.head, k, v); err != nil {
		return err
	}
	s.head++
	if s.head == s.capacity {
		s.head = 0
	}
	s.startToken++
	return nil
}

func (s *RingF32KV) Tokens() int {
	if s == nil {
		return 0
	}
	return s.tokens
}

func (s *RingF32KV) Dim() int {
	if s == nil {
		return 0
	}
	return s.dim
}

func (s *RingF32KV) Capacity() int {
	if s == nil {
		return 0
	}
	return s.capacity
}

func (s *RingF32KV) View() F32KVView {
	if s == nil {
		return F32KVView{}
	}
	view := F32KVView{StartToken: s.startToken}
	if err := s.validateState(); err != nil || s.tokens == 0 {
		return view
	}
	sum, ok := checked.AddInt(s.head, s.tokens)
	if !ok {
		return view
	}
	if sum <= s.capacity {
		view.FirstK, ok = rowSlice(s.k, s.dim, s.head, s.tokens)
		if !ok {
			return F32KVView{StartToken: s.startToken}
		}
		view.FirstV, ok = rowSlice(s.v, s.dim, s.head, s.tokens)
		if !ok {
			return F32KVView{StartToken: s.startToken}
		}
		return view
	}
	firstRows := s.capacity - s.head
	secondRows := s.tokens - firstRows
	view.FirstK, ok = rowSlice(s.k, s.dim, s.head, firstRows)
	if !ok {
		return F32KVView{StartToken: s.startToken}
	}
	view.FirstV, ok = rowSlice(s.v, s.dim, s.head, firstRows)
	if !ok {
		return F32KVView{StartToken: s.startToken}
	}
	view.SecondK, ok = rowSlice(s.k, s.dim, 0, secondRows)
	if !ok {
		return F32KVView{StartToken: s.startToken}
	}
	view.SecondV, ok = rowSlice(s.v, s.dim, 0, secondRows)
	if !ok {
		return F32KVView{StartToken: s.startToken}
	}
	return view
}

func (s *RingF32KV) Materialize() (k, v []float32, startToken int) {
	if s == nil {
		return nil, nil, 0
	}
	return materializeView(s.View())
}

func (s *RingF32KV) Reset() {
	if s == nil {
		return
	}
	s.tokens = 0
	s.head = 0
	s.startToken = 0
}

func (s *RingF32KV) Bytes() int64 {
	if s == nil {
		return 0
	}
	return logicalF32KVBytes(s.tokens, s.dim)
}

func (s *RingF32KV) Checkpoint() F32KVCheckpoint {
	if s == nil {
		return F32KVCheckpoint{}
	}
	k, v, startToken := s.Materialize()
	if s.tokens > 0 && (k == nil || v == nil) {
		return F32KVCheckpoint{}
	}
	return F32KVCheckpoint{
		valid:      true,
		dim:        s.dim,
		capacity:   s.capacity,
		startToken: startToken,
		tokens:     s.tokens,
		k:          k,
		v:          v,
	}
}

func (s *RingF32KV) Restore(cp F32KVCheckpoint) error {
	if s == nil {
		return fmt.Errorf("nil RingF32KV")
	}
	need, err := validateF32KVCheckpoint(cp)
	if err != nil {
		return err
	}
	if s.dim != cp.dim {
		return fmt.Errorf("ring KV dim=%d checkpoint dim=%d", s.dim, cp.dim)
	}
	if cp.tokens > s.capacity {
		return fmt.Errorf("ring KV checkpoint tokens=%d exceed capacity=%d", cp.tokens, s.capacity)
	}
	if err := s.validateStorage(); err != nil {
		return err
	}
	if need > 0 {
		copy(s.k[:need], cp.k)
		copy(s.v[:need], cp.v)
	}
	s.tokens = cp.tokens
	s.head = 0
	s.startToken = cp.startToken
	return nil
}

func (s *RingF32KV) validateState() error {
	if s.dim < 0 || s.capacity < 0 || s.tokens < 0 || s.startToken < 0 || s.head < 0 {
		return fmt.Errorf("ring KV malformed state")
	}
	if s.capacity == 0 {
		if s.tokens != 0 || s.head != 0 || len(s.k) != 0 || len(s.v) != 0 {
			return fmt.Errorf("ring KV malformed zero-capacity state")
		}
		return nil
	}
	if s.tokens > s.capacity || s.head >= s.capacity {
		return fmt.Errorf("ring KV malformed token/head state")
	}
	return s.validateStorage()
}

func (s *RingF32KV) validateStorage() error {
	expected, ok := checked.MulInt(s.capacity, s.dim)
	if !ok {
		return fmt.Errorf("ring KV storage size overflow")
	}
	if len(s.k) < expected || len(s.v) < expected {
		return fmt.Errorf("ring KV malformed storage len=%d/%d want >= %d", len(s.k), len(s.v), expected)
	}
	return nil
}

func (s *RingF32KV) copyRow(row int, k, v []float32) error {
	ks, ok := rowSlice(s.k, s.dim, row, 1)
	if !ok {
		return fmt.Errorf("ring KV row %d outside K storage", row)
	}
	vs, ok := rowSlice(s.v, s.dim, row, 1)
	if !ok {
		return fmt.Errorf("ring KV row %d outside V storage", row)
	}
	copy(ks, k)
	copy(vs, v)
	return nil
}

func validateF32KVCheckpoint(cp F32KVCheckpoint) (int, error) {
	if !cp.valid {
		return 0, fmt.Errorf("invalid checkpoint")
	}
	if cp.dim < 0 || cp.capacity < 0 || cp.tokens < 0 || cp.startToken < 0 {
		return 0, fmt.Errorf("checkpoint has negative fields")
	}
	if cp.tokens > 0 && cp.dim <= 0 {
		return 0, fmt.Errorf("checkpoint tokens=%d require dim > 0", cp.tokens)
	}
	if cp.capacity != 0 && cp.tokens > cp.capacity {
		return 0, fmt.Errorf("checkpoint tokens=%d exceed capacity=%d", cp.tokens, cp.capacity)
	}
	need, ok := checked.MulInt(cp.tokens, cp.dim)
	if !ok {
		return 0, fmt.Errorf("checkpoint tokens=%d dim=%d overflow", cp.tokens, cp.dim)
	}
	if len(cp.k) != need || len(cp.v) != need {
		return 0, fmt.Errorf("checkpoint K/V len=%d/%d want %d", len(cp.k), len(cp.v), need)
	}
	return need, nil
}

func materializeView(view F32KVView) (k, v []float32, startToken int) {
	startToken = view.StartToken
	if len(view.FirstK) != len(view.FirstV) || len(view.SecondK) != len(view.SecondV) {
		return nil, nil, startToken
	}
	total, ok := checked.AddInt(len(view.FirstK), len(view.SecondK))
	if !ok {
		return nil, nil, startToken
	}
	if total == 0 {
		return nil, nil, startToken
	}
	k = make([]float32, total)
	v = make([]float32, total)
	n := copy(k, view.FirstK)
	copy(k[n:], view.SecondK)
	n = copy(v, view.FirstV)
	copy(v[n:], view.SecondV)
	return k, v, startToken
}

func rowSlice(buf []float32, dim, startRow, rows int) ([]float32, bool) {
	if dim < 0 || startRow < 0 || rows < 0 {
		return nil, false
	}
	if rows == 0 {
		return nil, true
	}
	start, ok := checked.MulInt(startRow, dim)
	if !ok {
		return nil, false
	}
	count, ok := checked.MulInt(rows, dim)
	if !ok {
		return nil, false
	}
	end, ok := checked.AddInt(start, count)
	if !ok || end > len(buf) {
		return nil, false
	}
	return buf[start:end], true
}

func logicalF32KVBytes(tokens, dim int) int64 {
	if tokens < 0 || dim < 0 {
		return 0
	}
	elems, ok := checked.MulInt(tokens, dim)
	if !ok {
		return checked.MaxInt64()
	}
	return checked.SaturatingMulInt64(int64(elems), 8)
}
