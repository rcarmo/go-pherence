package kv

import (
	"fmt"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// FP16KVView exposes an experimental logical oldest-to-newest FP16 K/V window
// without forcing a copy. Ring-backed stores can return two segments when the
// logical view wraps.
type FP16KVView struct {
	FirstK     []uint16
	FirstV     []uint16
	SecondK    []uint16
	SecondV    []uint16
	StartToken int
}

// RingFP16KV is an experimental fixed-capacity paired K/V ring backed by FP16
// bits. Append accepts float32 rows and converts them to half precision.
type RingFP16KV struct {
	dim        int
	capacity   int
	startToken int
	tokens     int
	head       int // physical row index of the logical oldest token
	k          []uint16
	v          []uint16
}

func NewRingFP16KV(dim, capacity int) *RingFP16KV {
	if dim < 0 {
		dim = 0
	}
	if capacity < 0 {
		capacity = 0
	}
	if dim == 0 || capacity == 0 {
		return &RingFP16KV{dim: dim, capacity: capacity}
	}
	elems, ok := checked.MulInt(dim, capacity)
	if !ok {
		return &RingFP16KV{dim: dim}
	}
	return &RingFP16KV{
		dim:      dim,
		capacity: capacity,
		k:        make([]uint16, elems),
		v:        make([]uint16, elems),
	}
}

func (s *RingFP16KV) Append(k, v []float32) error {
	if s == nil {
		return fmt.Errorf("nil RingFP16KV")
	}
	if err := s.validateState(); err != nil {
		return err
	}
	if s.dim <= 0 {
		return fmt.Errorf("ring FP16 KV dim=%d must be > 0", s.dim)
	}
	if s.capacity <= 0 {
		return fmt.Errorf("ring FP16 KV capacity=%d must be > 0", s.capacity)
	}
	if len(k) != s.dim || len(v) != s.dim {
		return fmt.Errorf("ring FP16 KV row len=%d/%d want %d", len(k), len(v), s.dim)
	}
	if s.tokens < s.capacity {
		sum, ok := checked.AddInt(s.head, s.tokens)
		if !ok {
			return fmt.Errorf("ring FP16 KV write index overflow")
		}
		idx := sum
		if idx >= s.capacity {
			idx -= s.capacity
		}
		if err := s.encodeRow(idx, k, v); err != nil {
			return err
		}
		s.tokens++
		return nil
	}
	if _, ok := checked.AddInt(s.startToken, 1); !ok {
		return fmt.Errorf("ring FP16 KV startToken overflow")
	}
	if err := s.encodeRow(s.head, k, v); err != nil {
		return err
	}
	s.head++
	if s.head == s.capacity {
		s.head = 0
	}
	s.startToken++
	return nil
}

func (s *RingFP16KV) Tokens() int {
	if s == nil {
		return 0
	}
	return s.tokens
}

func (s *RingFP16KV) Dim() int {
	if s == nil {
		return 0
	}
	return s.dim
}

func (s *RingFP16KV) Capacity() int {
	if s == nil {
		return 0
	}
	return s.capacity
}

func (s *RingFP16KV) View() FP16KVView {
	if s == nil {
		return FP16KVView{}
	}
	view := FP16KVView{StartToken: s.startToken}
	if err := s.validateState(); err != nil || s.tokens == 0 {
		return view
	}
	sum, ok := checked.AddInt(s.head, s.tokens)
	if !ok {
		return view
	}
	if sum <= s.capacity {
		view.FirstK, ok = rowSliceU16(s.k, s.dim, s.head, s.tokens)
		if !ok {
			return FP16KVView{StartToken: s.startToken}
		}
		view.FirstV, ok = rowSliceU16(s.v, s.dim, s.head, s.tokens)
		if !ok {
			return FP16KVView{StartToken: s.startToken}
		}
		return view
	}
	firstRows := s.capacity - s.head
	secondRows := s.tokens - firstRows
	view.FirstK, ok = rowSliceU16(s.k, s.dim, s.head, firstRows)
	if !ok {
		return FP16KVView{StartToken: s.startToken}
	}
	view.FirstV, ok = rowSliceU16(s.v, s.dim, s.head, firstRows)
	if !ok {
		return FP16KVView{StartToken: s.startToken}
	}
	view.SecondK, ok = rowSliceU16(s.k, s.dim, 0, secondRows)
	if !ok {
		return FP16KVView{StartToken: s.startToken}
	}
	view.SecondV, ok = rowSliceU16(s.v, s.dim, 0, secondRows)
	if !ok {
		return FP16KVView{StartToken: s.startToken}
	}
	return view
}

func (s *RingFP16KV) MaterializeF32() (k, v []float32, startToken int) {
	if s == nil {
		return nil, nil, 0
	}
	return materializeFP16ViewF32(s.View())
}

func (s *RingFP16KV) Reset() {
	if s == nil {
		return
	}
	s.tokens = 0
	s.head = 0
	s.startToken = 0
}

// Bytes returns the current logical FP16 K/V payload in bytes.
func (s *RingFP16KV) Bytes() int64 {
	return s.LogicalBytes()
}

// LogicalBytes returns the current logical FP16 K/V payload in bytes.
func (s *RingFP16KV) LogicalBytes() int64 {
	if s == nil {
		return 0
	}
	return logicalFP16KVBytes(s.tokens, s.dim)
}

// PhysicalBytes returns the backing FP16 K/V storage reserved by the ring.
func (s *RingFP16KV) PhysicalBytes() int64 {
	if s == nil {
		return 0
	}
	return physicalFP16KVBytes(s.capacity, s.dim)
}

func (s *RingFP16KV) validateState() error {
	if s.dim < 0 || s.capacity < 0 || s.tokens < 0 || s.startToken < 0 || s.head < 0 {
		return fmt.Errorf("ring FP16 KV malformed state")
	}
	if s.capacity == 0 {
		if s.tokens != 0 || s.head != 0 || len(s.k) != 0 || len(s.v) != 0 {
			return fmt.Errorf("ring FP16 KV malformed zero-capacity state")
		}
		return nil
	}
	if s.tokens > s.capacity || s.head >= s.capacity {
		return fmt.Errorf("ring FP16 KV malformed token/head state")
	}
	return s.validateStorage()
}

func (s *RingFP16KV) validateStorage() error {
	expected, ok := checked.MulInt(s.capacity, s.dim)
	if !ok {
		return fmt.Errorf("ring FP16 KV storage size overflow")
	}
	if len(s.k) != expected || len(s.v) != expected {
		return fmt.Errorf("ring FP16 KV malformed storage len=%d/%d want %d", len(s.k), len(s.v), expected)
	}
	return nil
}

func (s *RingFP16KV) encodeRow(row int, k, v []float32) error {
	ks, ok := rowSliceU16(s.k, s.dim, row, 1)
	if !ok {
		return fmt.Errorf("ring FP16 KV row %d outside K storage", row)
	}
	vs, ok := rowSliceU16(s.v, s.dim, row, 1)
	if !ok {
		return fmt.Errorf("ring FP16 KV row %d outside V storage", row)
	}
	for i, x := range k {
		ks[i] = half.F32ToF16(x)
	}
	for i, x := range v {
		vs[i] = half.F32ToF16(x)
	}
	return nil
}

func materializeFP16ViewF32(view FP16KVView) (k, v []float32, startToken int) {
	startToken = view.StartToken
	if len(view.FirstK) != len(view.FirstV) || len(view.SecondK) != len(view.SecondV) {
		return nil, nil, startToken
	}
	total, ok := checked.AddInt(len(view.FirstK), len(view.SecondK))
	if !ok || total == 0 {
		return nil, nil, startToken
	}
	k = make([]float32, total)
	v = make([]float32, total)
	n := decodeF16Slice(k, view.FirstK)
	decodeF16Slice(k[n:], view.SecondK)
	n = decodeF16Slice(v, view.FirstV)
	decodeF16Slice(v[n:], view.SecondV)
	return k, v, startToken
}

func decodeF16Slice(dst []float32, src []uint16) int {
	for i, bits := range src {
		dst[i] = half.F16ToF32(bits)
	}
	return len(src)
}

func rowSliceU16(buf []uint16, dim, startRow, rows int) ([]uint16, bool) {
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

func logicalFP16KVBytes(tokens, dim int) int64 {
	if tokens < 0 || dim < 0 {
		return 0
	}
	elems, ok := checked.MulInt(tokens, dim)
	if !ok {
		return checked.MaxInt64()
	}
	return checked.SaturatingMulInt64(int64(elems), 4)
}

func physicalFP16KVBytes(capacity, dim int) int64 {
	if capacity < 0 || dim < 0 {
		return 0
	}
	elems, ok := checked.MulInt(capacity, dim)
	if !ok {
		return checked.MaxInt64()
	}
	return checked.SaturatingMulInt64(int64(elems), 4)
}
