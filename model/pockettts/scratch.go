package pockettts

import "fmt"

// Scratch is caller-owned monotonic F32 workspace. A generation session resets
// it at frame boundaries after copying the emitted PCM chunk. It is not shared
// across requests and performs no allocation while capacity is sufficient.
type Scratch struct {
	values []float32
	used   int
}

func NewScratch(floatCapacity int) (*Scratch, error) {
	if floatCapacity <= 0 || floatCapacity > int(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("invalid Pocket TTS scratch capacity %d", floatCapacity)
	}
	return &Scratch{values: make([]float32, floatCapacity)}, nil
}

func (s *Scratch) Reset() {
	if s != nil {
		s.used = 0
	}
}
func (s *Scratch) Mark() int {
	if s == nil {
		return 0
	}
	return s.used
}
func (s *Scratch) ResetTo(mark int) error {
	if s == nil || mark < 0 || mark > s.used {
		return fmt.Errorf("invalid Pocket TTS scratch mark %d", mark)
	}
	s.used = mark
	return nil
}
func (s *Scratch) Used() int {
	if s == nil {
		return 0
	}
	return s.used
}
func (s *Scratch) Capacity() int {
	if s == nil {
		return 0
	}
	return len(s.values)
}
func (s *Scratch) Take(n int) ([]float32, error) {
	if s == nil || n < 0 || n > len(s.values)-s.used {
		return nil, fmt.Errorf("Pocket TTS scratch exhausted used=%d request=%d capacity=%d", s.Used(), n, s.Capacity())
	}
	out := s.values[s.used : s.used+n]
	clear(out)
	s.used += n
	return out, nil
}
