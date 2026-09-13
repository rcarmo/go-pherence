package omnivoice

import (
	"context"
	"fmt"
)

// codecScratch is private to a decoder. Three live activation slots suffice:
// residual, current operand, result. A fourth allows the RVQ accumulator.
type codecScratch struct {
	slots                            [4][]float32
	used                             [4]bool
	packed, result, input, projected []float32
	frames                           int
}

func (d *CodecDecoder) Prepare(frames int) error {
	if frames < 1 || frames > 250 {
		return fmt.Errorf("omnivoice: codec frame limit 1..250")
	}
	maxElements := 32 * 960 * frames
	if d.scratch != nil && d.scratch.frames == frames {
		return nil
	}
	s := &codecScratch{frames: frames, packed: make([]float32, 512*7*64), result: make([]float32, 1024*64), input: make([]float32, 32*1024), projected: make([]float32, 32*512*16)}
	for i := range s.slots {
		s.slots[i] = make([]float32, maxElements)
	}
	d.scratch = s
	return nil
}
func (d *CodecDecoder) buffer(n int) []float32 {
	if d.scratch == nil {
		return make([]float32, n)
	}
	s := d.scratch
	for i := range s.slots {
		if !s.used[i] && n <= len(s.slots[i]) {
			s.used[i] = true
			out := s.slots[i][:n]
			clear(out)
			return out
		}
	}
	panic("omnivoice: internal codec workspace exhausted")
}
func (d *CodecDecoder) release(x signal) {
	if d.scratch == nil || len(x.data) == 0 {
		return
	}
	for i := range d.scratch.slots {
		if &d.scratch.slots[i][0] == &x.data[0] {
			d.scratch.used[i] = false
			return
		}
	}
}

// DecodeInto uses caller output plus prepared scratch. Output has frames*960
// samples for the supported codec. Decode remains the allocating wrapper.
func (d *CodecDecoder) DecodeInto(ctx context.Context, dst []float32, codes []int, books, frames int) error {
	if d.scratch == nil || d.scratch.frames != frames || len(dst) != frames*960 {
		return fmt.Errorf("omnivoice: prepare codec and provide exact output size")
	}
	d.scratch.used = [4]bool{}
	out, err := d.decode(ctx, codes, books, frames)
	if err != nil {
		return err
	}
	copy(dst, out)
	d.scratch.used = [4]bool{}
	return nil
}
