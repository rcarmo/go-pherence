package k3

import "github.com/rcarmo/go-pherence/backends/spacemit/ime2"

// packBufs holds pre-allocated buffers for activation quantize+pack.
type packBufs struct {
	xI8    []int8
	bc     []int8
	packed []int8
}

func newPackBufs(maxK int) *packBufs {
	Kp := ((maxK + 7) / 8) * 8
	return &packBufs{
		xI8:    make([]int8, Kp),
		bc:     make([]int8, 4*Kp),
		packed: make([]int8, 4*Kp),
	}
}

// quantizeAndPackInto quantizes and packs without allocation.
func quantizeAndPackInto(act []float32, Kp int, b *packBufs) ([]int8, float32) {
	xI8 := b.xI8[:Kp]
	for i := Kp - 1; i >= len(act); i-- {
		xI8[i] = 0
	}
	maxAbs := ime2.FindMaxAbsRVV(act)

	if maxAbs == 0 {
		for i := range xI8 {
			xI8[i] = 0
		}
		pk := b.packed[:4*Kp]
		for i := range pk {
			pk[i] = 0
		}
		return pk, 0
	}
	s := float32(127.0) / maxAbs
	ime2.QuantizeF32ToI8RVV(act, s, xI8[:len(act)])
	// Fused broadcast-pack (zero-alloc)
	pk := b.packed[:4*Kp]
	for ki := 0; ki < Kp; ki += 8 {
		tileBase := (ki / 8) * 32
		copy(pk[tileBase:tileBase+8], xI8[ki:ki+8])
		copy(pk[tileBase+8:tileBase+16], xI8[ki:ki+8])
		copy(pk[tileBase+16:tileBase+24], xI8[ki:ki+8])
		copy(pk[tileBase+24:tileBase+32], xI8[ki:ki+8])
	}
	return pk, maxAbs / 127.0
}

// Keep the allocating version for compatibility
func quantizeAndPackLocal(act []float32, Kp int) ([]int8, float32) {
	b := newPackBufs(Kp)
	return quantizeAndPackInto(act, Kp, b)
}

var _ = ime2.PackTiles // keep import
