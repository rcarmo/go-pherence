package main

import "github.com/rcarmo/go-pherence/backends/spacemit/ime2"

// quantizeAndPackLocal quantizes F32 act to INT8 and broadcast-packs for vmadot.
func quantizeAndPackLocal(act []float32, Kp int) ([]int8, float32) {
	xI8 := make([]int8, Kp)
	var maxAbs float32
	for _, v := range act {
		a := v
		if a < 0 { a = -a }
		if a > maxAbs { maxAbs = a }
	}
	if maxAbs == 0 {
		return make([]int8, 4*Kp), 0
	}
	s := float32(127.0) / maxAbs
	for i := 0; i < len(act) && i < Kp; i++ {
		v := act[i] * s
		if v > 127 { v = 127 } else if v < -128 { v = -128 }
		xI8[i] = int8(v)
	}
	bc := make([]int8, 4*Kp)
	copy(bc[0:Kp], xI8)
	copy(bc[Kp:2*Kp], xI8)
	copy(bc[2*Kp:3*Kp], xI8)
	copy(bc[3*Kp:4*Kp], xI8)
	return ime2.PackTiles(bc, 4, Kp), maxAbs / 127.0
}
