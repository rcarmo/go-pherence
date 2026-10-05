package training

import (
	"fmt"
	"math"
)

// EOSAdmissionShape describes a scalar conservative absolute-dot check over all
// rows, including padding. Inputs are hidden[rows*width], weight[width], bias[1].
// Output[1] is zero only when the legacy F64 EOS projection bound is certainly
// satisfied. This admission has no gradient and may reject near-boundary inputs.
func EOSAdmissionShape(rows, width int) (hidden int, budget uint32, err error) {
	if rows < 1 || rows > 1024 || width < 1 || width > 16384 {
		return 0, 0, fmt.Errorf("EOS admission shape")
	}
	// One integer unit is 2^95. Reserve a unit for possible positive rounding
	// error in the host's <=16385 exact-product F64 additions. Bitwise product
	// bounds plus ceil-to-unit rounding avoid float arithmetic and FTZ entirely.
	limit := float64(math.MaxFloat32) * .125 / float64(rows)
	budget = uint32(math.Floor(math.Ldexp(limit, -95))) - 1
	return rows * width, budget, nil
}

// eosAdmissionUnits returns an upward bound in 2^95 units, rejecting a term
// larger than remaining. Decode F32 bits including subnormals; integer mantissa
// bounds use 12 significant bits each and a 24-bit product, never overflow.
func eosAdmissionUnits(a, b float32, remaining uint32) (uint32, bool) {
	ab, bb := math.Float32bits(a)&0x7fffffff, math.Float32bits(b)&0x7fffffff
	ae, be := ab>>23, bb>>23
	if ae == 255 || be == 255 {
		return 0, false
	}
	am, bm := ab&0x7fffff, bb&0x7fffff
	if ae == 0 {
		ae = 1
	} else {
		am |= 0x800000
	}
	if be == 0 {
		be = 1
	} else {
		bm |= 0x800000
	}
	am = (am + 4095) >> 12
	bm = (bm + 4095) >> 12
	p := am * bm
	if p == 0 {
		return 0, true
	}
	sum := ae + be
	if sum >= 371 {
		shift := sum - 371
		if shift >= 31 {
			return 0, false
		}
		if p > remaining>>shift {
			return 0, false
		}
		return p << shift, true
	}
	shift := 371 - sum
	if shift >= 32 {
		if remaining == 0 {
			return 0, false
		}
		return 1, true
	}
	units := p >> shift
	mask := uint32(0xffffffff) >> (32 - shift)
	if p&mask != 0 {
		units++
	}
	return units, units <= remaining
}

// EOSAdmissionFlag is the offline reference for integer admission, not a native
// numerical oracle. Independent F64 tests verify soundness and rejection cost.
func EOSAdmissionFlag(hidden, weight []float32, bias float32, rows, width int) (float32, error) {
	n, budget, err := EOSAdmissionShape(rows, width)
	if err != nil || len(hidden) != n || len(weight) != width {
		return 0, fmt.Errorf("EOS admission extent")
	}
	for row := 0; row < rows; row++ {
		units, ok := eosAdmissionUnits(bias, 1, budget)
		if !ok {
			return 1, nil
		}
		remaining := budget - units
		for col, w := range weight {
			units, ok = eosAdmissionUnits(hidden[row*width+col], w, remaining)
			if !ok {
				return 1, nil
			}
			remaining -= units
		}
	}
	return 0, nil
}
