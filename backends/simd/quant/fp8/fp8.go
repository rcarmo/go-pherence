// Package fp8 implements weight-only float8 (E4M3FN) linear matrices as used by
// the Ideogram 4 FP8 DiT checkpoints. Each linear stores its weight as one
// E4M3 byte per element plus a separate `.weight_scale` tensor. Two scale
// granularities are supported:
//
//   - per-tensor: a single scalar scale applied to every element
//   - per-output-row (per-channel): one scale per output row
//
// Dequantized value = DecodeE4M3(byte) * scale[row?].
package fp8

import (
	"fmt"
	"math"
)

// e4m3LUT maps every possible E4M3FN byte to its float32 value, so decode is a
// branch-free table lookup in hot GEMV loops.
var e4m3LUT = func() [256]float32 {
	var t [256]float32
	for i := 0; i < 256; i++ {
		t[i] = decodeE4M3Slow(byte(i))
	}
	return t
}()

// DecodeE4M3 decodes a finite-only float8 E4M3FN byte (bias 7, subnormals at
// exponent 0, all-ones exponent+mantissa reserved as NaN, no infinities).
func DecodeE4M3(code byte) float32 { return e4m3LUT[code] }

func decodeE4M3Slow(code byte) float32 {
	sign := code & 0x80
	exp := (code >> 3) & 0x0f
	mant := code & 0x07
	var v float32
	switch {
	case exp == 0:
		if mant == 0 {
			v = 0
		} else {
			v = float32(mant) / 8 * float32(math.Ldexp(1, -6))
		}
	case exp == 0x0f && mant == 0x07:
		v = float32(math.NaN())
	default:
		v = (1 + float32(mant)/8) * float32(math.Ldexp(1, int(exp)-7))
	}
	if sign != 0 {
		return -v
	}
	return v
}

// Linear holds a row-major [OutDim, InDim] E4M3 weight plus its scale tensor.
// Scale is either length 1 (per-tensor) or length OutDim (per-row).
type Linear struct {
	OutDim int
	InDim  int
	Weight []byte    // OutDim*InDim E4M3 bytes, row-major
	Scale  []float32 // 1 (per-tensor) or OutDim (per-row)
	Bias   []float32 // nil or OutDim
}

// Validate checks shape consistency.
func (l Linear) Validate() error {
	if l.OutDim <= 0 || l.InDim <= 0 {
		return fmt.Errorf("fp8 linear invalid dims out=%d in=%d", l.OutDim, l.InDim)
	}
	if want := l.OutDim * l.InDim; len(l.Weight) != want {
		return fmt.Errorf("fp8 linear weight bytes=%d want=%d", len(l.Weight), want)
	}
	if len(l.Scale) != 1 && len(l.Scale) != l.OutDim {
		return fmt.Errorf("fp8 linear scale len=%d want 1 or %d", len(l.Scale), l.OutDim)
	}
	if l.Bias != nil && len(l.Bias) != l.OutDim {
		return fmt.Errorf("fp8 linear bias len=%d want %d", len(l.Bias), l.OutDim)
	}
	return nil
}

func (l Linear) scaleForRow(row int) float32 {
	if len(l.Scale) == 1 {
		return l.Scale[0]
	}
	return l.Scale[row]
}

// DequantRowTo decodes one output row into dst (length InDim).
func (l Linear) DequantRowTo(row int, dst []float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if row < 0 || row >= l.OutDim {
		return fmt.Errorf("fp8 linear row=%d out=%d", row, l.OutDim)
	}
	if len(dst) != l.InDim {
		return fmt.Errorf("fp8 linear dst=%d want=%d", len(dst), l.InDim)
	}
	scale := l.scaleForRow(row)
	base := row * l.InDim
	for j := 0; j < l.InDim; j++ {
		dst[j] = e4m3LUT[l.Weight[base+j]] * scale
	}
	return nil
}

func dotE4M3Scalar(x []float32, w []byte) float32 {
	var acc float32
	for j := 0; j < len(x); j++ {
		acc += e4m3LUT[w[j]] * x[j]
	}
	return acc
}

// GemvTo computes out = W * x, where W is [OutDim, InDim] and x is length
// InDim. out must have length OutDim. Weights are dequantized on the fly so no
// F32 weight expansion is materialized. On amd64 AVX2/FMA hosts the inner
// E4M3 decode+dot loop uses a gather-based SIMD kernel over the 256-entry LUT.
func (l Linear) GemvTo(x []float32, out []float32) error {
	if err := l.Validate(); err != nil {
		return err
	}
	if len(x) != l.InDim {
		return fmt.Errorf("fp8 gemv x=%d want=%d", len(x), l.InDim)
	}
	if len(out) != l.OutDim {
		return fmt.Errorf("fp8 gemv out=%d want=%d", len(out), l.OutDim)
	}
	for r := 0; r < l.OutDim; r++ {
		scale := l.scaleForRow(r)
		base := r * l.InDim
		acc := dotE4M3(x, l.Weight[base:base+l.InDim])
		out[r] = acc * scale
		if l.Bias != nil {
			out[r] += l.Bias[r]
		}
	}
	return nil
}
