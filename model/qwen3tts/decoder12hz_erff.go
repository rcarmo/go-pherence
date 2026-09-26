/* origin: FreeBSD /usr/src/lib/msun/src/s_erff.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

package qwen3tts

import "math"

// Port of libm-0.2.16 erff/expf for the CPU speech decoder.
//
// Each arithmetic helper returns float32 explicitly to keep intermediate
// rounding visible and discourage fused evaluation. decoderExpf/decoderScalbnf are
// direct float32 ports of libm-0.2.16 src/math/expf.rs and src/math/generic/scalbn.rs.

const (
	decoderErffERX  float32 = 8.4506291151e-01
	decoderErffEFX8 float32 = 1.0270333290e+00
	decoderErffPP0  float32 = 1.2837916613e-01
	decoderErffPP1  float32 = -3.2504209876e-01
	decoderErffPP2  float32 = -2.8481749818e-02
	decoderErffPP3  float32 = -5.7702702470e-03
	decoderErffPP4  float32 = -2.3763017452e-05
	decoderErffQQ1  float32 = 3.9791721106e-01
	decoderErffQQ2  float32 = 6.5022252500e-02
	decoderErffQQ3  float32 = 5.0813062117e-03
	decoderErffQQ4  float32 = 1.3249473704e-04
	decoderErffQQ5  float32 = -3.9602282413e-06
	decoderErffPA0  float32 = -2.3621185683e-03
	decoderErffPA1  float32 = 4.1485610604e-01
	decoderErffPA2  float32 = -3.7220788002e-01
	decoderErffPA3  float32 = 3.1834661961e-01
	decoderErffPA4  float32 = -1.1089469492e-01
	decoderErffPA5  float32 = 3.5478305072e-02
	decoderErffPA6  float32 = -2.1663755178e-03
	decoderErffQA1  float32 = 1.0642088205e-01
	decoderErffQA2  float32 = 5.4039794207e-01
	decoderErffQA3  float32 = 7.1828655899e-02
	decoderErffQA4  float32 = 1.2617121637e-01
	decoderErffQA5  float32 = 1.3637083583e-02
	decoderErffQA6  float32 = 1.1984500103e-02
	decoderErffRA0  float32 = -9.8649440333e-03
	decoderErffRA1  float32 = -6.9385856390e-01
	decoderErffRA2  float32 = -1.0558626175e+01
	decoderErffRA3  float32 = -6.2375331879e+01
	decoderErffRA4  float32 = -1.6239666748e+02
	decoderErffRA5  float32 = -1.8460508728e+02
	decoderErffRA6  float32 = -8.1287437439e+01
	decoderErffRA7  float32 = -9.8143291473e+00
	decoderErffSA1  float32 = 1.9651271820e+01
	decoderErffSA2  float32 = 1.3765776062e+02
	decoderErffSA3  float32 = 4.3456588745e+02
	decoderErffSA4  float32 = 6.4538726807e+02
	decoderErffSA5  float32 = 4.2900814819e+02
	decoderErffSA6  float32 = 1.0863500214e+02
	decoderErffSA7  float32 = 6.5702495575e+00
	decoderErffSA8  float32 = -6.0424413532e-02
	decoderErffRB0  float32 = -9.8649431020e-03
	decoderErffRB1  float32 = -7.9928326607e-01
	decoderErffRB2  float32 = -1.7757955551e+01
	decoderErffRB3  float32 = -1.6063638306e+02
	decoderErffRB4  float32 = -6.3756646729e+02
	decoderErffRB5  float32 = -1.0250950928e+03
	decoderErffRB6  float32 = -4.8351919556e+02
	decoderErffSB1  float32 = 3.0338060379e+01
	decoderErffSB2  float32 = 3.2579251099e+02
	decoderErffSB3  float32 = 1.5367296143e+03
	decoderErffSB4  float32 = 3.1998581543e+03
	decoderErffSB5  float32 = 2.5530502930e+03
	decoderErffSB6  float32 = 4.7452853394e+02
	decoderErffSB7  float32 = -2.2440952301e+01

	EXP_HALF_POS float32 = 5.0000000000e-01
	EXP_HALF_NEG float32 = -5.0000000000e-01
	EXP_LN2_HI   float32 = 6.9314575195e-01
	EXP_LN2_LO   float32 = 1.4286067653e-06
	EXP_INV_LN2  float32 = 1.4426950216e+00
	EXP_P1       float32 = 1.6666625440e-01
	EXP_P2       float32 = -2.7667332906e-03
)

func decoderaddf(a, b float32) float32 { return float32(a + b) }
func decodersubf(a, b float32) float32 { return float32(a - b) }
func decodermulf(a, b float32) float32 { return float32(a * b) }
func decoderdivf(a, b float32) float32 { return float32(a / b) }
func decodernegf(x float32) float32    { return float32(-x) }

func decoderabsf(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) & 0x7fffffff)
}

func decoderScalbnf(x float32, n int32) float32 {
	const (
		sigTotalBits int32 = 24
		expMax       int32 = 127
		expMin       int32 = -126
	)

	fExpMax := math.Float32frombits(0x7f000000)
	fExpMin := math.Float32frombits(0x00800000)
	fPowSubnorm := math.Float32frombits(0x4b800000)

	if n > expMax {
		x = decodermulf(x, fExpMax)
		n -= expMax
		if n > expMax {
			x = decodermulf(x, fExpMax)
			n -= expMax
			if n > expMax {
				n = expMax
			}
		}
	} else if n < expMin {
		mul := decodermulf(fExpMin, fPowSubnorm)
		add := int32(102)
		x = decodermulf(x, mul)
		n += add
		if n < expMin {
			x = decodermulf(x, mul)
			n += add
			if n < expMin {
				n = expMin
			}
		}
	}

	scale := math.Float32frombits(uint32(127+n) << 23)
	return decodermulf(x, scale)
}

func decoderExpf(x float32) float32 {
	x1p127 := math.Float32frombits(0x7f000000)
	x1p126 := math.Float32frombits(0x00800000)
	hx := math.Float32bits(x)
	sign := int32(hx >> 31)
	signb := sign != 0
	hx &= 0x7fffffff

	if hx >= 0x42aeac50 {
		if hx > 0x7f800000 {
			return x
		}
		if hx >= 0x42b17218 && !signb {
			x = decodermulf(x, x1p127)
			return x
		}
		if signb {
			_ = decoderdivf(decodernegf(x1p126), x)
			if hx >= 0x42cff1b5 {
				return 0.0
			}
		}
	}

	var (
		k  int32
		hi float32
		lo float32
	)
	if hx > 0x3eb17218 {
		if hx > 0x3f851592 {
			half := EXP_HALF_POS
			if sign != 0 {
				half = EXP_HALF_NEG
			}
			k = int32(decoderaddf(decodermulf(EXP_INV_LN2, x), half))
		} else {
			k = 1 - sign - sign
		}
		kf := float32(k)
		hi = decodersubf(x, decodermulf(kf, EXP_LN2_HI))
		lo = decodermulf(kf, EXP_LN2_LO)
		x = decodersubf(hi, lo)
	} else if hx > 0x39000000 {
		k = 0
		hi = x
		lo = 0.0
	} else {
		_ = decoderaddf(x1p127, x)
		return decoderaddf(1.0, x)
	}

	xx := decodermulf(x, x)
	c := decodersubf(x, decodermulf(xx, decoderaddf(EXP_P1, decodermulf(xx, EXP_P2))))
	t := decoderaddf(decodersubf(decoderdivf(decodermulf(x, c), decodersubf(2.0, c)), lo), hi)
	y := decoderaddf(1.0, t)
	if k == 0 {
		return y
	}
	return decoderScalbnf(y, k)
}

func decoderErfc1(x float32) float32 {
	s := decodersubf(decoderabsf(x), 1.0)
	p := decoderaddf(decoderErffPA0, decodermulf(s, decoderaddf(decoderErffPA1, decodermulf(s, decoderaddf(decoderErffPA2, decodermulf(s, decoderaddf(decoderErffPA3, decodermulf(s, decoderaddf(decoderErffPA4, decodermulf(s, decoderaddf(decoderErffPA5, decodermulf(s, decoderErffPA6))))))))))))
	q := decoderaddf(1.0, decodermulf(s, decoderaddf(decoderErffQA1, decodermulf(s, decoderaddf(decoderErffQA2, decodermulf(s, decoderaddf(decoderErffQA3, decodermulf(s, decoderaddf(decoderErffQA4, decodermulf(s, decoderaddf(decoderErffQA5, decodermulf(s, decoderErffQA6))))))))))))
	return decodersubf(decodersubf(1.0, decoderErffERX), decoderdivf(p, q))
}

func decoderErfc2(ix uint32, x float32) float32 {
	if ix < 0x3fa00000 {
		return decoderErfc1(x)
	}

	x = decoderabsf(x)
	s := decoderdivf(1.0, decodermulf(x, x))
	var r, bigS float32
	if ix < 0x4036db6d {
		r = decoderaddf(decoderErffRA0, decodermulf(s, decoderaddf(decoderErffRA1, decodermulf(s, decoderaddf(decoderErffRA2, decodermulf(s, decoderaddf(decoderErffRA3, decodermulf(s, decoderaddf(decoderErffRA4, decodermulf(s, decoderaddf(decoderErffRA5, decodermulf(s, decoderaddf(decoderErffRA6, decodermulf(s, decoderErffRA7))))))))))))))
		bigS = decoderaddf(1.0, decodermulf(s, decoderaddf(decoderErffSA1, decodermulf(s, decoderaddf(decoderErffSA2, decodermulf(s, decoderaddf(decoderErffSA3, decodermulf(s, decoderaddf(decoderErffSA4, decodermulf(s, decoderaddf(decoderErffSA5, decodermulf(s, decoderaddf(decoderErffSA6, decodermulf(s, decoderaddf(decoderErffSA7, decodermulf(s, decoderErffSA8))))))))))))))))
	} else {
		r = decoderaddf(decoderErffRB0, decodermulf(s, decoderaddf(decoderErffRB1, decodermulf(s, decoderaddf(decoderErffRB2, decodermulf(s, decoderaddf(decoderErffRB3, decodermulf(s, decoderaddf(decoderErffRB4, decodermulf(s, decoderaddf(decoderErffRB5, decodermulf(s, decoderErffRB6))))))))))))
		bigS = decoderaddf(1.0, decodermulf(s, decoderaddf(decoderErffSB1, decodermulf(s, decoderaddf(decoderErffSB2, decodermulf(s, decoderaddf(decoderErffSB3, decodermulf(s, decoderaddf(decoderErffSB4, decodermulf(s, decoderaddf(decoderErffSB5, decodermulf(s, decoderaddf(decoderErffSB6, decodermulf(s, decoderErffSB7))))))))))))))
	}
	ix = math.Float32bits(x)
	z := math.Float32frombits(ix & 0xffffe000)

	left := decoderExpf(decodersubf(decodermulf(decodernegf(z), z), 0.5625))
	right := decoderExpf(decoderaddf(decodermulf(decodersubf(z, x), decoderaddf(z, x)), decoderdivf(r, bigS)))
	return decoderdivf(decodermulf(left, right), x)
}

func decoderErff(x float32) float32 {
	ix := math.Float32bits(x)
	sign := ix >> 31
	ix &= 0x7fffffff
	if ix >= 0x7f800000 {
		return decoderaddf(decodersubf(1.0, decodermulf(2.0, float32(sign))), decoderdivf(1.0, x))
	}
	if ix < 0x3f580000 {
		if ix < 0x31800000 {
			return decodermulf(0.125, decoderaddf(decodermulf(8.0, x), decodermulf(decoderErffEFX8, x)))
		}
		z := decodermulf(x, x)
		r := decoderaddf(decoderErffPP0, decodermulf(z, decoderaddf(decoderErffPP1, decodermulf(z, decoderaddf(decoderErffPP2, decodermulf(z, decoderaddf(decoderErffPP3, decodermulf(z, decoderErffPP4))))))))
		s := decoderaddf(1.0, decodermulf(z, decoderaddf(decoderErffQQ1, decodermulf(z, decoderaddf(decoderErffQQ2, decodermulf(z, decoderaddf(decoderErffQQ3, decodermulf(z, decoderaddf(decoderErffQQ4, decodermulf(z, decoderErffQQ5))))))))))
		y := decoderdivf(r, s)
		return decoderaddf(x, decodermulf(x, y))
	}

	var y float32
	if ix < 0x40c00000 {
		y = decodersubf(1.0, decoderErfc2(ix, x))
	} else {
		y = decodersubf(1.0, math.Float32frombits(0x03800000))
	}
	if sign != 0 {
		return decodernegf(y)
	}
	return y
}
