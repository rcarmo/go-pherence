package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// f16ToF32 converts an IEEE 754 half-precision float to float32 without unsafe.
func f16ToF32(u uint16) float32 {
	sign := uint32(u >> 15)
	exp := uint32((u >> 10) & 0x1F)
	mant := uint32(u & 0x3FF)

	if exp == 0x1F {
		// inf or NaN
		return math.Float32frombits(sign<<31 | 0x7F800000 | mant<<13)
	}
	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31) // ±0
		}
		// Denormal: normalize
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	}
	return math.Float32frombits(sign<<31 | (exp+112)<<23 | mant<<13)
}

// dequantToF32 dequantizes raw GGUF tensor bytes to a []float32 of length n.
func dequantToF32(raw []byte, qt QuantType, n int) ([]float32, error) {
	switch qt {
	case QuantF32:
		return dequantF32(raw, n)
	case QuantF16:
		return dequantF16(raw, n)
	case QuantQ8_0:
		return dequantQ8_0(raw, n)
	case QuantQ2_K:
		return dequantQ2K(raw, n)
	case QuantQ3_K:
		return dequantQ3K(raw, n)
	case QuantQ4_K:
		return dequantQ4K(raw, n)
	case QuantQ6_K:
		return dequantQ6K(raw, n)
	default:
		return nil, fmt.Errorf("dequant: unsupported quant type %d", qt)
	}
}

// ── F32 ───────────────────────────────────────────────────────────────────────

func dequantF32(raw []byte, n int) ([]float32, error) {
	if len(raw) < n*4 {
		return nil, fmt.Errorf("dequantF32: need %d bytes, have %d", n*4, len(raw))
	}
	out := make([]float32, n)
	for i := range out {
		bits := binary.LittleEndian.Uint32(raw[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

// ── F16 ───────────────────────────────────────────────────────────────────────

func dequantF16(raw []byte, n int) ([]float32, error) {
	if len(raw) < n*2 {
		return nil, fmt.Errorf("dequantF16: need %d bytes, have %d", n*2, len(raw))
	}
	out := make([]float32, n)
	for i := range out {
		u := binary.LittleEndian.Uint16(raw[i*2:])
		out[i] = f16ToF32(u)
	}
	return out, nil
}

// ── Q8_0 ──────────────────────────────────────────────────────────────────────
// Block: 34 bytes per 32 elements
//   d    f16 @ bytes[0:2]
//   qs   int8[32] @ bytes[2:34]

func dequantQ8_0(raw []byte, n int) ([]float32, error) {
	const blockElems = 32
	const blockSize = 34
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ8_0: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2:34]
		base := b * blockElems
		for i := 0; i < blockElems; i++ {
			out[base+i] = d * float32(int8(qs[i]))
		}
	}
	return out, nil
}

// ── Q2_K ──────────────────────────────────────────────────────────────────────
// Block: 84 bytes per 256 elements (QK_K=256)
//   scales[16]  nibble-packed (lo=scale_i, hi=min_i, 16 groups of 16)
//   qs[64]      2-bit quants, 4 per byte
//   d     f16 @ bytes[80:82]
//   dmin  f16 @ bytes[82:84]

func dequantQ2K(raw []byte, n int) ([]float32, error) {
	const blockElems = 256
	const blockSize = 84
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ2K: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	y := 0
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		scales := blk[0:16]
		q := blk[16:80]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[80:82]))
		minv := f16ToF32(binary.LittleEndian.Uint16(blk[82:84]))
		is := 0
		qoff := 0
		for nn := 0; nn < blockElems; nn += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl := d * float32(sc&0x0F)
				ml := minv * float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[y] = dl*float32((q[qoff+l]>>shift)&3) - ml
					y++
				}

				sc = scales[is]
				is++
				dl = d * float32(sc&0x0F)
				ml = minv * float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[y] = dl*float32((q[qoff+l+16]>>shift)&3) - ml
					y++
				}
				shift += 2
			}
			qoff += 32
		}
	}
	return out, nil
}

// ── Q3_K ──────────────────────────────────────────────────────────────────────
// Block: 110 bytes per 256 elements (QK_K=256)
//   hmask[32]   high bit per element
//   qs[64]      low 2 bits per element
//   scales[12]  8 × 6-bit signed scales packed
//   d     f16 @ bytes[108:110]

func dequantQ3K(raw []byte, n int) ([]float32, error) {
	const blockElems = 256
	const blockSize = 110
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ3K: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	y := 0
	const kmask1 uint32 = 0x03030303
	const kmask2 uint32 = 0x0f0f0f0f
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		hm := blk[0:32]
		q := blk[32:96]
		s := blk[96:108]
		dAll := f16ToF32(binary.LittleEndian.Uint16(blk[108:110]))

		aux := [4]uint32{
			binary.LittleEndian.Uint32(s[0:4]),
			binary.LittleEndian.Uint32(s[4:8]),
			binary.LittleEndian.Uint32(s[8:12]),
			0,
		}
		tmp := aux[2]
		aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
		aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

		scales := [16]int8{}
		for i := 0; i < 4; i++ {
			u := aux[i]
			scales[4*i+0] = int8(byte(u >> 0))
			scales[4*i+1] = int8(byte(u >> 8))
			scales[4*i+2] = int8(byte(u >> 16))
			scales[4*i+3] = int8(byte(u >> 24))
		}

		is := 0
		m := byte(1)
		qoff := 0
		for nn := 0; nn < blockElems; nn += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					lo := int8((q[qoff+l+0] >> shift) & 3)
					if hm[l+0]&m == 0 {
						lo -= 4
					}
					out[y] = dl * float32(lo)
					y++
				}

				dl = dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					lo := int8((q[qoff+l+16] >> shift) & 3)
					if hm[l+16]&m == 0 {
						lo -= 4
					}
					out[y] = dl * float32(lo)
					y++
				}
				shift += 2
				m <<= 1
			}
			qoff += 32
		}
	}
	return out, nil
}

// ── Q4_K ──────────────────────────────────────────────────────────────────────
// Block: 144 bytes per 256 elements (QK_K=256)
//   d      f16 @ bytes[0:2]
//   dmin   f16 @ bytes[2:4]
//   scales[12]  8 super-block scales+mins packed (6 bits each) @ bytes[4:16]
//   qs[128]     4-bit quants, 2 per byte @ bytes[16:144]

func dequantQ4K(raw []byte, n int) ([]float32, error) {
	const blockElems = 256
	const blockSize = 144
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ4K: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		sc := blk[4:16]
		qs := blk[16:144]

		// Unpack 8 (scale,min) pairs from 12 bytes.
		// Each pair is 6 bits; layout matches llama.cpp get_scale_min_k4.
		var scales [8]float32
		var mins [8]float32
		for j := 0; j < 4; j++ {
			scales[j] = float32(sc[j]&63) * d
			mins[j] = float32(sc[j+4]&63) * dmin
		}
		for j := 4; j < 8; j++ {
			k := j - 4
			scales[j] = float32((sc[j+4]&0xF)|((sc[k]>>6)<<4)) * d
			mins[j] = float32((sc[j+4]>>4)|((sc[k+4]>>6)<<4)) * dmin
		}

		base := b * blockElems
		for group := 0; group < 8; group++ {
			q := qs[(group/2)*32:]
			for i := 0; i < 16; i++ {
				var q0, q1 int
				if group%2 == 0 {
					q0 = int(q[i] & 0x0F)
					q1 = int(q[i+16] & 0x0F)
				} else {
					q0 = int(q[i] >> 4)
					q1 = int(q[i+16] >> 4)
				}
				out[base+group*32+i] = scales[group]*float32(q0) - mins[group]
				out[base+group*32+16+i] = scales[group]*float32(q1) - mins[group]
			}
		}
	}
	return out, nil
}

// ── Q6_K ──────────────────────────────────────────────────────────────────────
// Block: 210 bytes per 256 elements (QK_K=256)
//   ql[128]     lower 4 bits per element
//   qh[64]      upper 2 bits per element
//   scales[16]  int8 per 16-element group @ bytes[192:208]
//   d     f16 @ bytes[208:210]

func dequantQ6K(raw []byte, n int) ([]float32, error) {
	const blockElems = 256
	const blockSize = 210
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ6K: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	yBase := 0
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		ql := blk[0:128]
		qh := blk[128:192]
		sc := blk[192:208]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
		y := yBase
		qlOff := 0
		qhOff := 0
		scOff := 0
		for nn := 0; nn < blockElems; nn += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l+0]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				out[y+l+0] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
				out[y+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
				out[y+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
				out[y+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
			}
			y += 128
			qlOff += 64
			qhOff += 32
			scOff += 8
		}
		yBase += blockElems
	}
	return out, nil
}
