package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/half"
)

// dequantToF32 dequantizes raw GGUF tensor bytes to a []float32 of length n.
// DequantToF32 dequantizes raw quantized data to a new F32 slice.
func DequantToF32(raw []byte, qt QuantType, n int) ([]float32, error) {
	return dequantToF32(raw, qt, n)
}

func dequantToF32(raw []byte, qt QuantType, n int) ([]float32, error) {
	switch qt {
	case QuantF32:
		return dequantF32(raw, n)
	case QuantF16:
		return dequantF16(raw, n)
	case QuantBF16:
		return dequantBF16(raw, n)
	case QuantQ4_0:
		return dequantQ4_0(raw, n)
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
	case QuantQ5_0:
		return dequantQ5_0(raw, n)
	case QuantQ5_1:
		return dequantQ5_1(raw, n)
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
		out[i] = half.F16ToF32(u)
	}
	return out, nil
}

func dequantBF16(raw []byte, n int) ([]float32, error) {
	if len(raw) < n*2 {
		return nil, fmt.Errorf("dequantBF16: need %d bytes, have %d", n*2, len(raw))
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
	}
	return out, nil
}

// ── Q4_0 ──────────────────────────────────────────────────────────────────────
// Block: 18 bytes per 32 elements
//   d    f16 @ bytes[0:2]
//   qs   uint4[32] @ bytes[2:18], low nibble first, value biased by -8

func dequantQ4_0(raw []byte, n int) ([]float32, error) {
	const blockElems = 32
	const blockSize = 18
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ4_0: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2:18]
		base := b * blockElems
		for i := 0; i < 16; i++ {
			q := qs[i]
			out[base+i] = d * float32(int(q&0x0f)-8)
			out[base+16+i] = d * float32(int(q>>4)-8)
		}
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
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
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
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[80:82]))
		minv := half.F16ToF32(binary.LittleEndian.Uint16(blk[82:84]))
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
		dAll := half.F16ToF32(binary.LittleEndian.Uint16(blk[108:110]))

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
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
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
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
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

// ── Q5_0 ──────────────────────────────────────────────────────────────────────
// Block: 22 bytes per 32 elements
//   d    f16 @ bytes[0:2]
//   qh   uint32 @ bytes[2:6]  — 5th bit for each element
//   qs   [16]byte @ bytes[6:22] — low 4 bits, 2 per byte

func dequantQ5_0(raw []byte, n int) ([]float32, error) {
	const blockElems = 32
	const blockSize = 22
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ5_0: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qh := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:22]
		base := b * blockElems
		for i := 0; i < 16; i++ {
			q0 := int(qs[i] & 0x0F)
			q1 := int(qs[i] >> 4)
			if qh&(1<<uint(i)) != 0 {
				q0 |= 16
			}
			if qh&(1<<uint(i+16)) != 0 {
				q1 |= 16
			}
			out[base+i] = d * float32(q0-16)
			out[base+i+16] = d * float32(q1-16)
		}
	}
	return out, nil
}

// ── Q5_1 ──────────────────────────────────────────────────────────────────────
// Block: 24 bytes per 32 elements
//   d    f16 @ bytes[0:2]
//   m    f16 @ bytes[2:4]
//   qh   uint32 @ bytes[4:8]
//   qs   [16]byte @ bytes[8:24]

func dequantQ5_1(raw []byte, n int) ([]float32, error) {
	const blockElems = 32
	const blockSize = 24
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return nil, fmt.Errorf("dequantQ5_1: need %d bytes, have %d", nBlocks*blockSize, len(raw))
	}
	out := make([]float32, n)
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		m := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		qh := binary.LittleEndian.Uint32(blk[4:8])
		qs := blk[8:24]
		base := b * blockElems
		for i := 0; i < 16; i++ {
			q0 := int(qs[i] & 0x0F)
			q1 := int(qs[i] >> 4)
			if qh&(1<<uint(i)) != 0 {
				q0 |= 16
			}
			if qh&(1<<uint(i+16)) != 0 {
				q1 |= 16
			}
			out[base+i] = d*float32(q0) + m
			out[base+i+16] = d*float32(q1) + m
		}
	}
	return out, nil
}
