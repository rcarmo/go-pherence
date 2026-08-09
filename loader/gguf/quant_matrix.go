package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/half"
)

// QuantMatrix is a GGUF-encoded 2D matrix kept in its original quantized form.
// GGUF tensor shapes are [inDim, outDim], so each output row is contiguous.
type QuantMatrix struct {
	Name   string
	QType  QuantType
	Raw    []byte
	InDim  int
	OutDim int

	// llamaQ4_0x8 replaces Raw only for loader-selected Q4_0 projection
	// matrices. Keeping it private prevents packed bytes being mistaken for the
	// canonical GGUF row layout by external Raw consumers.
	llamaQ4_0x8 []byte
}

// MatrixFromTensor reads tensor t as a raw quantized matrix. Tensor shape must be [inDim, outDim].
func (g *GGUF) MatrixFromTensor(t TensorInfo) (*QuantMatrix, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("gguf: tensor %q shape %v is not a matrix", t.Name, t.Shape)
	}
	raw, err := g.Raw(t)
	if err != nil {
		return nil, err
	}
	return &QuantMatrix{Name: t.Name, QType: t.QType, Raw: raw, InDim: int(t.Shape[0]), OutDim: int(t.Shape[1])}, nil
}

// RowBytes returns the encoded byte count for a single output row.
func (m *QuantMatrix) RowBytes() (int, error) { return TensorRawBytes(m.QType, m.InDim) }

// DequantRowTo dequantizes one output row into dst.
func (m *QuantMatrix) DequantRowTo(dst []float32, row int) error {
	if row < 0 || row >= m.OutDim || len(dst) < m.InDim {
		return fmt.Errorf("gguf row %s: bad row=%d dst=%d in=%d", m.Name, row, len(dst), m.InDim)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	start := row * rowBytes
	end := start + rowBytes
	if end <= len(m.Raw) {
		return dequantRowTo(dst[:m.InDim], m.Raw[start:end], m.QType, m.InDim)
	}
	if m.QType == QuantQ4_0 && len(m.llamaQ4_0x8) > 0 {
		canonical := make([]byte, rowBytes)
		if err := unpackQ4_0x8RowTo(canonical, m.llamaQ4_0x8, row, m.OutDim, m.InDim/qk8_0); err != nil {
			return fmt.Errorf("gguf row %s: %w", m.Name, err)
		}
		return dequantRowTo(dst[:m.InDim], canonical, m.QType, m.InDim)
	}
	return fmt.Errorf("gguf row %s: row %d raw short", m.Name, row)
}

func dequantRowTo(dst []float32, raw []byte, qt QuantType, n int) error {
	switch qt {
	case QuantF32:
		if len(raw) < n*4 {
			return fmt.Errorf("F32 row raw short")
		}
		for i := 0; i < n; i++ {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return nil
	case QuantF16:
		if len(raw) < n*2 {
			return fmt.Errorf("F16 row raw short")
		}
		for i := 0; i < n; i++ {
			dst[i] = half.F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
		return nil
	case QuantBF16:
		if len(raw) < n*2 {
			return fmt.Errorf("BF16 row raw short")
		}
		for i := 0; i < n; i++ {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:])) << 16)
		}
		return nil
	case QuantQ4_0:
		return dequantRowQ4_0To(dst, raw, n)
	case QuantQ2_K:
		return dequantRowQ2KTo(dst, raw, n)
	case QuantQ3_K:
		return dequantRowQ3KTo(dst, raw, n)
	case QuantQ4_K:
		return dequantRowQ4KTo(dst, raw, n)
	case QuantQ6_K:
		return dequantRowQ6KTo(dst, raw, n)
	case QuantQ8_0:
		return dequantRowQ8_0To(dst, raw, n)
	case QuantQ5_0:
		return dequantRowQ5_0To(dst, raw, n)
	case QuantQ5_1:
		return dequantRowQ5_1To(dst, raw, n)
	default:
		return fmt.Errorf("unsupported row quant type %d", qt)
	}
}

func dequantRowQ4_0To(dst []float32, raw []byte, n int) error {
	const blockElems = 32
	const blockSize = 18
	if n%blockElems != 0 {
		return fmt.Errorf("Q4_0 row n=%d not multiple of 32", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q4_0 row raw short")
	}
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2:18]
		base := b * blockElems
		for i := 0; i < 16; i++ {
			q := qs[i]
			dst[base+i] = d * float32(int(q&0x0f)-8)
			dst[base+16+i] = d * float32(int(q>>4)-8)
		}
	}
	return nil
}

func dequantRowQ8_0To(dst []float32, raw []byte, n int) error {
	const blockElems = 32
	const blockSize = 34
	if n%blockElems != 0 {
		return fmt.Errorf("Q8_0 row n=%d not multiple of 32", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q8_0 row raw short")
	}
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2:34]
		base := b * blockElems
		for i := 0; i < blockElems; i++ {
			dst[base+i] = d * float32(int8(qs[i]))
		}
	}
	return nil
}

func dequantRowQ2KTo(dst []float32, raw []byte, n int) error {
	const blockElems = 256
	const blockSize = 84
	if n%blockElems != 0 {
		return fmt.Errorf("Q2_K row n=%d not multiple of 256", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q2_K row raw short")
	}
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
					dst[y] = dl*float32((q[qoff+l]>>shift)&3) - ml
					y++
				}
				sc = scales[is]
				is++
				dl = d * float32(sc&0x0F)
				ml = minv * float32(sc>>4)
				for l := 0; l < 16; l++ {
					dst[y] = dl*float32((q[qoff+l+16]>>shift)&3) - ml
					y++
				}
				shift += 2
			}
			qoff += 32
			_ = nn
		}
	}
	return nil
}

func dequantRowQ3KTo(dst []float32, raw []byte, n int) error {
	const blockElems = 256
	const blockSize = 110
	if n%blockElems != 0 {
		return fmt.Errorf("Q3_K row n=%d not multiple of 256", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q3_K row raw short")
	}
	const kmask1 uint32 = 0x03030303
	const kmask2 uint32 = 0x0f0f0f0f
	y := 0
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
		var scales [16]int8
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
					dst[y] = dl * float32(lo)
					y++
				}
				dl = dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					lo := int8((q[qoff+l+16] >> shift) & 3)
					if hm[l+16]&m == 0 {
						lo -= 4
					}
					dst[y] = dl * float32(lo)
					y++
				}
				shift += 2
				m <<= 1
			}
			qoff += 32
			_ = nn
		}
	}
	return nil
}

func dequantRowQ4KTo(dst []float32, raw []byte, n int) error {
	const blockElems = 256
	const blockSize = 144
	if n%blockElems != 0 {
		return fmt.Errorf("Q4_K row n=%d not multiple of 256", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q4_K row raw short")
	}
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		sc := blk[4:16]
		qs := blk[16:144]
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
		// ggml Q4_K stores each 64-value chunk as 32 low nibbles followed by
		// 32 high nibbles. Groups 0/1 share qs[0:32], groups 2/3 share
		// qs[32:64], etc. Do not read as alternating low/high nibbles.
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
				dst[base+group*32+i] = scales[group]*float32(q0) - mins[group]
				dst[base+group*32+16+i] = scales[group]*float32(q1) - mins[group]
			}
		}
	}
	return nil
}

func dequantRowQ6KTo(dst []float32, raw []byte, n int) error {
	const blockElems = 256
	const blockSize = 210
	if n%blockElems != 0 {
		return fmt.Errorf("Q6_K row n=%d not multiple of 256", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q6_K row raw short")
	}
	yBase := 0
	for b := 0; b < nBlocks; b++ {
		blk := raw[b*blockSize:]
		ql := blk[0:128]
		qh := blk[128:192]
		sc := blk[192:208]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
		y := yBase
		qlOff, qhOff, scOff := 0, 0, 0
		for nn := 0; nn < blockElems; nn += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l+0]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				dst[y+l+0] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
				dst[y+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
				dst[y+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
				dst[y+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
			}
			y += 128
			qlOff += 64
			qhOff += 32
			scOff += 8
			_ = nn
		}
		yBase += blockElems
	}
	return nil
}

// Q5_0 block: 22 bytes per 32 elements
//
//	d    f16 @ bytes[0:2]
//	qh   uint32 @ bytes[2:6]  — high bits (bit 4) for each of 32 elements
//	qs   [16]byte @ bytes[6:22] — low 4 bits, 2 per byte
func dequantRowQ5_0To(dst []float32, raw []byte, n int) error {
	const blockElems = 32
	const blockSize = 22
	if n%blockElems != 0 {
		return fmt.Errorf("Q5_0 row n=%d not multiple of 32", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q5_0 row raw short")
	}
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
			dst[base+i] = d * float32(q0-16)
			dst[base+i+16] = d * float32(q1-16)
		}
	}
	return nil
}

// Q5_1 block: 24 bytes per 32 elements
//
//	d    f16 @ bytes[0:2]
//	m    f16 @ bytes[2:4]
//	qh   uint32 @ bytes[4:8]
//	qs   [16]byte @ bytes[8:24]
func dequantRowQ5_1To(dst []float32, raw []byte, n int) error {
	const blockElems = 32
	const blockSize = 24
	if n%blockElems != 0 {
		return fmt.Errorf("Q5_1 row n=%d not multiple of 32", n)
	}
	nBlocks := n / blockElems
	if len(raw) < nBlocks*blockSize {
		return fmt.Errorf("Q5_1 row raw short")
	}
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
			dst[base+i] = d*float32(q0) + m
			dst[base+i+16] = d*float32(q1) + m
		}
	}
	return nil
}
