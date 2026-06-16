package gguf

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/half"
)

const qkK = 256

const qk8_0 = 32

type q8KBlock struct {
	qs    [qkK]int8
	bsums [qkK / 16]int16
	d     float32
}

type q8_0Block struct {
	d  float32
	qs [qk8_0]int8
}

// QuantizeQ8K quantizes a float row using llama.cpp's reference Q8_K row
// quantizer. It is used by scalar fidelity paths for GGUF quantized matmuls.
func QuantizeQ8K(x []float32) ([]q8KBlock, error) {
	if len(x)%qkK != 0 {
		return nil, fmt.Errorf("Q8_K quantize len=%d not multiple of %d", len(x), qkK)
	}
	blocks := make([]q8KBlock, len(x)/qkK)
	for bi := range blocks {
		row := x[bi*qkK : (bi+1)*qkK]
		var max, amax float32
		for _, v := range row {
			av := float32(math.Abs(float64(v)))
			if av > amax {
				amax = av
				max = v
			}
		}
		if amax == 0 {
			continue
		}
		iscale := float32(-127.0) / max
		for j, v := range row {
			q := nearestIntGGML(iscale * v)
			if q > 127 {
				q = 127
			}
			if q < -128 {
				q = -128
			}
			blocks[bi].qs[j] = int8(q)
		}
		for j := 0; j < qkK/16; j++ {
			sum := 0
			for ii := 0; ii < 16; ii++ {
				sum += int(blocks[bi].qs[j*16+ii])
			}
			blocks[bi].bsums[j] = int16(sum)
		}
		blocks[bi].d = 1 / iscale
	}
	return blocks, nil
}

func nearestIntGGML(v float32) int {
	// ggml nearest_int: add 12582912.0f and extract mantissa bits. This matches
	// the CPU quantizer's round-to-nearest-even behavior for the bounded inputs
	// used by Q8_K activation quantization.
	bits := math.Float32bits(v + 12582912.0)
	return int(bits&0x007fffff) - 0x00400000
}

// QuantizeQ8_0 quantizes a float row using llama.cpp's reference Q8_0 row
// quantizer. The block scale is rounded through FP16 because GGML stores it as ggml_half.
func QuantizeQ8_0(x []float32) ([]q8_0Block, error) {
	if len(x)%qk8_0 != 0 {
		return nil, fmt.Errorf("Q8_0 quantize len=%d not multiple of %d", len(x), qk8_0)
	}
	blocks := make([]q8_0Block, len(x)/qk8_0)
	for bi := range blocks {
		row := x[bi*qk8_0 : (bi+1)*qk8_0]
		var amax float32
		for _, v := range row {
			av := float32(math.Abs(float64(v)))
			if av > amax {
				amax = av
			}
		}
		d := amax / 127.0
		id := float32(0)
		if d != 0 {
			id = 1 / d
		}
		blocks[bi].d = half.F16ToF32(half.F32ToF16(d))
		for j, v := range row {
			q := nearestIntGGML(v * id)
			if q > 127 {
				q = 127
			}
			if q < -128 {
				q = -128
			}
			blocks[bi].qs[j] = int8(q)
		}
	}
	return blocks, nil
}

// DotQ4_0Q8_0 computes llama.cpp's scalar ggml_vec_dot_q4_0_q8_0 over one
// Q4_0 row and a pre-quantized Q8_0 activation row.
func DotQ4_0Q8_0(raw []byte, y []q8_0Block, n int) (float32, error) {
	if n%qk8_0 != 0 {
		return 0, fmt.Errorf("Q4_0 dot n=%d not multiple of %d", n, qk8_0)
	}
	nb := n / qk8_0
	const blockSize = 18
	if len(raw) < nb*blockSize || len(y) < nb {
		return 0, fmt.Errorf("Q4_0 dot raw/activation short raw=%d y=%d nb=%d", len(raw), len(y), nb)
	}
	var sum float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		qs := blk[2:18]
		s0 := 0
		s1 := 0
		for j := 0; j < qk8_0/2; j++ {
			v0 := int(qs[j]&0x0F) - 8
			v1 := int(qs[j]>>4) - 8
			s0 += v0 * int(y[bi].qs[j])
			s1 += v1 * int(y[bi].qs[j+qk8_0/2])
		}
		sum += float32(s0+s1) * d
	}
	return sum, nil
}

// GemvQ4_0Q8_0Rows computes all rows of a Q4_0 matrix against x using llama.cpp's
// Q8_0 activation quantization. It returns false when dims/types are unsuitable.
func GemvQ4_0Q8_0Rows(out, x []float32, m *QuantMatrix) bool {
	if m == nil || m.QType != QuantQ4_0 || len(out) != m.OutDim || len(x) != m.InDim || m.InDim%qk8_0 != 0 {
		return false
	}
	q8, err := QuantizeQ8_0(x)
	if err != nil {
		return false
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return false
	}
	for r := 0; r < m.OutDim; r++ {
		start := r * rowBytes
		v, err := DotQ4_0Q8_0(m.Raw[start:start+rowBytes], q8, m.InDim)
		if err != nil {
			return false
		}
		out[r] = v
	}
	return true
}

// DotQ6KQ8K computes llama.cpp's scalar ggml_vec_dot_q6_K_q8_K over one
// Q6_K row and a pre-quantized Q8_K activation row.
func DotQ6KQ8K(raw []byte, y []q8KBlock, n int) (float32, error) {
	if n%qkK != 0 {
		return 0, fmt.Errorf("Q6_K dot n=%d not multiple of %d", n, qkK)
	}
	nb := n / qkK
	const blockSize = 210
	if len(raw) < nb*blockSize || len(y) < nb {
		return 0, fmt.Errorf("Q6_K dot raw/activation short raw=%d y=%d nb=%d", len(raw), len(y), nb)
	}
	var sums [8]float32
	var aux8 [qkK]int8
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		q4 := blk[0:128]
		qh := blk[128:192]
		sc := blk[192:208]
		a := 0
		q4Off, qhOff := 0, 0
		for j := 0; j < qkK; j += 128 {
			_ = j
			for l := 0; l < 32; l++ {
				aux8[a+l+0] = int8((q4[q4Off+l+0]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				aux8[a+l+32] = int8((q4[q4Off+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				aux8[a+l+64] = int8((q4[q4Off+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				aux8[a+l+96] = int8((q4[q4Off+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			}
			a += 128
			q4Off += 64
			qhOff += 32
		}
		var aux32 [8]int32
		a = 0
		q8 := 0
		is := 0
		for j := 0; j < qkK/16; j++ {
			_ = j
			scale := int(int8(sc[is]))
			is++
			for l := 0; l < 8; l++ {
				aux32[l] += int32(scale * int(y[bi].qs[q8+l]) * int(aux8[a+l]))
			}
			q8 += 8
			a += 8
			for l := 0; l < 8; l++ {
				aux32[l] += int32(scale * int(y[bi].qs[q8+l]) * int(aux8[a+l]))
			}
			q8 += 8
			a += 8
		}
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210])) * y[bi].d
		for l := 0; l < 8; l++ {
			sums[l] += d * float32(aux32[l])
		}
	}
	var sum float32
	for l := 0; l < 8; l++ {
		sum += sums[l]
	}
	return sum, nil
}

// GemvQ6KQ8KRows computes all rows of a Q6_K matrix against x using llama.cpp's
// Q8_K activation quantization. It returns false when dims/types are unsuitable.
func GemvQ6KQ8KRows(out, x []float32, m *QuantMatrix) bool {
	if m == nil || m.QType != QuantQ6_K || len(out) != m.OutDim || len(x) != m.InDim || m.InDim%qkK != 0 {
		return false
	}
	q8, err := QuantizeQ8K(x)
	if err != nil {
		return false
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return false
	}
	for r := 0; r < m.OutDim; r++ {
		start := r * rowBytes
		v, err := DotQ6KQ8K(m.Raw[start:start+rowBytes], q8, m.InDim)
		if err != nil {
			return false
		}
		out[r] = v
	}
	return true
}
