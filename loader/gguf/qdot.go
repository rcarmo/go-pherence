package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"

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

type q4Q8Correction [qk8_0 / 4]int32

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
			q := int(math.Round(float64(v * id)))
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

// DotQ4_0Q8_0 computes llama.cpp's AVX/VNNI-lane ggml_vec_dot_q4_0_q8_0 over one
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
	return dotQ4_0Q8_0Packed(raw, y, nb), nil
}

func dotQ4_0Q8_0Scalar(raw []byte, y []q8_0Block, nb int) float32 {
	const blockSize = 18
	var acc [8]float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		qs := blk[2:18]
		for lane := 0; lane < 4; lane++ {
			j := lane * 4
			s := (int(qs[j+0]&0x0F)-8)*int(y[bi].qs[j+0]) + (int(qs[j+1]&0x0F)-8)*int(y[bi].qs[j+1]) + (int(qs[j+2]&0x0F)-8)*int(y[bi].qs[j+2]) + (int(qs[j+3]&0x0F)-8)*int(y[bi].qs[j+3])
			acc[lane] = float32(math.FMA(float64(d), float64(float32(s)), float64(acc[lane])))
		}
		for lane := 0; lane < 4; lane++ {
			j := lane * 4
			s := (int(qs[j+0]>>4)-8)*int(y[bi].qs[j+16]) + (int(qs[j+1]>>4)-8)*int(y[bi].qs[j+17]) + (int(qs[j+2]>>4)-8)*int(y[bi].qs[j+18]) + (int(qs[j+3]>>4)-8)*int(y[bi].qs[j+19])
			acc[lane+4] = float32(math.FMA(float64(d), float64(float32(s)), float64(acc[lane+4])))
		}
	}
	r0 := acc[0] + acc[4]
	r1 := acc[1] + acc[5]
	r2 := acc[2] + acc[6]
	r3 := acc[3] + acc[7]
	r0 = r0 + r2
	r1 = r1 + r3
	return r0 + r1
}

func q4Q8Corrections(y []q8_0Block) []q4Q8Correction {
	corrections := make([]q4Q8Correction, len(y))
	for bi := range y {
		for lane := range corrections[bi] {
			base := lane * 4
			corrections[bi][lane] = 8 * (int32(y[bi].qs[base]) + int32(y[bi].qs[base+1]) + int32(y[bi].qs[base+2]) + int32(y[bi].qs[base+3]))
		}
	}
	return corrections
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
	if supportsQ4_0Q8_0Rows8() {
		corrections := q4Q8Corrections(q8)
		groups := m.OutDim / 8
		if groups > 0 && !gemvRowsParallel(groups, rowBytes*8, func(group int) bool {
			r := group * 8
			start := r * rowBytes
			var values [8]float32
			if !dotQ4_0Q8_0Rows8VNNI(m.Raw[start:start+8*rowBytes], rowBytes, q8, corrections, m.InDim/qk8_0, &values) {
				return false
			}
			copy(out[r:r+8], values[:])
			return true
		}) {
			return false
		}
		for r := groups * 8; r < m.OutDim; r++ {
			start := r * rowBytes
			out[r] = dotQ4_0Q8_0Packed(m.Raw[start:start+rowBytes], q8, m.InDim/qk8_0)
		}
		return true
	}
	groups := m.OutDim / 4
	if groups > 0 && !gemvRowsParallel(groups, rowBytes*4, func(group int) bool {
		r := group * 4
		start := r * rowBytes
		var values [4]float32
		dotQ4_0Q8_0Rows4(m.Raw[start:start+4*rowBytes], rowBytes, q8, m.InDim/qk8_0, &values)
		copy(out[r:r+4], values[:])
		return true
	}) {
		return false
	}
	for r := groups * 4; r < m.OutDim; r++ {
		start := r * rowBytes
		out[r] = dotQ4_0Q8_0Packed(m.Raw[start:start+rowBytes], q8, m.InDim/qk8_0)
	}
	return true
}

// DotQ5_0Q8_0 computes a Q5_0 row against a pre-quantized Q8_0 activation row.
func DotQ5_0Q8_0(raw []byte, y []q8_0Block, n int) (float32, error) {
	if n%qk8_0 != 0 {
		return 0, fmt.Errorf("Q5_0 dot n=%d not multiple of %d", n, qk8_0)
	}
	nb := n / qk8_0
	const blockSize = 22
	if len(raw) < nb*blockSize || len(y) < nb {
		return 0, fmt.Errorf("Q5_0 dot raw/activation short raw=%d y=%d nb=%d", len(raw), len(y), nb)
	}
	var sum float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		qh := binary.LittleEndian.Uint32(blk[2:6])
		qs := blk[6:22]
		blockSum := 0
		for j := 0; j < qk8_0/2; j++ {
			q0 := int(qs[j] & 0x0F)
			q1 := int(qs[j] >> 4)
			if qh&(1<<uint(j)) != 0 {
				q0 |= 16
			}
			if qh&(1<<uint(j+16)) != 0 {
				q1 |= 16
			}
			blockSum += (q0 - 16) * int(y[bi].qs[j])
			blockSum += (q1 - 16) * int(y[bi].qs[j+16])
		}
		sum = float32(math.FMA(float64(d), float64(float32(blockSum)), float64(sum)))
	}
	return sum, nil
}

// DotQ8_0Q8_0 computes a Q8_0 row against a pre-quantized Q8_0 activation row.
func DotQ8_0Q8_0(raw []byte, y []q8_0Block, n int) (float32, error) {
	if n%qk8_0 != 0 {
		return 0, fmt.Errorf("Q8_0 dot n=%d not multiple of %d", n, qk8_0)
	}
	nb := n / qk8_0
	const blockSize = 34
	if len(raw) < nb*blockSize || len(y) < nb {
		return 0, fmt.Errorf("Q8_0 dot raw/activation short raw=%d y=%d nb=%d", len(raw), len(y), nb)
	}
	var sum float32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		qs := blk[2:34]
		blockSum := 0
		for j := 0; j < qk8_0; j++ {
			blockSum += int(int8(qs[j])) * int(y[bi].qs[j])
		}
		sum = float32(math.FMA(float64(d), float64(float32(blockSum)), float64(sum)))
	}
	return sum, nil
}

// DotQ4KQ8K computes llama.cpp's generic ggml_vec_dot_q4_K_q8_K over one
// Q4_K row and a pre-quantized Q8_K activation row.
func DotQ4KQ8K(raw []byte, y []q8KBlock, n int) (float32, error) {
	if n%qkK != 0 {
		return 0, fmt.Errorf("Q4_K dot n=%d not multiple of %d", n, qkK)
	}
	nb := n / qkK
	const blockSize = 144
	if len(raw) < nb*blockSize || len(y) < nb {
		return 0, fmt.Errorf("Q4_K dot raw/activation short raw=%d y=%d nb=%d", len(raw), len(y), nb)
	}
	const kmask1 uint32 = 0x3f3f3f3f
	const kmask2 uint32 = 0x0f0f0f0f
	const kmask3 uint32 = 0x03030303
	var sums [8]float32
	var aux8 [qkK]int8
	var aux32 [8]int32
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		q4 := blk[16:144]
		q8 := y[bi].qs[:]
		for i := range aux32 {
			aux32[i] = 0
		}
		a := aux8[:]
		for j := 0; j < qkK/64; j++ {
			for l := 0; l < 32; l++ {
				a[l] = int8(q4[l] & 0x0F)
			}
			a = a[32:]
			for l := 0; l < 32; l++ {
				a[l] = int8(q4[l] >> 4)
			}
			a = a[32:]
			q4 = q4[32:]
		}
		var utmp [4]uint32
		utmp[0] = binary.LittleEndian.Uint32(blk[4:8])
		utmp[1] = binary.LittleEndian.Uint32(blk[8:12])
		utmp[2] = binary.LittleEndian.Uint32(blk[12:16])
		utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4)
		uaux := utmp[1] & kmask1
		utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4)
		utmp[2] = uaux
		utmp[0] &= kmask1
		var scaleBytes [16]byte
		binary.LittleEndian.PutUint32(scaleBytes[0:4], utmp[0])
		binary.LittleEndian.PutUint32(scaleBytes[4:8], utmp[1])
		binary.LittleEndian.PutUint32(scaleBytes[8:12], utmp[2])
		binary.LittleEndian.PutUint32(scaleBytes[12:16], utmp[3])
		minsBytes := scaleBytes[8:]
		sumi := 0
		for j := 0; j < qkK/16; j++ {
			sumi += int(y[bi].bsums[j]) * int(minsBytes[j/2])
		}
		a = aux8[:]
		is := 0
		q8p := q8[:]
		for j := 0; j < qkK/32; j++ {
			scale := int32(scaleBytes[is])
			is++
			for chunk := 0; chunk < 4; chunk++ {
				for l := 0; l < 8; l++ {
					aux32[l] += scale * int32(q8p[l]) * int32(a[l])
				}
				q8p = q8p[8:]
				a = a[8:]
			}
		}
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2])) * y[bi].d
		for l := 0; l < 8; l++ {
			sums[l] = float32(math.FMA(float64(d), float64(float32(aux32[l])), float64(sums[l])))
		}
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4])) * y[bi].d
		sums[0] = float32(math.FMA(float64(-dmin), float64(float32(sumi)), float64(sums[0])))
	}
	var sum float32
	for l := 0; l < 8; l++ {
		sum += sums[l]
	}
	return sum, nil
}

// DotQ6KQ8K computes llama.cpp's AVX-style ggml_vec_dot_q6_K_q8_K over one
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
	for bi := 0; bi < nb; bi++ {
		blk := raw[bi*blockSize : (bi+1)*blockSize]
		ql := blk[0:128]
		qh := blk[128:192]
		sc := blk[192:208]
		var sumi [8]int32
		var q8sclsub [8]int32
		for l := 0; l < 8; l++ {
			q8sclsub[l] = int32((int(y[bi].bsums[2*l])*int(int8(sc[2*l])) + int(y[bi].bsums[2*l+1])*int(int8(sc[2*l+1]))) << 5)
		}
		for halfBlock := 0; halfBlock < 2; halfBlock++ {
			qlOff := halfBlock * 64
			qhOff := halfBlock * 32
			q8Base := halfBlock * 128
			scaleBase := halfBlock * 8
			for vec := 0; vec < 4; vec++ {
				for lane := 0; lane < 8; lane++ {
					scale := int(int8(sc[scaleBase+vec*2]))
					if lane >= 4 {
						scale = int(int8(sc[scaleBase+vec*2+1]))
					}
					seg := 0
					for p := 0; p < 4; p++ {
						idx := vec*32 + lane*4 + p
						var q int
						switch vec {
						case 0:
							q = int((ql[qlOff+idx] & 0x0F) | (((qh[qhOff+idx] >> 0) & 3) << 4))
						case 1:
							q = int((ql[qlOff+idx] & 0x0F) | (((qh[qhOff+idx-32] >> 2) & 3) << 4))
						case 2:
							q = int((ql[qlOff+idx-64] >> 4) | (((qh[qhOff+idx-64] >> 4) & 3) << 4))
						case 3:
							q = int((ql[qlOff+idx-64] >> 4) | (((qh[qhOff+idx-96] >> 6) & 3) << 4))
						}
						seg += q * int(y[bi].qs[q8Base+idx])
					}
					sumi[lane] += int32(scale * seg)
				}
			}
		}
		for l := 0; l < 8; l++ {
			sumi[l] -= q8sclsub[l]
		}
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210])) * y[bi].d
		for l := 0; l < 8; l++ {
			sums[l] = float32(math.FMA(float64(d), float64(float32(sumi[l])), float64(sums[l])))
		}
	}
	r0 := sums[0] + sums[4]
	r1 := sums[1] + sums[5]
	r2 := sums[2] + sums[6]
	r3 := sums[3] + sums[7]
	r0 = r0 + r2
	r1 = r1 + r3
	return r0 + r1, nil
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
	return gemvRowsParallel(m.OutDim, rowBytes, func(r int) bool {
		start := r * rowBytes
		out[r] = dotQ6KQ8KGemvFast(m.Raw[start:start+rowBytes], q8, m.InDim/qkK)
		return true
	})
}

func gemvRowsParallel(outDim, rowBytes int, fn func(row int) bool) bool {
	if outDim <= 0 || rowBytes <= 0 {
		return false
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 || outDim < 256 {
		for r := 0; r < outDim; r++ {
			if !fn(r) {
				return false
			}
		}
		return true
	}
	if workers > outDim {
		workers = outDim
	}
	chunk := (outDim + workers - 1) / workers
	var wg sync.WaitGroup
	okCh := make(chan bool, workers)
	for start := 0; start < outDim; start += chunk {
		end := start + chunk
		if end > outDim {
			end = outDim
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for r := start; r < end; r++ {
				if !fn(r) {
					okCh <- false
					return
				}
			}
			okCh <- true
		}(start, end)
	}
	wg.Wait()
	close(okCh)
	for ok := range okCh {
		if !ok {
			return false
		}
	}
	return true
}
