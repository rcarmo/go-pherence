package diffusiongemma

import (
	"encoding/binary"
	"fmt"

	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func quantizeQ8KBlockForLMHeadDot(x []float32) (d float32, qs [256]int8) {
	if len(x) < 256 {
		return 0, qs
	}
	maxVal, amax := float32(0), float32(0)
	for i := 0; i < 256; i++ {
		v := x[i]
		av := v
		if av < 0 {
			av = -av
		}
		if av > amax {
			amax = av
			maxVal = v
		}
	}
	if amax == 0 {
		return 0, qs
	}
	iscale := float32(-127) / maxVal
	d = 1 / iscale
	for i := 0; i < 256; i++ {
		q := ggmlNearestInt(iscale * x[i])
		if q > 127 {
			q = 127
		}
		if q < -128 {
			q = -128
		}
		qs[i] = int8(q)
	}
	return d, qs
}

type q8KPrequantRows struct {
	positions int
	blocks    int
	ds        []float32
	qs        []int8
}

func prequantizeQ8KRowsForLMHead(hidden []float32, positions, hiddenSize int) (*q8KPrequantRows, error) {
	if positions <= 0 || hiddenSize <= 0 || hiddenSize%256 != 0 || len(hidden) < positions*hiddenSize {
		return nil, fmt.Errorf("invalid Q8_K prequant rows positions=%d hidden=%d len=%d", positions, hiddenSize, len(hidden))
	}
	blocks := hiddenSize / 256
	out := &q8KPrequantRows{positions: positions, blocks: blocks, ds: make([]float32, positions*blocks), qs: make([]int8, positions*blocks*256)}
	for pos := 0; pos < positions; pos++ {
		row := hidden[pos*hiddenSize : (pos+1)*hiddenSize]
		for b := 0; b < blocks; b++ {
			d, q := quantizeQ8KBlockForLMHeadDot(row[b*256 : b*256+256])
			out.ds[pos*blocks+b] = d
			copy(out.qs[(pos*blocks+b)*256:(pos*blocks+b+1)*256], q[:])
		}
	}
	return out, nil
}

func ggufQ6KRawRowDotPrequant(raw []byte, inDim int, q8d []float32, q8qs []int8) (float32, error) {
	if inDim <= 0 || inDim%256 != 0 || len(q8d) < inDim/256 || len(q8qs) < inDim {
		return 0, fmt.Errorf("invalid Q6_K prequant row dot inDim=%d q8d=%d q8qs=%d", inDim, len(q8d), len(q8qs))
	}
	const blockSize = 210
	blocks := inDim / 256
	if len(raw) < blocks*blockSize {
		return 0, fmt.Errorf("Q6_K row raw short bytes=%d want=%d", len(raw), blocks*blockSize)
	}
	var total float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize : (b+1)*blockSize]
		ql := blk[0:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
		q8 := q8qs[b*256 : (b+1)*256]
		qlOff, qhOff := 0, 0
		var q6 [256]int8
		for j := 0; j < 256; j += 128 {
			for l := 0; l < 32; l++ {
				q6[j+l+0] = int8((ql[qlOff+l+0]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q6[j+l+32] = int8((ql[qlOff+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q6[j+l+64] = int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q6[j+l+96] = int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			}
			qlOff += 64
			qhOff += 32
		}
		isum, ok := q6KBlockISum(q8, &q6, scales)
		if !ok {
			return 0, fmt.Errorf("Q6_K/Q8_K block dot rejected")
		}
		total += d * q8d[b] * float32(isum)
	}
	return total, nil
}

func ggufQ6KRawRowDotQ8K(raw []byte, inDim int, x []float32) (float32, error) {
	if inDim <= 0 || inDim%256 != 0 || len(x) < inDim {
		return 0, fmt.Errorf("invalid Q6_K row dot inDim=%d x=%d", inDim, len(x))
	}
	const blockSize = 210
	blocks := inDim / 256
	if len(raw) < blocks*blockSize {
		return 0, fmt.Errorf("Q6_K row raw short bytes=%d want=%d", len(raw), blocks*blockSize)
	}
	var total float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize : (b+1)*blockSize]
		ql := blk[0:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
		q8d, q8 := quantizeQ8KBlockForLMHeadDot(x[b*256 : b*256+256])
		qlOff, qhOff := 0, 0
		var q6 [256]int8
		for j := 0; j < 256; j += 128 {
			for l := 0; l < 32; l++ {
				q6[j+l+0] = int8((ql[qlOff+l+0]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q6[j+l+32] = int8((ql[qlOff+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q6[j+l+64] = int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q6[j+l+96] = int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			}
			qlOff += 64
			qhOff += 32
		}
		isum, ok := q6KBlockISum(q8[:], &q6, scales)
		if !ok {
			return 0, fmt.Errorf("Q6_K/Q8_K block dot rejected")
		}
		total += d * q8d * float32(isum)
	}
	return total, nil
}

func ggufQ6KRawRowDotPrequantRows(raw []byte, inDim int, q8 *q8KPrequantRows, out []float32) error {
	if inDim <= 0 || inDim%256 != 0 || q8 == nil || q8.blocks != inDim/256 || len(out) < q8.positions {
		return fmt.Errorf("invalid Q6_K prequant rows dot inDim=%d", inDim)
	}
	const blockSize = 210
	blocks := inDim / 256
	if len(raw) < blocks*blockSize {
		return fmt.Errorf("Q6_K row raw short bytes=%d want=%d", len(raw), blocks*blockSize)
	}
	for i := 0; i < q8.positions; i++ {
		out[i] = 0
	}
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize : (b+1)*blockSize]
		ql := blk[0:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
		qlOff, qhOff := 0, 0
		var q6 [256]int8
		for j := 0; j < 256; j += 128 {
			for l := 0; l < 32; l++ {
				q6[j+l+0] = int8((ql[qlOff+l+0]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q6[j+l+32] = int8((ql[qlOff+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q6[j+l+64] = int8((ql[qlOff+l+0]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q6[j+l+96] = int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
			}
			qlOff += 64
			qhOff += 32
		}
		coeff, ok := q6KBlockScaledCoeffs(&q6, scales)
		if !ok {
			return fmt.Errorf("Q6_K scaled coefficient build rejected")
		}
		for pos := 0; pos < q8.positions; pos++ {
			q8base := (pos*blocks + b) * 256
			q8q := q8.qs[q8base : q8base+256]
			isum, ok := q6KBlockCoeffISum(q8q, &coeff)
			if !ok {
				return fmt.Errorf("Q6_K/Q8_K coeff block dot rejected")
			}
			out[pos] += d * q8.ds[pos*blocks+b] * float32(isum)
		}
	}
	return nil
}

func ggufQ6KMatrixRowDotPrequantRows(m *gguf.QuantMatrix, row int, q8 *q8KPrequantRows, out []float32) error {
	if m == nil || m.QType != gguf.QuantQ6_K || row < 0 || row >= m.OutDim {
		return fmt.Errorf("invalid Q6_K matrix prequant rows dot row=%d", row)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return err
	}
	start := row * rowBytes
	end := start + rowBytes
	if start < 0 || end > len(m.Raw) {
		return fmt.Errorf("Q6_K matrix row outside raw row=%d", row)
	}
	return ggufQ6KRawRowDotPrequantRows(m.Raw[start:end], m.InDim, q8, out)
}

func ggufQ6KMatrixRowDotPrequant(m *gguf.QuantMatrix, row int, q8d []float32, q8qs []int8) (float32, error) {
	if m == nil || m.QType != gguf.QuantQ6_K || row < 0 || row >= m.OutDim {
		return 0, fmt.Errorf("invalid Q6_K matrix prequant row dot row=%d", row)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return 0, err
	}
	start := row * rowBytes
	end := start + rowBytes
	if start < 0 || end > len(m.Raw) {
		return 0, fmt.Errorf("Q6_K matrix row outside raw row=%d", row)
	}
	return ggufQ6KRawRowDotPrequant(m.Raw[start:end], m.InDim, q8d, q8qs)
}

func ggufQ6KMatrixRowDotQ8K(m *gguf.QuantMatrix, row int, x []float32) (float32, error) {
	if m == nil || m.QType != gguf.QuantQ6_K || row < 0 || row >= m.OutDim || len(x) < m.InDim {
		return 0, fmt.Errorf("invalid Q6_K matrix row dot row=%d", row)
	}
	rowBytes, err := m.RowBytes()
	if err != nil {
		return 0, err
	}
	start := row * rowBytes
	end := start + rowBytes
	if start < 0 || end > len(m.Raw) {
		return 0, fmt.Errorf("Q6_K matrix row outside raw row=%d", row)
	}
	return ggufQ6KRawRowDotQ8K(m.Raw[start:end], m.InDim, x)
}
