package nvidia

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func TestGemvQ5KBatchMatchesIndependentDequant(t *testing.T) {
	if !SgemmReady() {
		if Available() {
			t.Fatal("CUDA device is available but the PTX runtime is not ready")
		}
		t.Skip("CUDA not available")
	}
	const inDim, outDim, batch = 512, 7, 3
	raw := syntheticQKRaw(outDim, inDim, 176, func(blk []byte, row, block int) {
		binary.LittleEndian.PutUint16(blk[0:2], half.F32ToF16(0.025+float32(row+block)*0.002))
		binary.LittleEndian.PutUint16(blk[2:4], half.F32ToF16(0.004+float32((row+block)%3)*0.001))
		for i := 0; i < 12; i++ {
			blk[4+i] = byte((row*11 + block*7 + i*13 + 3) & 0xff)
		}
		for i := 0; i < 32; i++ {
			blk[16+i] = byte((row*5 + block*3 + i*17 + 1) & 0xff)
		}
		for i := 0; i < 128; i++ {
			blk[48+i] = byte((row*19 + block*23 + i*29 + 9) & 0xff)
		}
	})
	m, err := UploadQ5KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := syntheticQKInput(batch * inDim)
	got := make([]float32, batch*outDim)
	if err := GemvQ5KBatch(got, x, batch, m); err != nil {
		t.Fatal(err)
	}
	assertQKBatchClose(t, got, x, raw, batch, inDim, outDim, 176, dequantQ5KTest, 2e-4)
}

func TestGemvQ6KBatchMatchesIndependentDequant(t *testing.T) {
	if !SgemmReady() {
		if Available() {
			t.Fatal("CUDA device is available but the PTX runtime is not ready")
		}
		t.Skip("CUDA not available")
	}
	const inDim, outDim, batch = 512, 5, 3
	raw := syntheticQKRaw(outDim, inDim, 210, func(blk []byte, row, block int) {
		for i := 0; i < 128; i++ {
			blk[i] = byte((row*13 + block*7 + i*19 + 5) & 0xff)
		}
		for i := 0; i < 64; i++ {
			blk[128+i] = byte((row*17 + block*11 + i*23 + 3) & 0xff)
		}
		for i := 0; i < 16; i++ {
			blk[192+i] = byte(int8((row*5+block*3+i*7)%31 - 15))
		}
		binary.LittleEndian.PutUint16(blk[208:210], half.F32ToF16(0.017+float32(row+block)*0.001))
	})
	m, err := UploadQ6KMatrixRows(raw, inDim, outDim)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Free()
	x := syntheticQKInput(batch * inDim)
	got := make([]float32, batch*outDim)
	if err := GemvQ6KBatch(got, x, batch, m); err != nil {
		t.Fatal(err)
	}
	assertQKBatchClose(t, got, x, raw, batch, inDim, outDim, 210, dequantQ6KTest, 2e-4)
}

func TestUploadQKMatrixRowsValidation(t *testing.T) {
	if _, err := UploadQ5KMatrixRows(nil, 255, 1); err == nil {
		t.Fatal("accepted unaligned Q5_K input")
	}
	if _, err := UploadQ6KMatrixRows(make([]byte, 209), 256, 1); err == nil {
		t.Fatal("accepted short Q6_K row")
	}
	if err := GemvQ5KBatchToBuffer(nil, nil, 1, nil); err == nil {
		t.Fatal("accepted nil Q5_K buffers")
	}
}

func syntheticQKRaw(outDim, inDim, blockSize int, fill func([]byte, int, int)) []byte {
	blocks := inDim / 256
	raw := make([]byte, outDim*blocks*blockSize)
	for row := 0; row < outDim; row++ {
		for block := 0; block < blocks; block++ {
			off := (row*blocks + block) * blockSize
			fill(raw[off:off+blockSize], row, block)
		}
	}
	return raw
}

func syntheticQKInput(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*17)%37-18) * 0.013
	}
	return x
}

func assertQKBatchClose(t *testing.T, got, x []float32, raw []byte, batch, inDim, outDim, blockSize int, dequant func([]byte, int) []float32, tolerance float32) {
	t.Helper()
	rowBytes := (inDim / 256) * blockSize
	for row := 0; row < outDim; row++ {
		w := dequant(raw[row*rowBytes:(row+1)*rowBytes], inDim)
		for b := 0; b < batch; b++ {
			var want float32
			for i, weight := range w {
				want += weight * x[b*inDim+i]
			}
			value := got[b*outDim+row]
			if diff := float32(math.Abs(float64(value - want))); diff > tolerance*maxFloat32(1, float32(math.Abs(float64(want)))) {
				t.Fatalf("batch=%d row=%d got=%g want=%g diff=%g", b, row, value, want, diff)
			}
		}
	}
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func dequantQ5KTest(raw []byte, n int) []float32 {
	out := make([]float32, n)
	for b := 0; b < n/256; b++ {
		blk := raw[b*176:]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := half.F16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		sc, qh, ql := blk[4:16], blk[16:48], blk[48:176]
		var scales, mins [8]float32
		for j := 0; j < 4; j++ {
			scales[j], mins[j] = float32(sc[j]&63)*d, float32(sc[j+4]&63)*dmin
		}
		for j := 4; j < 8; j++ {
			k := j - 4
			scales[j] = float32((sc[j+4]&15)|((sc[k]>>6)<<4)) * d
			mins[j] = float32((sc[j+4]>>4)|((sc[k+4]>>6)<<4)) * dmin
		}
		for group := 0; group < 4; group++ {
			for i := 0; i < 32; i++ {
				q0, q1 := int(ql[group*32+i]&15), int(ql[group*32+i]>>4)
				if qh[i]&(1<<uint(group*2)) != 0 {
					q0 += 16
				}
				if qh[i]&(1<<uint(group*2+1)) != 0 {
					q1 += 16
				}
				base := b*256 + group*64
				out[base+i] = scales[group*2]*float32(q0) - mins[group*2]
				out[base+32+i] = scales[group*2+1]*float32(q1) - mins[group*2+1]
			}
		}
	}
	return out
}

func dequantQ6KTest(raw []byte, n int) []float32 {
	out := make([]float32, n)
	for b := 0; b < n/256; b++ {
		blk := raw[b*210:]
		ql, qh, sc := blk[:128], blk[128:192], blk[192:208]
		d := half.F16ToF32(binary.LittleEndian.Uint16(blk[208:210]))
		for i := 0; i < 256; i++ {
			halfBlock, group, lane := i/128, (i%128)/32, i%32
			qlByte := ql[halfBlock*64+(group%2)*32+lane]
			low := qlByte & 15
			if group >= 2 {
				low = qlByte >> 4
			}
			high := (qh[halfBlock*32+lane] >> uint(group*2)) & 3
			q := int8(low|(high<<4)) - 32
			scale := int8(sc[halfBlock*8+(lane/16)+group*2])
			out[b*256+i] = d * float32(scale) * float32(q)
		}
	}
	return out
}
