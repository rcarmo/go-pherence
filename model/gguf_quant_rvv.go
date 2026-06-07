package model

import (
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func QuantGemvRVVBlocks(out, x []float32, w *gguf.QuantMatrix) error {
	if len(out) < w.OutDim || len(x) < w.InDim {
		return fmt.Errorf("quantGemvRVVBlocks %s bad sizes", w.Name)
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > w.OutDim {
		workers = w.OutDim
	}
	if workers < 2 || w.OutDim < 512 {
		var scratch [256]float32
		for row := 0; row < w.OutDim; row++ {
			sum, err := quantDotRowBlocks(w, row, x[:w.InDim], scratch[:])
			if err != nil {
				return err
			}
			out[row] = sum
		}
		return nil
	}
	chunk := (w.OutDim + workers - 1) / workers
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	for start := 0; start < w.OutDim; start += chunk {
		end := start + chunk
		if end > w.OutDim {
			end = w.OutDim
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			var scratch [256]float32
			for row := start; row < end; row++ {
				sum, err := quantDotRowBlocks(w, row, x[:w.InDim], scratch[:])
				if err != nil {
					errCh <- err
					return
				}
				out[row] = sum
			}
		}(start, end)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return err
		}
	}
	return nil
}

func quantDotRowBlocks(w *gguf.QuantMatrix, row int, x []float32, scratch []float32) (float32, error) {
	rowBytes, err := w.RowBytes()
	if err != nil {
		return 0, err
	}
	start := row * rowBytes
	end := start + rowBytes
	if end > len(w.Raw) {
		return 0, fmt.Errorf("%s row %d raw short", w.Name, row)
	}
	raw := w.Raw[start:end]
	switch w.QType {
	case gguf.QuantQ2_K:
		return dotQ2KBlocks(raw, x, scratch, w.InDim)
	case gguf.QuantQ3_K:
		return dotQ3KBlocks(raw, x, scratch, w.InDim)
	case gguf.QuantQ6_K:
		return dotQ6KBlocks(raw, x, scratch, w.InDim)
	case gguf.QuantQ8_0:
		return dotQ8_0Blocks(raw, x, scratch, w.InDim)
	case gguf.QuantF32, gguf.QuantF16:
		// Rare in TinyLlama hot matrices; use existing row dequant + RVV dot.
		if err := w.DequantRowTo(scratch[:w.InDim], row); err != nil {
			return 0, err
		}
		return simd.Sdot(x[:w.InDim], scratch[:w.InDim]), nil
	default:
		return 0, fmt.Errorf("unsupported quant type %d", w.QType)
	}
}

func f16bitsToF32(u uint16) float32 {
	return half.F16ToF32(u)
}

func dotQ2KBlocks(raw []byte, x []float32, scratch []float32, inDim int) (float32, error) {
	const blockElems, blockSize = 256, 84
	if inDim%blockElems != 0 {
		return 0, fmt.Errorf("Q2_K inDim %d not multiple of 256", inDim)
	}
	blocks := inDim / blockElems
	if len(raw) < blocks*blockSize {
		return 0, fmt.Errorf("Q2_K raw short")
	}
	var total float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize:]
		scales := blk[0:16]
		q := blk[16:80]
		d := f16bitsToF32(binary.LittleEndian.Uint16(blk[80:82]))
		minv := f16bitsToF32(binary.LittleEndian.Uint16(blk[82:84]))
		y, is, qoff := 0, 0, 0
		for nn := 0; nn < blockElems; nn += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl := d * float32(sc&0x0F)
				ml := minv * float32(sc>>4)
				for l := 0; l < 16; l++ {
					scratch[y] = dl*float32((q[qoff+l]>>shift)&3) - ml
					y++
				}
				sc = scales[is]
				is++
				dl = d * float32(sc&0x0F)
				ml = minv * float32(sc>>4)
				for l := 0; l < 16; l++ {
					scratch[y] = dl*float32((q[qoff+l+16]>>shift)&3) - ml
					y++
				}
				shift += 2
			}
			qoff += 32
			_ = nn
		}
		xb := b * blockElems
		total += simd.Sdot(x[xb:xb+blockElems], scratch[:blockElems])
	}
	return total, nil
}

func dotQ3KBlocks(raw []byte, x []float32, scratch []float32, inDim int) (float32, error) {
	const blockElems, blockSize = 256, 110
	if inDim%blockElems != 0 {
		return 0, fmt.Errorf("Q3_K inDim %d not multiple of 256", inDim)
	}
	blocks := inDim / blockElems
	if len(raw) < blocks*blockSize {
		return 0, fmt.Errorf("Q3_K raw short")
	}
	const kmask1 uint32 = 0x03030303
	const kmask2 uint32 = 0x0f0f0f0f
	var total float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize:]
		hm, q, s := blk[0:32], blk[32:96], blk[96:108]
		dAll := f16bitsToF32(binary.LittleEndian.Uint16(blk[108:110]))
		aux := [4]uint32{binary.LittleEndian.Uint32(s[0:4]), binary.LittleEndian.Uint32(s[4:8]), binary.LittleEndian.Uint32(s[8:12]), 0}
		tmp := aux[2]
		aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
		aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
		aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
		aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
		var scales [16]int8
		for i := 0; i < 4; i++ {
			u := aux[i]
			scales[4*i+0] = int8(byte(u))
			scales[4*i+1] = int8(byte(u >> 8))
			scales[4*i+2] = int8(byte(u >> 16))
			scales[4*i+3] = int8(byte(u >> 24))
		}
		y, is, m, qoff := 0, 0, byte(1), 0
		for nn := 0; nn < blockElems; nn += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				dl := dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					lo := int8((q[qoff+l] >> shift) & 3)
					if hm[l]&m == 0 {
						lo -= 4
					}
					scratch[y] = dl * float32(lo)
					y++
				}
				dl = dAll * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					lo := int8((q[qoff+l+16] >> shift) & 3)
					if hm[l+16]&m == 0 {
						lo -= 4
					}
					scratch[y] = dl * float32(lo)
					y++
				}
				shift += 2
				m <<= 1
			}
			qoff += 32
			_ = nn
		}
		xb := b * blockElems
		total += simd.Sdot(x[xb:xb+blockElems], scratch[:blockElems])
	}
	return total, nil
}

func dotQ6KBlocks(raw []byte, x []float32, scratch []float32, inDim int) (float32, error) {
	const blockElems, blockSize = 256, 210
	if inDim%blockElems != 0 {
		return 0, fmt.Errorf("Q6_K inDim %d not multiple of 256", inDim)
	}
	blocks := inDim / blockElems
	if len(raw) < blocks*blockSize {
		return 0, fmt.Errorf("Q6_K raw short")
	}
	var total float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize:]
		ql, qh, sc := blk[0:128], blk[128:192], blk[192:208]
		d := f16bitsToF32(binary.LittleEndian.Uint16(blk[208:210]))
		y, qlOff, qhOff, scOff := 0, 0, 0, 0
		for nn := 0; nn < blockElems; nn += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((ql[qlOff+l]&0x0F)|(((qh[qhOff+l]>>0)&3)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0F)|(((qh[qhOff+l]>>2)&3)<<4)) - 32
				q3 := int8((ql[qlOff+l]>>4)|(((qh[qhOff+l]>>4)&3)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((qh[qhOff+l]>>6)&3)<<4)) - 32
				scratch[y+l+0] = d * float32(int8(sc[scOff+is+0])) * float32(q1)
				scratch[y+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
				scratch[y+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
				scratch[y+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
			}
			y += 128
			qlOff += 64
			qhOff += 32
			scOff += 8
			_ = nn
		}
		xb := b * blockElems
		total += simd.Sdot(x[xb:xb+blockElems], scratch[:blockElems])
	}
	return total, nil
}

func dotQ8_0Blocks(raw []byte, x []float32, scratch []float32, inDim int) (float32, error) {
	const blockElems, blockSize = 32, 34
	if inDim%blockElems != 0 {
		return 0, fmt.Errorf("Q8_0 inDim %d not multiple of 32", inDim)
	}
	blocks := inDim / blockElems
	var total float32
	for b := 0; b < blocks; b++ {
		blk := raw[b*blockSize:]
		d := f16bitsToF32(binary.LittleEndian.Uint16(blk[0:2]))
		qs := blk[2:34]
		for i := 0; i < 32; i++ {
			scratch[i] = d * float32(int8(qs[i]))
		}
		xb := b * blockElems
		total += simd.Sdot(x[xb:xb+blockElems], scratch[:blockElems])
	}
	return total, nil
}
