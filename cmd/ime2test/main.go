package main

import (
	"fmt"
	"os"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/inference"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

// extractQ4KToINT8 extracts Q4_K nibbles directly to INT8 without F32 intermediate.
// Each Q4_K block (144 bytes) produces 256 int8 values (range 0-15).
func extractQ4KToINT8(data []byte, rows, cols int) ([]int8, []float32, []float32) {
	blockSize := 256
	bytesPerBlock := 144
	blocksPerRow := cols / blockSize

	out := make([]int8, rows*cols)
	scales := make([]float32, rows*blocksPerRow*8) // per sub-block scale
	mins := make([]float32, rows*blocksPerRow*8)   // per sub-block min

	for row := 0; row < rows; row++ {
		for blk := 0; blk < blocksPerRow; blk++ {
			offset := (row*blocksPerRow + blk) * bytesPerBlock
			b := data[offset : offset+bytesPerBlock]

			d := fp16(b[0], b[1])
			dmin := fp16(b[2], b[3])

			// Decode scales
			var sc, mn [8]float32
			for i := 0; i < 4; i++ {
				sc[i] = float32(b[4+i] & 63)
				mn[i] = float32(b[8+i] & 63)
			}
			for i := 0; i < 4; i++ {
				sc[i+4] = float32((b[4+i]>>6) | ((b[12+i]&0xf)<<2))
				mn[i+4] = float32((b[8+i]>>6) | ((b[12+i]>>4)<<2))
			}

			scaleIdx := (row*blocksPerRow + blk) * 8
			for sb := 0; sb < 8; sb++ {
				scales[scaleIdx+sb] = d * sc[sb]
				mins[scaleIdx+sb] = dmin * mn[sb]
			}

			// Extract nibbles to INT8 (0-15 range, stored as signed int8)
			base := row*cols + blk*blockSize
			for sb := 0; sb < 8; sb++ {
				qOff := 16 + sb*16
				for j := 0; j < 16; j++ {
					q := b[qOff+j]
					out[base+sb*32+j] = int8(q & 0xf)
					out[base+sb*32+j+16] = int8(q >> 4)
				}
			}
		}
	}
	return out, scales, mins
}

func fp16(lo, hi byte) float32 {
	// same as before - convert fp16 to f32
	h := uint16(lo) | uint16(hi)<<8
	if h == 0 { return 0 }
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	if exp == 0 {
		for mant&0x400 == 0 { mant <<= 1; exp-- }
		exp++; mant &= 0x3ff
	}
	exp = exp + (127 - 15)
	bits := (sign << 31) | (exp << 23) | (mant << 13)
	return *(*float32)(unsafe.Pointer(&bits))
}

func main() {
	g, _ := gguf.Open(os.Args[1])
	wqT, _ := g.TensorByName("blk.0.attn_q.weight")
	wqData, _ := g.Raw(wqT)
	rows := int(wqT.Shape[1])
	cols := int(wqT.Shape[0])
	fmt.Fprintf(os.Stderr, "wq: %dx%d (%d bytes)\n", rows, cols, len(wqData))

	// Extract Q4K directly to INT8
	t0 := time.Now()
	wqI8, _, _ := extractQ4KToINT8(wqData, rows, cols)
	extractTime := time.Since(t0)
	fmt.Fprintf(os.Stderr, "Extract Q4K→INT8: %.3fms\n", float64(extractTime.Microseconds())/1000)

	// Pack tiles
	t1 := time.Now()
	colsPad := ((cols + 7) / 8) * 8
	rowsPad := ((rows + 3) / 4) * 4
	wqI8Pad := make([]int8, rowsPad*colsPad)
	for r := 0; r < rows; r++ {
		copy(wqI8Pad[r*colsPad:r*colsPad+cols], wqI8[r*cols:(r+1)*cols])
	}
	wqPacked := ime2.PackTiles(wqI8Pad, rowsPad, colsPad)
	packTime := time.Since(t1)
	fmt.Fprintf(os.Stderr, "PackTiles: %.3fms\n", float64(packTime.Microseconds())/1000)

	// Test activation (synthetic)
	x := make([]float32, cols)
	for i := range x { x[i] = 0.01 * float32(i%100-50) }

	// Pre-pack activation
	t2 := time.Now()
	actPacked, actScale := inference.PackActivation(padF32(x, colsPad), colsPad)
	packActTime := time.Since(t2)
	fmt.Fprintf(os.Stderr, "PackActivation: %.3fms\n", float64(packActTime.Microseconds())/1000)

	// Run matmul
	t3 := time.Now()
	cI32 := make([]int32, rowsPad*4)
	ime2.GemmINT8Packed(rowsPad, 4, colsPad, wqPacked, actPacked, cI32)
	matmulTime := time.Since(t3)
	fmt.Fprintf(os.Stderr, "GemmINT8Packed (%dx%d): %.3fms\n", rowsPad, colsPad, float64(matmulTime.Microseconds())/1000)

	// Total pipeline
	fmt.Fprintf(os.Stderr, "\nTotal matvec (extract+pack+compute): %.3fms\n",
		float64((extractTime+packTime+packActTime+matmulTime).Microseconds())/1000)
	fmt.Fprintf(os.Stderr, "  vmadot only: %.3fms (%.1f%% of total)\n",
		float64(matmulTime.Microseconds())/1000,
		float64(matmulTime)*100/float64(extractTime+packTime+packActTime+matmulTime))

	_ = actScale
	_ = cI32
}

func padF32(src []float32, n int) []float32 {
	if len(src) >= n { return src[:n] }
	dst := make([]float32, n)
	copy(dst, src)
	return dst
}
