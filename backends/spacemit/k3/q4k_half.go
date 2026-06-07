package k3

import (
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// q4kBlockMatVecAIPackedHalf uses native 8x16 vmadot tiles with a separate
// activation scale/sum for each 16-element half of a Q4_K 32-element subblock.
// This is not the llama.cpp q8 block contract; it is an accuracy-oriented fused
// Q4_K diagnostic that preserves exact Q4_K min correction more closely.
func q4kBlockMatVecAIPackedHalf(M, K int, wPacked []int8, scales, mins []float32, act []float32, out []float32, pool *AIWorkerPool) {
	if K%32 != 0 || M%8 != 0 {
		panic("q4kBlockMatVecAIPackedHalf: unsupported shape")
	}
	subsPerRow := K / 32
	halves := K / 16
	actI8 := make([]int8, K)
	actScale := make([]float32, halves)
	actSum := make([]int32, halves)
	for h := 0; h < halves; h++ {
		base := h * 16
		var maxAbs float32
		for i := 0; i < 16; i++ {
			a := act[base+i]
			if a < 0 {
				a = -a
			}
			if a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs == 0 {
			continue
		}
		actScale[h] = maxAbs / 127.0
		s := float32(127.0) / maxAbs
		var sum int32
		for i := 0; i < 16; i++ {
			v := act[base+i] * s
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			q := int8(v)
			actI8[base+i] = q
			sum += int32(q)
		}
		actSum[h] = sum
	}
	pool.Run(func(workerID, nWorkers int) {
		rowStart := (workerID * M / nWorkers / 8) * 8
		rowEnd := ((workerID + 1) * M / nWorkers / 8) * 8
		if workerID == nWorkers-1 {
			rowEnd = M
		}
		tilesPerRow := K / 16
		var aTile [128]int8
		for row := rowStart; row < rowEnd; row += 8 {
			for r := 0; r < 8 && row+r < M; r++ {
				out[row+r] = 0
			}
			for sb := 0; sb < subsPerRow; sb++ {
				for half := 0; half < 2; half++ {
					h := sb*2 + half
					as := actScale[h]
					if as == 0 {
						continue
					}
					kBase := h * 16
					for r := 0; r < 8; r++ {
						copy(aTile[r*16:(r+1)*16], actI8[kBase:kBase+16])
					}
					var acc [64]int32
					wOff := ((row/8)*tilesPerRow + kBase/16) * 128
					ime2.VmadotKLoop1024((*byte)(unsafe.Pointer(&wPacked[wOff])), (*byte)(unsafe.Pointer(&aTile[0])), &acc[0], 16)
					for r := 0; r < 8 && row+r < M; r++ {
						idx := (row+r)*subsPerRow + sb
						out[row+r] += float32(acc[r*8])*scales[idx]*as - float32(actSum[h])*mins[idx]*as
					}
				}
			}
		}
	})
}
