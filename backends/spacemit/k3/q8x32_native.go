package k3

import (
	"math"
	"unsafe"
)

type q8Q80x32 struct {
	M, K  int
	BData []byte // per N32 group, per K32: fp16 d[32] + int8 qs[32][32] = 1088B
	Valid bool
}

func repackF32ToQ80x32(M, K int, f32 []float32) q8Q80x32 {
	if M%32 != 0 || K%32 != 0 || len(f32) < M*K {
		return q8Q80x32{}
	}
	subs := K / 32
	groups := M / 32
	out := q8Q80x32{M: M, K: K, BData: make([]byte, groups*subs*(64+1024)), Valid: true}
	var qtmp [32]int8
	for rg := 0; rg < groups; rg++ {
		for sb := 0; sb < subs; sb++ {
			base := (rg*subs + sb) * 1088
			for r := 0; r < 32; r++ {
				row := rg*32 + r
				var maxAbs float32
				for k := 0; k < 32; k++ {
					v := f32[row*K+sb*32+k]
					if v < 0 {
						v = -v
					}
					if v > maxAbs {
						maxAbs = v
					}
				}
				d := float32(0)
				if maxAbs != 0 {
					d = maxAbs / 127.0
				}
				bits := f32ToF16Bits(d)
				out.BData[base+r*2+0] = byte(bits)
				out.BData[base+r*2+1] = byte(bits >> 8)
				if d == 0 {
					for k := range qtmp {
						qtmp[k] = 0
					}
				} else {
					inv := 1.0 / d
					for k := 0; k < 32; k++ {
						q := float32(math.Round(float64(f32[row*K+sb*32+k] * inv)))
						if q > 127 {
							q = 127
						} else if q < -128 {
							q = -128
						}
						qtmp[k] = int8(q)
					}
				}
				copy(out.BData[base+64+r*32:base+64+(r+1)*32], unsafe.Slice((*byte)(unsafe.Pointer(&qtmp[0])), 32))
			}
		}
	}
	return out
}

//go:noescape
func k3I8I8M1(a *byte, b *byte, c *float32, kBlks int, nBlks int)

//go:noescape
func k3I8I8M1Groups(a *byte, b *byte, c *float32, kBlks int, nGroups int)

//go:noescape
func k3I8I8M4(a *byte, b *byte, c *float32, kBlks int, ldcBytes int)

func q8I8Dispatcher(a, b *byte, c *float32, countM, countN, kBlks, ldc int) int {
	if countM >= 4 {
		for n := 0; n < countN; n += 32 {
			k3I8I8M4(a, (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(b))+uintptr((n/32)*kBlks*1088))), (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(c))+uintptr(n*4))), kBlks, ldc*4)
		}
		return 4
	}
	for n := 0; n < countN; n += 32 {
		k3I8I8M1(a, (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(b))+uintptr((n/32)*kBlks*1088))), (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(c))+uintptr(n*4))), kBlks, 32)
	}
	return 1
}

func q8Q80x32MatVec4Native(w q8Q80x32, acts [4][]float32, outs [4][]float32) bool {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 {
		return false
	}
	for i := 0; i < 4; i++ {
		if len(acts[i]) < w.K || len(outs[i]) < w.M {
			return false
		}
	}
	subs := w.K / 32
	groups := w.M / 32
	packedA := quantizeQ8RowsM4Bytes(acts, subs)
	aPtr := (*byte)(unsafe.Pointer(&packedA[0]))
	for rg := 0; rg < groups; rg++ {
		var tmp [4 * 32]float32
		bPtr := (*byte)(unsafe.Pointer(&w.BData[rg*subs*1088]))
		handled := q8I8Dispatcher(aPtr, bPtr, &tmp[0], 4, 32, subs, 32)
		if handled != 4 {
			return false
		}
		for r := 0; r < 4; r++ {
			copy(outs[r][rg*32:(rg+1)*32], tmp[r*32:(r+1)*32])
		}
	}
	return true
}

func q8Q80x32MatVecNative(w q8Q80x32, act []float32, out []float32, pool *AIWorkerPool) bool {
	if !w.Valid || w.K%32 != 0 || w.M%32 != 0 || len(out) < w.M {
		return false
	}
	quantA := quantizeQ8Blocks32Bytes(act)
	subs := w.K / 32
	groups := w.M / 32
	pairBarrier := &q4kPairBarrier{}
	pool.Run(func(workerID, nWorkers int) {
		quantPtr := (*byte)(unsafe.Pointer(&quantA[0]))
		tcmSlice := getTCMSlice(workerID)
		if len(tcmSlice) >= len(quantA) {
			copy(tcmSlice[:len(quantA)], quantA)
			quantPtr = (*byte)(unsafe.Pointer(&tcmSlice[0]))
		}
		bBytes := subs * 1088
		bOff := (len(quantA) + 63) &^ 63
		if int8TCMBWaveOn && nWorkers%2 == 0 && groups >= nWorkers && (groups%nWorkers)%2 == 0 && len(tcmSlice) >= bOff+bBytes {
			pair := workerID / 2
			rg := workerID
			if workerID%2 == 0 {
				copyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[rg*subs*1088:(rg+1)*subs*1088])
			}
			pairBarrier.wait(pair)
			if workerID%2 != 0 {
				copyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[rg*subs*1088:(rg+1)*subs*1088])
			}
			bPtr := (*byte)(unsafe.Pointer(&tcmSlice[bOff]))
			for ; rg < groups; rg += nWorkers {
				if workerID%2 != 0 {
					pairBarrier.wait(pair)
				}
				k3I8I8M1Groups(quantPtr, bPtr, &out[rg*32], subs, 1)
				if workerID%2 == 0 {
					pairBarrier.wait(pair)
				}
				nextRg := rg + nWorkers
				if nextRg < groups {
					copyTCMBytes(tcmSlice[bOff:bOff+bBytes], w.BData[nextRg*subs*1088:(nextRg+1)*subs*1088])
				}
			}
			return
		}
		gStart := workerID * groups / nWorkers
		gEnd := (workerID + 1) * groups / nWorkers
		if gEnd > gStart {
			k3I8I8M1Groups(quantPtr, (*byte)(unsafe.Pointer(&w.BData[gStart*subs*1088])), &out[gStart*32], subs, gEnd-gStart)
		}
	})
	return true
}
