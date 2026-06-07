package k3

import (
	"runtime"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/k3/aipool"
)

func makeBenchQ4X32(M, K int) q4kQ41x32 {
	subs := K / 32
	raw := make([]int8, M*K)
	scales := make([]float32, M*subs)
	mins := make([]float32, M*subs)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			raw[m*K+k] = int8((m*7 + k*5 + 3) & 15)
		}
		for sb := 0; sb < subs; sb++ {
			scales[m*subs+sb] = 0.001 + float32((m+sb)%17)*0.0004
			mins[m*subs+sb] = 0.005 + float32((m*3+sb)%13)*0.0009
		}
	}
	return repackQ4KToQ41x32(M, K, raw, scales, mins)
}

func makeBenchQRows4(K int) [4][]byte {
	var rows [4][]byte
	for r := 0; r < 4; r++ {
		act := make([]float32, K)
		for k := range act {
			act[k] = float32(((r+1)*(k%31) - 15)) / 9.0
		}
		rows[r] = quantizeQ8Blocks32Bytes(act)
	}
	return rows
}

func BenchmarkK3I8I4M1x4VsM4(b *testing.B) {
	K := 1024
	subs := K / 32
	w := makeBenchQ4X32(32, K)
	rows := makeBenchQRows4(K)
	packedA := packQ8RowsM4(rows, subs)
	outM1 := make([]float32, 4*32)
	outM4 := make([]float32, 4*32)
	b.Run("four_m1", func(b *testing.B) {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		aipool.RegisterAIThread(8)
		for i := 0; i < b.N; i++ {
			for r := 0; r < 4; r++ {
				k3I8I4M1((*byte)(unsafe.Pointer(&rows[r][0])), (*byte)(unsafe.Pointer(&w.BData[0])), &outM1[r*32], subs, 32)
			}
		}
	})
	b.Run("m4_dispatch", func(b *testing.B) {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		aipool.RegisterAIThread(8)
		for i := 0; i < b.N; i++ {
			q4Handled := k3I8I4((*byte)(unsafe.Pointer(&packedA[0])), (*byte)(unsafe.Pointer(&w.BData[0])), &outM4[0], 4, 32, subs, 32)
			if q4Handled != 4 {
				b.Fatalf("handled=%d", q4Handled)
			}
		}
	})
}
