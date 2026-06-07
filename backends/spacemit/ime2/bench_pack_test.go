package ime2

import "testing"

func BenchmarkPackTiles_4x1024(b *testing.B) {
	src := make([]int8, 4*1024)
	for i := range src {
		src[i] = int8(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PackTiles(src, 4, 1024)
	}
}

func BenchmarkManualPack_4x1024(b *testing.B) {
	src := make([]int8, 4*1024)
	dst := make([]int8, 4*1024)
	K := 1024
	for i := range src {
		src[i] = int8(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for ki := 0; ki < K; ki += 8 {
			tileBase := (ki / 8) * 32
			for r := 0; r < 4; r++ {
				copy(dst[tileBase+r*8:tileBase+r*8+8], src[r*K+ki:r*K+ki+8])
			}
		}
	}
}
