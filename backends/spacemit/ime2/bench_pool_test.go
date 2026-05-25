package ime2

import "testing"

func BenchmarkGemmPooled_2048x4x2048(b *testing.B) {
	M, N, K := 2048, 4, 2048
	Ap := PackTiles(make([]int8, M*K), M, K)
	Bp := PackTiles(make([]int8, N*K), N, K)
	C := make([]int32, M*N)
	pool := NewWorkerPool(8)
	defer pool.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmINT8PackedPool(M, N, K, Ap, Bp, C, pool)
	}
}

func BenchmarkGemmSingle_2048x4x2048(b *testing.B) {
	M, N, K := 2048, 4, 2048
	Ap := PackTiles(make([]int8, M*K), M, K)
	Bp := PackTiles(make([]int8, N*K), N, K)
	C := make([]int32, M*N)
	b.ResetTimer()
	for i := 0; i < b.N; i++ { GemmINT8Packed(M, N, K, Ap, Bp, C) }
}

func BenchmarkPoolOverhead(b *testing.B) {
	pool := NewWorkerPool(8)
	defer pool.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pool.Run(func(workerID, nWorkers int) {})
	}
}


func BenchmarkChanRoundtrip(b *testing.B) {
	ch := make(chan struct{}, 1)
	done := make(chan struct{}, 1)
	go func() { for range ch { done <- struct{}{} } }()
	b.ResetTimer()
	for i := 0; i < b.N; i++ { ch <- struct{}{}; <-done }
	close(ch)
}
