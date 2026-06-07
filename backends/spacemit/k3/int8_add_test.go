package k3

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/inference"
)

func TestVmadotI8GroupsAddMatchesSeparateAdd(t *testing.T) {
	M, K := 64, 1024
	f32 := make([]float32, M*K)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			f32[m*K+k] = float32(((m*13+k*7)%37)-18) / 11.0
		}
	}
	i8 := make([]int8, M*K)
	wScale := inference.QuantizeF32ToINT8(f32, i8)
	wPacked := ime2.PackTiles1024(i8, M, K)
	actF := make([]float32, K)
	for k := range actF {
		actF[k] = float32(((k*5)%31)-15) / 9.0
	}
	actI8 := make([]int8, K)
	actScale := quantizeToI8(actF, actI8)
	actPacked := make([]int8, 8*K)
	broadcastPack1024Into(actI8, K, actPacked)
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	want := make([]float32, M)
	got := make([]float32, M)
	for i := range want {
		want[i] = float32(i%17) / 7.0
		got[i] = want[i]
	}
	prod := make([]float32, M)
	GemmAIPooled(M, K, wPacked, actPacked, wScale, actScale, prod, pool)
	for i := range want {
		want[i] += prod[i]
	}
	GemmAIPooledAdd(M, K, wPacked, actPacked, wScale, actScale, got, pool)
	var maxDiff float64
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("maxDiff=%.6f", maxDiff)
	if maxDiff > 1e-5 {
		t.Fatalf("maxDiff %.6f > tolerance", maxDiff)
	}
}

func TestVmadotI8GroupsAddTCMBWaveMatchesDirectAdd(t *testing.T) {
	old := int8TCMBWaveOn
	defer func() { int8TCMBWaveOn = old }()
	M, K := 1024, 3072
	f32 := make([]float32, M*K)
	for m := 0; m < M; m++ {
		for k := 0; k < K; k++ {
			f32[m*K+k] = float32(((m*5+k*11)%43)-21) / 17.0
		}
	}
	i8 := make([]int8, M*K)
	wScale := inference.QuantizeF32ToINT8(f32, i8)
	wPacked := ime2.PackTiles1024(i8, M, K)
	actF := make([]float32, K)
	for k := range actF { actF[k] = float32(((k*7)%31)-15) / 10.0 }
	actI8 := make([]int8, K)
	actScale := quantizeToI8(actF, actI8)
	actPacked := make([]int8, 8*K)
	broadcastPack1024Into(actI8, K, actPacked)
	pool := NewAIWorkerPool(6)
	defer pool.Close()
	direct := make([]float32, M)
	wave := make([]float32, M)
	for i := range direct { direct[i] = float32(i%19) / 13.0; wave[i] = direct[i] }
	int8TCMBWaveOn = false
	GemmAIPooledAdd(M, K, wPacked, actPacked, wScale, actScale, direct, pool)
	int8TCMBWaveOn = true
	GemmAIPooledAdd(M, K, wPacked, actPacked, wScale, actScale, wave, pool)
	var maxDiff float64
	for i := range direct { if d := math.Abs(float64(direct[i]-wave[i])); d > maxDiff { maxDiff = d } }
	t.Logf("maxDiff=%.6f", maxDiff)
	if maxDiff > 1e-5 { t.Fatalf("maxDiff %.6f > tolerance", maxDiff) }
}
