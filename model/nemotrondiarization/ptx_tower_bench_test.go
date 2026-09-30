package nemotrondiarization

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Bounded resident-tower diagnostic: model load is excluded. Each PTX call
// includes CPU layer 0, one H2D input, all 30 layers, final norm, D2H output
// and bounded scratch cleanup. Weight upload and teardown are separate metrics.
func BenchmarkReleasedPTXAudioTower(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	if !ptx.SgemmReady() {
		b.Skip("CUDA unavailable")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	model, err := LoadOfflineAudioTower(file)
	if err != nil {
		b.Fatal(err)
	}
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	for _, rows := range []int{13, 103, 138} {
		b.Run(fmt.Sprintf("rows=%d", rows), func(b *testing.B) {
			input := benchmarkTowerInput(b, rows)
			ctx := context.Background()
			started := time.Now()
			tower, err := NewPTXAudioTower(model, maxPreparedDiarizationRows)
			if err != nil {
				b.Fatal(err)
			}
			setup := time.Since(started)
			defer tower.Close()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := tower.ForwardRows(ctx, input, rows); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			started = time.Now()
			tower.Close()
			b.ReportMetric(float64(setup.Microseconds()), "setup-us")
			b.ReportMetric(float64(time.Since(started).Microseconds()), "close-us")
		})
	}
}
