package audio

import "testing"

// 100s of tiled JFK PCM supplied in 5s calls; excludes WAV loading, includes
// the owned per-call feature output but does not retain it across calls.
func BenchmarkNemotronMelStreamHundredSeconds(b *testing.B) {
	pcm := nemotronJFK(b)
	chunk := make([]float32, nemotronStreamMaxChunk)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var stream NemotronMelStream
		for part := 0; part < 20; part++ {
			for j := range chunk {
				chunk[j] = pcm[(part*len(chunk)+j)%len(pcm)]
			}
			if _, err := stream.AppendPCM(chunk); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := stream.Finish(); err != nil {
			b.Fatal(err)
		}
	}
}
