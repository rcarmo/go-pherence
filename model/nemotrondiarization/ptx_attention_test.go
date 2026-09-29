package nemotrondiarization

import (
	"fmt"
	"math"
	"os"
	"testing"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Qualifies exact-key, unmasked PTX RoPE and attention at released layer-1
// shapes. The CPU output projection only makes the result comparable to the
// independent PyTorch attention fixture; it is not a resident layer test.
func TestReleasedLayer1PTXAttentionOperatorsParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	if !ptx.SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	layer0, err := LoadLayer0Complete(file)
	if err != nil {
		t.Fatal(err)
	}
	layer1, err := LoadLayer1Attention(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	stacked := readStackingFixture(t, "testdata/jfk_stacking_transformers_5_18.f32.gz", 138*projectedWidth)
	for _, rows := range []int{16, 138} {
		t.Run(fmt.Sprintf("rows=%d", rows), func(t *testing.T) {
			hidden, err := layer0.ForwardOffline(stacked[:rows*projectedWidth], rows)
			if err != nil {
				t.Fatal(err)
			}
			q, k, v, err := layer1.qkv.Project(hidden, rows)
			if err != nil {
				t.Fatal(err)
			}
			if rows == 138 {
				for _, candidate := range []struct {
					name   string
					values []float32
				}{{"q", q}, {"k", k}, {"v", v}} {
					ref := readStackingFixture(t, "testdata/jfk_full_layer1_"+candidate.name+".f32.gz", len(candidate.values))
					var max, sum float64
					var outside int
					for i, value := range candidate.values {
						d := math.Abs(float64(value - ref[i]))
						max = math.Max(max, d)
						sum += d
						if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || d > 3e-4+2e-5*math.Abs(float64(ref[i])) {
							outside++
						}
					}
					mean := sum / float64(len(ref))
					t.Logf("%s projection vs PyTorch max_abs=%g mean_abs=%g outside=%d", candidate.name, max, mean, outside)
					if outside != 0 || mean > 2e-5 {
						t.Fatalf("%s projection differs from PyTorch", candidate.name)
					}
				}
			}
			originalQ, originalK, originalV := append([]float32(nil), q...), append([]float32(nil), k...), append([]float32(nil), v...)
			const half = diarizationHeadWidth / 2
			freqs := make([]float32, rows*half*2)
			for row := 0; row < rows; row++ {
				for d := 0; d < half; d++ {
					angle := float32(row) * float32(1/math.Pow(10000, float64(2*d)/diarizationHeadWidth))
					freqs[(row*half+d)*2] = float32(math.Cos(float64(angle)))
					freqs[(row*half+d)*2+1] = float32(math.Sin(float64(angle)))
				}
			}
			bQ, bK, bV, bFreq := ptx.NewDevBufFrom(q), ptx.NewDevBufFrom(k), ptx.NewDevBufFrom(v), ptx.NewDevBufFrom(freqs)
			defer bQ.Free()
			defer bK.Free()
			defer bV.Free()
			defer bFreq.Free()
			for _, b := range []*ptx.DevBuf{bQ, bK, bV, bFreq} {
				if err := b.EnsureGPU(); err != nil {
					t.Fatal(err)
				}
			}
			bMixed, err := ptx.NewDevBufGPU(len(q))
			if err != nil {
				t.Fatal(err)
			}
			defer bMixed.Free()
			for _, b := range []*ptx.DevBuf{bQ, bK} {
				if err := ptx.RoPEPartialSequenceBuffer(b.GPUBuffer(), bFreq.GPUBuffer(), rows, 0, diarizationHeads, diarizationHeadWidth, half); err != nil {
					t.Fatal(err)
				}
			}
			for _, item := range []struct {
				name     string
				b        *ptx.DevBuf
				original []float32
			}{{"q", bQ, originalQ}, {"k", bK, originalK}} {
				rotated := make([]float32, len(item.original))
				if err := item.b.GPUBuffer().Download(rotated); err != nil {
					t.Fatal(err)
				}
				var max float64
				for row := 0; row < rows; row++ {
					for head := 0; head < diarizationHeads; head++ {
						for d := 0; d < half; d++ {
							base := row*projectedWidth + head*diarizationHeadWidth + d
							c, s := freqs[(row*half+d)*2], freqs[(row*half+d)*2+1]
							a, b := item.original[base], item.original[base+half]
							max = math.Max(max, math.Abs(float64(rotated[base]-(a*c-b*s))))
							max = math.Max(max, math.Abs(float64(rotated[base+half]-(b*c+a*s))))
						}
					}
				}
				t.Logf("%s PTX RoPE max_abs=%g", item.name, max)
				if max > 1e-5 {
					t.Fatalf("%s PTX RoPE drift", item.name)
				}
			}
			if err := ptx.WhisperAttentionFullOnlineBuffer(bMixed.GPUBuffer(), bQ.GPUBuffer(), bK.GPUBuffer(), bV.GPUBuffer(), rows, rows, diarizationHeads, diarizationHeadWidth, 1.0/8.0); err != nil {
				t.Fatal(err)
			}
			if err := ptx.SyncErr(); err != nil {
				t.Fatal(err)
			}
			mixed := make([]float32, len(q))
			if err := bMixed.GPUBuffer().Download(mixed); err != nil {
				t.Fatal(err)
			}
			for _, check := range []struct {
				b    *ptx.DevBuf
				want []float32
			}{{bV, originalV}, {bFreq, freqs}} {
				got := make([]float32, len(check.want))
				if err := check.b.GPUBuffer().Download(got); err != nil {
					t.Fatal(err)
				}
				for i := range got {
					if got[i] != check.want[i] {
						t.Fatalf("operator mutated its read-only input at %d", i)
					}
				}
			}
			for _, check := range []struct{ got, want []float32 }{{q, originalQ}, {k, originalK}, {v, originalV}} {
				for i := range check.got {
					if check.got[i] != check.want[i] {
						t.Fatalf("operator mutated caller input at %d", i)
					}
				}
			}
			attention := make([]float32, len(mixed))
			if !diarizationDenseMLP(attention, mixed, layer1.outWeight, layer1.outPacked, rows, projectedWidth, projectedWidth) {
				t.Fatal("CPU output projection failed")
			}
			for i := range attention {
				attention[i] += layer1.outBias[i%projectedWidth]
			}
			name := "testdata/jfk_layer1_attention.f32.gz"
			if rows == 138 {
				name = "testdata/jfk_full_layer1_attention.f32.gz"
			}
			ref := readStackingFixture(t, name, len(attention))
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range attention {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			mean := sumAbs / float64(len(attention))
			t.Logf("max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
			if outside != 0 || mean > 2e-5 {
				t.Fatal("PTX RoPE/attention differs from PyTorch")
			}
		})
	}
}
