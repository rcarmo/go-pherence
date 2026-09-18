package community1

import (
	"context"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestCommunity1TrainedDiagnostic(t *testing.T) {
	if os.Getenv("GO_PHERENCE_TEST_COMMUNITY1_DIAGNOSTIC") != "1" {
		t.Skip("explicit diagnostic")
	}
	dir, mf := trainedSegmentationAssets(t)
	ctx := context.Background()
	src, e := safetensors.Open(filepath.Join(dir, "segmentation.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	m, e := LoadSegmentationSource(ctx, src, mf.Config)
	src.Close()
	if e != nil {
		t.Fatal(e)
	}
	fs, e := safetensors.Open(filepath.Join(dir, "filters.safetensors"))
	if e != nil {
		t.Fatal(e)
	}
	filters := trainedTensor(t, fs, "sincnet.filters")
	fs.Close()
	for _, c := range mf.Cases {
		t.Run(c.Name, func(t *testing.T) {
			o, e := safetensors.Open(filepath.Join(dir, c.File))
			if e != nil {
				t.Fatal(e)
			}
			defer o.Close()
			for stage := 0; stage < 3; stage++ {
				in, out, k, stride := 1, 80, 251, 10
				w := filters
				var bias []float32
				if stage > 0 {
					in, out, k, stride = 80, 60, 5, 1
					if stage == 2 {
						in = 60
					}
					w = m.sincnet.Conv[stage-1].Weight
					bias = m.sincnet.Conv[stage-1].Bias
				}
				x := trainedTensor(t, o, fmt.Sprintf("conv_input.%d", stage))
				n := len(x) / in
				conv, frames, e := sincNetConvolve(ctx, x, in, n, w, bias, out, k, stride, SincNetSIMDFMA)
				if e != nil {
					t.Fatal(e)
				}
				trainedCompare(t, fmt.Sprintf("exact-input-conv.%d", stage), conv, trainedTensor(t, o, fmt.Sprintf("conv.%d", stage)))
				if stage == 0 {
					for i, v := range conv {
						conv[i] = float32(math.Abs(float64(v)))
					}
				}
				pool, _, e := sincNetPool(ctx, conv, out, frames)
				if e != nil {
					t.Fatal(e)
				}
				wantpool := trainedTensor(t, o, fmt.Sprintf("pool.%d", stage))
				trainedCompare(t, fmt.Sprintf("exact-input-pool.%d", stage), pool, wantpool)
				norm := append([]float32(nil), wantpool...)
				if e = sincNetNorm(ctx, norm, out, len(norm)/out, m.sincnet.Norm[stage]); e != nil {
					t.Fatal(e)
				}
				for i, v := range norm {
					if v < 0 {
						norm[i] = v * .01
					}
				}
				trainedCompare(t, fmt.Sprintf("exact-pool-norm.%d", stage), norm, trainedTensor(t, o, fmt.Sprintf("sincnet.%d", stage)))
				if c.Name == "silence-1s" && stage == 1 {
					want := trainedTensor(t, o, "conv.1")
					for ch := 0; ch < out; ch++ {
						min, max := float32(math.Inf(1)), float32(math.Inf(-1))
						for _, v := range want[ch*frames : (ch+1)*frames] {
							if v < min {
								min = v
							}
							if v > max {
								max = v
							}
						}
						if max != min {
							t.Logf("ref conv channel%d nonconstant range%g bias%g", ch, max-min, bias[ch])
						}
					}
				}
			}
		})
	}
}
