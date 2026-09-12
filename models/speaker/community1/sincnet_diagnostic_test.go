package community1

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"math"
	"os"
	"testing"
)

func TestSincNetReductionDiagnostic(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_SINCNET_TRACE")
	if path == "" {
		t.Skip("diagnostic trace not supplied")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	z, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	data, err = io.ReadAll(io.LimitReader(z, 32<<20))
	z.Close()
	if err != nil {
		t.Fatal(err)
	}
	var trace struct {
		Cases []struct {
			Stride int
			Kind   string
			Stages []struct {
				Input             []float32
				InputFrames       int `json:"input_frames"`
				Convolution       []float32
				ConvolutionFrames int `json:"convolution_frames"`
				Pooled            []float32
				PooledFrames      int `json:"pooled_frames"`
			}
		}
	}
	if err := json.Unmarshal(data, &trace); err != nil {
		t.Fatal(err)
	}
	f := loadSincNetFixture(t)
	diff := func(a, b []float32) (float64, int) {
		maxerr := float64(0)
		exact := 0
		for i, v := range a {
			e := math.Abs(float64(v - b[i]))
			maxerr = math.Max(maxerr, e)
			if v == b[i] {
				exact++
			}
		}
		return maxerr, exact
	}
	for ci, c := range trace.Cases {
		if c.Kind != "wave" {
			continue
		}
		model, err := NewSincNet(context.Background(), c.Stride, f.Weights)
		if err != nil {
			t.Fatal(err)
		}
		for stage, b := range c.Stages {
			in, out, k, stride := 1, 80, 251, c.Stride
			weight := f.Filters
			var bias []float32
			if stage > 0 {
				in = 80
				if stage == 2 {
					in = 60
				}
				out, k, stride = 60, 5, 1
				weight = f.Weights.Conv[stage-1].Weight
				bias = f.Weights.Conv[stage-1].Bias
			}
			for _, mode := range []SincNetMode{SincNetScalar, SincNetSIMD} {
				conv, _, err := sincNetConvolve(context.Background(), b.Input, in, b.InputFrames, weight, bias, out, k, stride, mode)
				if err != nil {
					t.Fatal(err)
				}
				e, n := diff(conv, b.Convolution)
				t.Logf("stride%d stage%d exact-input/filter mode%d conv err%g exact%d/%d", c.Stride, stage, mode, e, n, len(conv))
			}
			for _, kind := range []string{"float64", "fma32", "unroll4", "unroll8"} {
				conv := make([]float32, len(b.Convolution))
				for o := 0; o < out; o++ {
					for t := 0; t < b.ConvolutionFrames; t++ {
						var sum64 float64
						var sum32 float32
						var partial [8]float32
						for i := 0; i < in; i++ {
							for j := 0; j < k; j++ {
								a, w := b.Input[i*b.InputFrames+t*stride+j], weight[(o*in+i)*k+j]
								if kind == "float64" {
									sum64 += float64(a) * float64(w)
								} else if kind == "fma32" {
									sum32 = float32(math.FMA(float64(a), float64(w), float64(sum32)))
								} else {
									lanes := 4
									if kind == "unroll8" {
										lanes = 8
									}
									lane := (i*k + j) % lanes
									partial[lane] = float32(math.FMA(float64(a), float64(w), float64(partial[lane])))
								}
							}
						}
						if kind == "float64" {
							sum32 = float32(sum64)
						} else if kind == "unroll4" || kind == "unroll8" {
							for _, v := range partial {
								sum32 += v
							}
						}
						if bias != nil {
							sum32 += bias[o]
						}
						conv[o*b.ConvolutionFrames+t] = sum32
					}
				}
				e, n := diff(conv, b.Convolution)
				t.Logf("stride%d stage%d %s conv err%g exact%d/%d", c.Stride, stage, kind, e, n, len(conv))
			}
			x := append([]float32(nil), b.Pooled...)
			if err := sincNetNorm(context.Background(), x, out, b.PooledFrames, f.Weights.Norm[stage]); err != nil {
				t.Fatal(err)
			}
			for i, v := range x {
				if v < 0 {
					x[i] = v * .01
				}
			}
			e, n := diff(x, f.Cases[ci].Boundaries[stage+1].Values)
			t.Logf("stride%d stage%d exact-pool norm err%g exact%d/%d", c.Stride, stage, e, n, len(x))
		}
		e, _ := diff(model.filters, f.Filters)
		t.Logf("filters error%g", e)
	}
}
