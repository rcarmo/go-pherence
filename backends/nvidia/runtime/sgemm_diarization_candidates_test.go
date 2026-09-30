package nvidia

import (
	"fmt"
	"math"
	"testing"
)

// The six F32 projection shapes used by the 30-layer diarization tower. This
// compares candidate kernels against the independent F64 matrix product and
// measures only the already-resident GPU operation. Request timings must still
// include frontend, setup, transfers, teardown, head and speaker cache.
const diarizationIntermediateTest = 2048

func TestSgemmDiarizationCandidateParity(t *testing.T) {
	if !SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	for _, s := range []struct{ m, n, k int }{
		{13, 512, 512}, {103, 512, 512}, {138, 512, 512},
		{13, diarizationIntermediateTest, 512}, {103, diarizationIntermediateTest, 512},
		{103, 512, diarizationIntermediateTest},
	} {
		t.Run(fmt.Sprintf("m=%d/n=%d/k=%d", s.m, s.n, s.k), func(t *testing.T) {
			a, b := make([]float32, s.m*s.k), make([]float32, s.k*s.n)
			for i := range a {
				a[i] = float32((i*17)%37-18) / 31
			}
			for i := range b {
				b[i] = float32((i*23)%41-20) / 67
			}
			want := make([]float32, s.m*s.n)
			for i := 0; i < s.m; i++ {
				for j := 0; j < s.n; j++ {
					var sum float64
					for k := 0; k < s.k; k++ {
						sum += float64(a[i*s.k+k]) * float64(b[k*s.n+j])
					}
					want[i*s.n+j] = float32(sum)
				}
			}
			up := func(x []float32) *Buffer {
				d, e := Malloc(len(x))
				if e != nil {
					t.Fatal(e)
				}
				t.Cleanup(d.Free)
				if e = d.Upload(x); e != nil {
					t.Fatal(e)
				}
				return d
			}
			da, db := up(a), up(b)
			out, e := Malloc(len(want))
			if e != nil {
				t.Fatal(e)
			}
			t.Cleanup(out.Free)
			for _, v := range []struct {
				name string
				fn   CUfunction
				kind string
			}{{"oracle", sgemmOracleFn, "oracle"}, {"reg2", sgemmReg2Fn, "reg2"}, {"skinny", sgemmSkinnyFn, "skinny"}} {
				t.Run(v.name, func(t *testing.T) {
					if v.fn == 0 {
						t.Skip("candidate unavailable")
					}
					if e := launchSgemmVariant(v.fn, v.kind, s.m, s.n, s.k, 1, da, db, out); e != nil {
						t.Fatal(e)
					}
					if e := SyncErr(); e != nil {
						t.Fatal(e)
					}
					got := make([]float32, len(want))
					if e := out.Download(got); e != nil {
						t.Fatal(e)
					}
					var max, sum float64
					var outside int
					for i, x := range got {
						d := math.Abs(float64(x - want[i]))
						max = math.Max(max, d)
						sum += d
						if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || d > 3e-4+2e-5*math.Abs(float64(want[i])) {
							outside++
						}
					}
					mean := sum / float64(len(got))
					t.Logf("max_abs=%g mean_abs=%g outside=%d", max, mean, outside)
					if outside != 0 || mean > 2e-5 {
						t.Fatal("candidate differs from F64 matrix product")
					}
				})
			}
		})
	}
}

func BenchmarkSgemmDiarizationCandidates(b *testing.B) {
	if !SgemmReady() {
		b.Skip("CUDA unavailable")
	}
	for _, s := range []struct{ m, n, k int }{{13, 512, 512}, {103, 512, 512}, {103, 2048, 512}, {103, 512, 2048}} {
		b.Run(fmt.Sprintf("m=%d/n=%d/k=%d", s.m, s.n, s.k), func(b *testing.B) {
			da, e := Malloc(s.m * s.k)
			if e != nil {
				b.Fatal(e)
			}
			defer da.Free()
			db, e := Malloc(s.k * s.n)
			if e != nil {
				b.Fatal(e)
			}
			defer db.Free()
			out, e := Malloc(s.m * s.n)
			if e != nil {
				b.Fatal(e)
			}
			defer out.Free()
			for _, v := range []struct {
				name string
				fn   CUfunction
				kind string
			}{{"oracle", sgemmOracleFn, "oracle"}, {"reg2", sgemmReg2Fn, "reg2"}, {"skinny", sgemmSkinnyFn, "skinny"}} {
				if v.fn == 0 {
					continue
				}
				b.Run(v.name, func(b *testing.B) {
					if e := launchSgemmVariant(v.fn, v.kind, s.m, s.n, s.k, 1, da, db, out); e != nil {
						b.Fatal(e)
					}
					if e := SyncErr(); e != nil {
						b.Fatal(e)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if e := launchSgemmVariant(v.fn, v.kind, s.m, s.n, s.k, 1, da, db, out); e != nil {
							b.Fatal(e)
						}
						SyncForTiming()
					}
					b.StopTimer()
				})
			}
		})
	}
}
