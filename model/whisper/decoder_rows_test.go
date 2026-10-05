package whisper

import (
	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
	"runtime"
	"sync"
	"testing"
)

func TestDecoderLinearRowsExact(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {17, 513}, {1280, 1280}, {1280, 5120}, {5120, 1280}, {33, 519}} {
		k, n := shape[0], shape[1]
		x := make([]float32, k)
		w := make([]float32, k*n)
		for i := range x {
			x[i] = float32(i%29-14) * .0073
		}
		for i := range w {
			w[i] = float32(i%31-15) * .0137
		}
		for _, bn := range []int{0, 3, n} {
			b := make([]float32, bn)
			for i := range b {
				b[i] = float32(i%11-5) * .017
			}
			ref := make([]float32, n)
			for o := range ref {
				v := simdrt.Sdot(x, w[o*k:(o+1)*k])
				if o < len(b) {
					v += b[o]
				}
				ref[o] = v
			}
			for _, workers := range []int{0, 1, 2, 4, 7} {
				out := make([]float32, n+2)
				out[0] = 3
				out[n+1] = 7
				decoderLinearRowsInto(out[1:n+1], x, w, b, k, n, workers)
				for i, v := range ref {
					if math.Float32bits(out[i+1]) != math.Float32bits(v) {
						t.Fatal("bits", shape, bn, workers, i)
					}
				}
				if out[0] != 3 || out[n+1] != 7 {
					t.Fatal("guard")
				}
			}
		}
	}
}
func TestDecoderRowWorkersExplicit(t *testing.T) {
	old := linearWorkers
	linearWorkers = 4
	defer func() { linearWorkers = old }()
	for _, flag := range []string{"", "0", "true", "yes", "2"} {
		t.Setenv(envDecoderParallelRows, flag)
		if decoderRowWorkers(1280) != 1 {
			t.Fatal("default", flag)
		}
	}
	t.Setenv(envDecoderParallelRows, "1")
	if decoderRowWorkers(511) != 1 || decoderRowWorkers(512) != min(4, runtime.GOMAXPROCS(0)) {
		t.Fatal("gate/budget")
	}
	linearWorkers = 1
	if decoderRowWorkers(512) != 1 {
		t.Fatal("workerbudget")
	}
}
func TestDecoderLinearIntoOptInExact(t *testing.T) {
	old := linearWorkers
	linearWorkers = 4
	defer func() { linearWorkers = old }()
	k, n := 33, 513
	x := make([]float32, k)
	w := make([]float32, k*n)
	b := make([]float32, 3)
	for i := range x {
		x[i] = float32(i%7-3) * .0037
	}
	for i := range w {
		w[i] = float32(i%11-5) * .0173
	}
	b[0] = math.Float32frombits(0x80000000)
	b[1] = .7
	b[2] = -.7
	t.Setenv(envDecoderParallelRows, "0")
	want := make([]float32, n)
	linearInto(want, x, w, b, k, n)
	t.Setenv(envDecoderParallelRows, "1")
	got := make([]float32, n)
	linearInto(got, x, w, b, k, n)
	for i, v := range want {
		if math.Float32bits(v) != math.Float32bits(got[i]) {
			t.Fatal("dispatch bits", i)
		}
	}
}

func TestDecoderLinearRowsConcurrentOwners(t *testing.T) {
	var wg sync.WaitGroup
	for task := 0; task < 4; task++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			x := make([]float32, 17)
			w := make([]float32, 513*17)
			out := make([]float32, 513)
			decoderLinearRowsInto(out, x, w, nil, 17, 513, 4)
		}()
	}
	wg.Wait()
}
func TestDecoderLinearRowsExtentPanicSynchronous(t *testing.T) {
	for _, sizes := range [][3]int{{0, 17, 513 * 17}, {513, 0, 513 * 17}, {513, 17, 0}} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("extentmustpanicincaller")
				}
			}()
			decoderLinearRowsInto(make([]float32, sizes[0]), make([]float32, sizes[1]), make([]float32, sizes[2]), nil, 17, 513, 4)
		}()
	}
}
