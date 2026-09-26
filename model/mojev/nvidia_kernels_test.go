package mojev

import (
	"math"
	"os"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestMoJevPTXKernels(t *testing.T) {
	if os.Getenv("GO_PHERENCE_MOJEV_NVIDIA") != "1" {
		t.Skip("opt-in PTX kernels")
	}
	if !nvidia.Init() {
		t.Fatal("CUDA unavailable")
	}
	module, e := nvidia.LoadPTXFunctions(ptx.MoJev, []string{"mj_gemm", "mj_norm", "mj_attention", "mj_delta"})
	if e != nil {
		t.Fatal(e)
	}
	defer module.Close()
	g := &NVIDIATextScorer{cpu: &TextScorer{eps: 1e-6}, kernels: map[string]nvidia.CUfunction{}}
	for _, name := range []string{"mj_gemm", "mj_norm", "mj_attention", "mj_delta"} {
		g.kernels[name] = module.Function(name)
	}
	upload := func(x []float32) *nvidia.Buffer {
		b, e := nvidia.Malloc(len(x))
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(b.Free)
		if e = b.Upload(x); e != nil {
			t.Fatal(e)
		}
		return b
	}
	for _, batched := range []bool{false, true} {
		if batched {
			g.commands = make([]nvidia.KernelLaunch, 8)
		} else {
			g.commands = nil
		}
		for _, shape := range [][3]int{{1, 17, 3}, {7, 65, 33}, {33, 67, 35}, {31, 63, 31}, {32, 64, 32}, {63, 127, 65}, {65, 129, 63}, {512, 65, 33}} {
			g.commandCount = 0
			m, n, k := shape[0], shape[1], shape[2]
			a, b := make([]float32, m*k), make([]float32, k*n)
			for i := range a {
				a[i] = float32(i%11-5) * 0.07
			}
			for i := range b {
				b[i] = float32(i%7-3) * 0.03
			}
			da, db, dc := upload(a), upload(b), upload(make([]float32, m*n))
			if e = g.project(da, db, dc, m, k, n); e != nil {
				t.Fatal(e)
			}
			if batched {
				if e = nvidia.LaunchBatch(g.commands[:g.commandCount]); e != nil {
					t.Fatal(e)
				}
			}
			out := make([]float32, m*n)
			if e = dc.Download(out); e != nil {
				t.Fatal(e)
			}
			for r := 0; r < m; r++ {
				for c := 0; c < n; c++ {
					var want float32
					for j := 0; j < k; j++ {
						want += a[r*k+j] * b[j*n+c]
					}
					if math.Abs(float64(out[r*n+c]-want)) > 1e-5 {
						t.Fatal("GEMM tail mismatch", shape, r, c)
					}
				}
			}
		}
	}
	// Exercise repeated shared-memory reduction with unequal node lengths.
	n, ns, nq := int32(7), int32(2), int32(3)
	q, k, v := make([]float32, 7*4096), make([]float32, 7*512), make([]float32, 7*512)
	for i := range v {
		v[i] = float32(i%17-8) * 0.1
	}
	dq, dk, dv, do := upload(q), upload(k), upload(v), upload(make([]float32, 7*2048))
	out := make([]float32, 7*2048)
	for repeat := 0; repeat < 20; repeat++ {
		if e = g.launch("mj_attention", 7*8, unsafe.Pointer(&dq.Ptr), unsafe.Pointer(&dk.Ptr), unsafe.Pointer(&dv.Ptr), unsafe.Pointer(&do.Ptr), unsafe.Pointer(&n), unsafe.Pointer(&ns), unsafe.Pointer(&nq)); e != nil {
			t.Fatal(e)
		}
		if e = do.Download(out); e != nil {
			t.Fatal(e)
		}
		for tkn := 0; tkn < 7; tkn++ {
			end := 7
			if tkn < 2 {
				end = 2
			} else if tkn < 5 {
				end = 5
			}
			for h := 0; h < 8; h++ {
				for d := 0; d < 256; d++ {
					var sum float32
					for j := 0; j < end; j++ {
						sum += v[j*512+(h/4)*256+d]
					}
					want := sum / float32(end) * 0.5
					if math.Abs(float64(out[tkn*2048+h*256+d]-want)) > 1e-6 {
						t.Fatal("attention ancestor/reduction", repeat, tkn, h, d)
					}
				}
			}
		}
	}
	// Two recurrent timesteps exercise row ownership and launch-local reset.
	qkv := make([]float32, 2*6144)
	for tkn := 0; tkn < 2; tkn++ {
		for i := 0; i < 2048; i++ {
			qkv[tkn*6144+i] = 0.01
			qkv[tkn*6144+2048+i] = 0.02
			qkv[tkn*6144+4096+i] = float32((i%7)+1) * 0.1
		}
	}
	dqkv, da, db, ddt, decay, dout := upload(qkv), upload(make([]float32, 32)), upload(make([]float32, 32)), upload(make([]float32, 16)), upload(make([]float32, 16)), upload(make([]float32, 2*2048))
	tokens := int32(2)
	delta := make([]float32, 2*2048)
	for repeat := 0; repeat < 2; repeat++ {
		if e = g.launch("mj_delta", 256, unsafe.Pointer(&dqkv.Ptr), unsafe.Pointer(&da.Ptr), unsafe.Pointer(&db.Ptr), unsafe.Pointer(&ddt.Ptr), unsafe.Pointer(&decay.Ptr), unsafe.Pointer(&dout.Ptr), unsafe.Pointer(&tokens)); e != nil {
			t.Fatal(e)
		}
		if e = dout.Download(delta); e != nil {
			t.Fatal(e)
		}
		for i := 0; i < 2048; i++ {
			state := float32(0)
			for tkn := 0; tkn < 2; tkn++ {
				v := qkv[tkn*6144+4096+i]
				memory := 128 * state * 0.02
				state += 0.02 * (v - memory) * 0.5
				want := 128 * state * 0.01 * 0.088388347648
				if math.Abs(float64(delta[tkn*2048+i]-want)) > 1e-6 {
					t.Fatal("delta mismatch", repeat, tkn, i)
				}
			}
		}
	}
	if e = module.Close(); e != nil {
		t.Fatal(e)
	}
	if module.Function("mj_gemm") != 0 {
		t.Fatal("closed module handle")
	}
	if e = module.Close(); e != nil {
		t.Fatal(e)
	}
}
