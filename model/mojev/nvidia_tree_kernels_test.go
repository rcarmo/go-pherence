package mojev

import (
	"math"
	"os"
	"reflect"
	"testing"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/nvidia/ptx"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestMoJevPTXTreeKernels(t *testing.T) {
	if os.Getenv("GO_PHERENCE_MOJEV_NVIDIA") != "1" {
		t.Skip("set GO_PHERENCE_MOJEV_NVIDIA=1")
	}
	names := []string{"mj_conv", "mj_tree_conv", "mj_delta", "mj_tree_delta", "mj_qk_norm_rope", "mj_tree_qk_norm_rope", "mj_attention", "mj_tree_attention"}
	module, err := nvidia.LoadPTXFunctions(ptx.MoJev, names)
	if err != nil {
		t.Fatal(err)
	}
	defer module.Close()
	const total, ns, nq = 10, 2, 2
	ends := []int{5, 8, 10}
	meta := make([]uint32, total*4)
	if err := fillGPUTree(meta, total, ns, nq, ends); err != nil {
		t.Fatal(err)
	}
	run := func(name string, ids []int, tree bool) []float32 {
		t.Helper()
		var buffers []*nvidia.Buffer
		defer func() {
			for _, b := range buffers {
				b.Free()
			}
		}()
		upload := func(x []float32) *nvidia.Buffer {
			b, e := nvidia.Malloc(len(x))
			if e != nil {
				t.Fatal(e)
			}
			buffers = append(buffers, b)
			if e = b.Upload(x); e != nil {
				t.Fatal(e)
			}
			return b
		}
		// Token values depend only on original row index, so compacting a branch
		// physically removes siblings without changing its mathematical inputs.
		values := func(width, salt int) []float32 {
			x := make([]float32, len(ids)*width)
			for i, id := range ids {
				for j := 0; j < width; j++ {
					x[i*width+j] = float32((id*width+j+salt)%23-11) * 0.01
				}
			}
			return x
		}
		x := upload(values(6144, 0))
		qg := upload(values(4096, 7))
		k := upload(values(512, 13))
		v := upload(values(512, 17))
		alpha := upload(values(16, 3))
		beta := upload(values(16, 5))
		cw := make([]float32, 4*6144)
		for i := range cw {
			cw[i] = float32(i%11-5) * 0.02
		}
		convWeight := upload(cw)
		av := make([]float32, 16)
		for i := range av {
			av[i] = -0.2
		}
		a, dt := upload(av), upload(make([]float32, 16))
		norm := upload(make([]float32, 256))
		md, e := nvidia.Malloc(len(meta))
		if e != nil {
			t.Fatal(e)
		}
		buffers = append(buffers, md)
		if e = md.UploadUint32(meta); e != nil {
			t.Fatal(e)
		}
		n := int32(len(ids))
		prefix, stateLen, questionLen := int32(ns+nq), int32(ns), int32(nq)
		heads, stride, headStride := int32(8), int32(4096), int32(512)
		eps := float32(1e-6)
		outWidth, blocks := 2048, 256
		if name == "mj_conv" {
			outWidth = 6144
			blocks = (len(ids)*6144 + 255) / 256
		}
		if name == "mj_qk_norm_rope" {
			outWidth = 4096
			blocks = len(ids) * 8
		}
		if name == "mj_attention" {
			blocks = len(ids) * 8
		}
		out := upload(make([]float32, len(ids)*outWidth))
		var args []unsafe.Pointer
		switch name {
		case "mj_conv":
			args = []unsafe.Pointer{unsafe.Pointer(&x.Ptr), unsafe.Pointer(&convWeight.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&n)}
		case "mj_delta":
			args = []unsafe.Pointer{unsafe.Pointer(&x.Ptr), unsafe.Pointer(&alpha.Ptr), unsafe.Pointer(&beta.Ptr), unsafe.Pointer(&dt.Ptr), unsafe.Pointer(&a.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&n)}
		case "mj_qk_norm_rope":
			args = []unsafe.Pointer{unsafe.Pointer(&qg.Ptr), unsafe.Pointer(&norm.Ptr), unsafe.Pointer(&n), unsafe.Pointer(&heads), unsafe.Pointer(&stride), unsafe.Pointer(&headStride), unsafe.Pointer(&eps)}
			out = qg
		case "mj_attention":
			args = []unsafe.Pointer{unsafe.Pointer(&qg.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&n)}
			if tree {
				args = append(args, unsafe.Pointer(&md.Ptr), unsafe.Pointer(&prefix))
			} else {
				args = append(args, unsafe.Pointer(&stateLen), unsafe.Pointer(&questionLen))
			}
		}
		kernel := name
		if tree {
			kernel = "mj_tree_" + name[3:]
			if name != "mj_attention" {
				args = append(args, unsafe.Pointer(&md.Ptr))
			}
		}
		if e = nvidia.LaunchKernel(module.Function(kernel), uint32(blocks), 1, 1, 256, 1, 1, 0, args...); e != nil {
			t.Fatal(e)
		}
		got := make([]float32, len(ids)*outWidth)
		if e = out.Download(got); e != nil {
			t.Fatal(e)
		}
		return got
	}
	all := make([]int, total)
	for i := range all {
		all[i] = i
	}
	for _, name := range []string{"mj_conv", "mj_delta", "mj_qk_norm_rope", "mj_attention"} {
		t.Run(name, func(t *testing.T) {
			got := run(name, all, true)
			width := len(got) / total
			start := ns + nq
			for _, end := range ends {
				ids := []int{0, 1, 2, 3}
				for i := start; i < end; i++ {
					ids = append(ids, i)
				}
				want := run(name, ids, false)
				for i, id := range ids {
					for j := 0; j < width; j++ {
						x, y := got[id*width+j], want[i*width+j]
						if math.IsNaN(float64(x)) || x != y {
							t.Fatal("tree differs from compact branch", name, id, j, x, y)
						}
					}
				}
				start = end
			}
			if again := run(name, all, true); !reflect.DeepEqual(again, got) {
				t.Fatal("tree nondeterminism")
			}
		})
	}
}
