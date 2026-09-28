package tensor

import (
	"fmt"
	"runtime"
	"slices"
	"testing"
	"unsafe"
)

func assertPanicsWithMessage(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic")
		}
		if got := fmt.Sprint(r); got != want {
			t.Fatalf("panic=%q want %q", got, want)
		}
	}()
	fn()
}

func TestFromOwnedFloat32Valid(t *testing.T) {
	tests := []struct {
		name  string
		data  []float32
		shape []int
	}{
		{name: "scalar", data: []float32{42}, shape: []int{}},
		{name: "empty", data: []float32{}, shape: []int{0}},
		{name: "multidim", data: []float32{1, 2, 3, 4, 5, 6}, shape: []int{2, 3}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := append([]float32(nil), tc.data...)
			shape := append([]int(nil), tc.shape...)
			x := FromOwnedFloat32(data, shape)

			if x.uop == nil || x.uop.Op != OpBuffer {
				t.Fatalf("uop=%#v want OpBuffer", x.uop)
			}
			if x.uop.buf == nil {
				t.Fatal("expected realized buffer")
			}
			if x.uop.buf.DType != Float32 {
				t.Fatalf("dtype=%v want %v", x.uop.buf.DType, Float32)
			}
			if x.uop.buf.Length != len(tc.data) {
				t.Fatalf("length=%d want %d", x.uop.buf.Length, len(tc.data))
			}
			if got := x.Shape(); !slices.Equal(got, tc.shape) {
				t.Fatalf("shape=%v want %v", got, tc.shape)
			}
			if got := x.Data(); !slices.Equal(got, tc.data) {
				t.Fatalf("data=%v want %v", got, tc.data)
			}
			if len(tc.data) > 0 {
				if unsafe.SliceData(x.Data()) != unsafe.SliceData(data) {
					t.Fatal("FromOwnedFloat32 copied payload")
				}
			}
			for i := range shape {
				shape[i] = 99 + i
			}
			if got := x.Shape(); !slices.Equal(got, tc.shape) {
				t.Fatalf("shape changed after caller mutation: %v want %v", got, tc.shape)
			}
			if arg, ok := x.uop.Arg.([]int); !ok || !slices.Equal(arg, tc.shape) {
				t.Fatalf("uop arg=%v want %v", x.uop.Arg, tc.shape)
			}
		})
	}
}

func TestFromOwnedFloat32ShapeValidation(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	assertPanicsWithMessage(t, "shape mismatch", func() {
		_ = FromOwnedFloat32([]float32{1}, []int{-1})
	})
	assertPanicsWithMessage(t, "shape mismatch", func() {
		_ = FromOwnedFloat32([]float32{1, 2, 3}, []int{2, 2})
	})
	assertPanicsWithMessage(t, "shape mismatch", func() {
		_ = FromOwnedFloat32([]float32{1}, []int{maxInt/2 + 1, 3})
	})
}

func TestFromOwnedFloat32ShapeCallerMutationUnaffected(t *testing.T) {
	shape := []int{2, 3}
	x := FromOwnedFloat32([]float32{1, 2, 3, 4, 5, 6}, shape)
	shape[0], shape[1] = 7, 8
	if got := x.Shape(); !slices.Equal(got, []int{2, 3}) {
		t.Fatalf("shape=%v want [2 3]", got)
	}
	if arg := x.uop.Arg.([]int); !slices.Equal(arg, []int{2, 3}) {
		t.Fatalf("uop arg=%v want [2 3]", arg)
	}
}

func TestFromFloat32StillCopiesPayload(t *testing.T) {
	data := []float32{1, 2, 3, 4}
	x := FromFloat32(data, []int{2, 2})
	if unsafe.SliceData(x.Data()) == unsafe.SliceData(data) {
		t.Fatal("FromFloat32 unexpectedly aliased caller payload")
	}
	data[0], data[1] = 99, 100
	if got := x.Data(); !slices.Equal(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("data=%v want [1 2 3 4]", got)
	}
}

func makeOwnedTensorForLiveness() (*Tensor, uintptr) {
	data := []float32{1, 2, 3, 4}
	ptr := uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	x := FromOwnedFloat32(data, []int{2, 2})
	return x, ptr
}

func TestFromOwnedFloat32KeepsBackingAliveAndArithmeticWorks(t *testing.T) {
	x, wantPtr := makeOwnedTensorForLiveness()

	for i := 0; i < 8; i++ {
		runtime.GC()
		scratch := make([][]byte, 16)
		for j := range scratch {
			scratch[j] = make([]byte, 1<<15)
		}
	}

	got := x.Data()
	if !slices.Equal(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("data=%v want [1 2 3 4]", got)
	}
	if gotPtr := uintptr(unsafe.Pointer(unsafe.SliceData(got))); gotPtr != wantPtr {
		t.Fatalf("data pointer=%#x want %#x", gotPtr, wantPtr)
	}

	y := x.Add(Ones([]int{2, 2}))
	if got := y.Data(); !slices.Equal(got, []float32{2, 3, 4, 5}) {
		t.Fatalf("sum=%v want [2 3 4 5]", got)
	}
	if got := x.Data(); !slices.Equal(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("source data changed after arithmetic: %v", got)
	}

	runtime.KeepAlive(x)
}

func BenchmarkFromFloat32Constructors(b *testing.B) {
	template := make([]float32, 256)
	for i := range template {
		template[i] = float32(i)
	}
	shape := []int{32, 8}
	b.SetBytes(int64(len(template) * 4))

	b.Run("owned", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// Clone a bounded template per iteration so the owned constructor can
			// legally take exclusive ownership without retaining an input reused by
			// another iteration. The same setup cost is charged to both variants.
			src := append([]float32(nil), template...)
			_ = FromOwnedFloat32(src, shape)
		}
	})

	b.Run("copy", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			src := append([]float32(nil), template...)
			_ = FromFloat32(src, shape)
		}
	})
}
