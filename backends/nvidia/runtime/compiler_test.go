package nvidia

import (
	"math"
	"testing"
)

func TestCompilerSiLUMul(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}

	spec := FusedSiLUMulSpec()
	k, err := Compile(spec)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}

	n := 1024
	a := NewDevBuf(n)
	b := NewDevBuf(n)
	out := NewDevBuf(n)
	for i := range a.Data() {
		a.Data()[i] = float32(i)*0.01 - 5.0
		b.Data()[i] = float32(i)*0.005 + 0.5
	}

	// CPU reference
	cpuOut := make([]float32, n)
	for i := 0; i < n; i++ {
		x := a.Data()[i]
		s := x / (1.0 + float32(math.Exp(float64(-x))))
		cpuOut[i] = s * b.Data()[i]
	}

	// GPU via compiled kernel
	a.ToGPU()
	b.ToGPU()
	out.ToGPU()
	if !k.Launch(n, a.GPUPtr(), b.GPUPtr(), out.GPUPtr()) {
		t.Fatal("JIT SiLU*Mul launch failed")
	}
	Sync()
	gpuOut := out.Data()

	maxDiff := float32(0)
	for i := 0; i < n; i++ {
		d := float32(math.Abs(float64(gpuOut[i] - cpuOut[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("JIT SiLU*Mul %d: maxDiff=%e", n, maxDiff)
	if maxDiff > 0.01 {
		t.Fatalf("drift: %e (gpu[0]=%v cpu[0]=%v)", maxDiff, gpuOut[0], cpuOut[0])
	}
}

func TestCompilerAdd(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}

	// Use batch-compiled kernel
	InitAllKernels()
	k := jitAdd
	if k == nil {
		t.Skip("JIT Add kernel not available")
	}

	n := 512
	a := NewDevBuf(n)
	b := NewDevBuf(n)
	out := NewDevBuf(n)
	for i := range a.Data() {
		a.Data()[i] = float32(i) * 0.1
		b.Data()[i] = float32(n-i) * 0.1
	}

	a.ToGPU()
	b.ToGPU()
	out.ToGPU()
	if !k.Launch(n, a.GPUPtr(), b.GPUPtr(), out.GPUPtr()) {
		t.Fatal("JIT Add launch failed")
	}
	Sync()

	d := out.Data()
	for i := 0; i < n; i++ {
		want := a.cpu[i] + b.cpu[i]
		if math.Abs(float64(d[i]-want)) > 0.001 {
			t.Fatalf("add[%d]=%v want %v", i, d[i], want)
		}
	}
	t.Log("JIT Add OK")
}

func TestCompilerCache(t *testing.T) {
	if !SgemmReady() {
		t.Skip("no GPU")
	}

	// Compile same spec twice — should hit cache
	spec1 := FusedSiLUMulSpec()
	k1, _ := Compile(spec1)

	spec2 := FusedSiLUMulSpec()
	k2, _ := Compile(spec2)

	if k1 != k2 {
		t.Fatal("cache miss for identical spec")
	}
	t.Log("kernel cache: OK")
}

func TestCompilerValidationRejectsMalformedSpecs(t *testing.T) {
	if _, err := Compile(nil); err == nil {
		t.Fatal("Compile accepted nil spec")
	}
	if _, err := Compile(&KernelSpec{Name: "bad", NumBufs: 1}); err == nil {
		t.Fatal("Compile accepted spec with no nodes")
	}
	if _, err := Compile(&KernelSpec{Name: "bad", NumBufs: 1, Nodes: []*KNode{nil}}); err == nil {
		t.Fatal("Compile accepted nil node")
	}
	if _, err := Compile(&KernelSpec{Name: "bad", NumBufs: 1, Nodes: []*KNode{{Op: KOpLoad, BufIdx: 1}}}); err == nil {
		t.Fatal("Compile accepted out-of-range buffer index")
	}
	badInput := &KNode{Op: KOpAdd, Inputs: []*KNode{nil}}
	if _, err := Compile(&KernelSpec{Name: "bad", NumBufs: 1, Nodes: []*KNode{badInput}}); err == nil {
		t.Fatal("Compile accepted nil node input")
	}
	var k *CompiledKernel
	if k.Launch(1) {
		t.Fatal("nil compiled kernel launch reported success")
	}
	if (&CompiledKernel{Fn: 1, NumBufs: 1, GridDiv: 0, BlockSz: 256}).Launch(1, &Buffer{Ptr: 1, Size: 4}) {
		t.Fatal("zero grid divisor launch reported success")
	}
	if (&CompiledKernel{Fn: 1, NumBufs: 1, GridDiv: 1, BlockSz: 256}).Launch(1) {
		t.Fatal("missing buffer launch reported success")
	}
	if (&CompiledKernel{Fn: 1, NumBufs: 1, GridDiv: 1, BlockSz: 256}).Launch(1, &Buffer{Ptr: 0, Size: 4}) {
		t.Fatal("zero pointer launch reported success")
	}
	maxU32 := int(^uint32(0))
	if (&CompiledKernel{Fn: 1, NumBufs: 1, GridDiv: 1, BlockSz: 256}).Launch(maxU32+1, &Buffer{Ptr: 1, Size: maxU32}) {
		t.Fatal("oversized element count launch reported success")
	}
	if (&CompiledKernel{Fn: 1, NumBufs: 1, GridDiv: 1, BlockSz: maxU32 + 1}).Launch(1, &Buffer{Ptr: 1, Size: 4}) {
		t.Fatal("oversized block size launch reported success")
	}
}

func TestJITKeysIncludeConstantsEdgesAndABI(t *testing.T) {
	makeSpec := func(value float32, reverse bool) *KernelSpec {
		a := &KNode{Op: KOpLoad, BufIdx: 0}
		b := &KNode{Op: KOpConst, ConstVal: value}
		inputs := []*KNode{a, b}
		if reverse {
			inputs = []*KNode{b, a}
		}
		sub := &KNode{Op: KOpSub, Inputs: inputs}
		store := &KNode{Op: KOpStore, BufIdx: 1, Inputs: []*KNode{sub}}
		return &KernelSpec{Name: "key", NumBufs: 2, Nodes: []*KNode{a, b, sub, store}}
	}
	a, b, c := makeSpec(1, false), makeSpec(2, false), makeSpec(1, true)
	if specKey(a) == specKey(b) || specKey(a) == specKey(c) {
		t.Fatal("cache key ignores constants or edges")
	}
	clone := cloneKernelSpec(a)
	genPTX(clone)
	if a.Nodes[0].RegName != "" {
		t.Fatal("codegen mutated caller graph")
	}
	clone.NumBufs = 3
	if specKey(a) == specKey(clone) {
		t.Fatal("cache key ignores ABI")
	}
	for _, spec := range []*KernelSpec{{Name: "bad", NumBufs: 1, Nodes: []*KNode{{Op: KOpAdd}}}, {Name: "bad", NumBufs: 1, HasReduce: true, Nodes: []*KNode{{Op: KOpLoad}}}, {Name: "bad", NumBufs: 9, Nodes: []*KNode{{Op: KOpLoad}}}} {
		if err := validateKernelSpec(spec); err == nil {
			t.Fatal("malformed ABI/codegen accepted")
		}
	}
	k := &CompiledKernel{Fn: 1, NumBufs: 1, GridDiv: 256, BlockSz: 256}
	if k.Launch(1, &Buffer{Ptr: 1, Size: 4}, &Buffer{Ptr: 2, Size: 4}) {
		t.Fatal("extra buffer shifts N kernel argument")
	}
}
