package nvidia

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
	"unsafe"
)

func TestPTXTrainingOfflineDeviceScale(t *testing.T) {
	oldOK, oldCtx := gpuOK, gpuCtx
	oldSet, oldSync, oldLaunch := cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel
	defer func() {
		gpuOK, gpuCtx = oldOK, oldCtx
		cuCtxSetCurrent, cuCtxSynchronize, cuLaunchKernel = oldSet, oldSync, oldLaunch
	}()
	gpuOK, gpuCtx = true, 123
	cuCtxSetCurrent = func(CUcontext) CUresult { return CUDA_SUCCESS }
	cuCtxSynchronize = func() CUresult { return CUDA_SUCCESS }
	makeDriver := func() *PTXResidentDriver {
		return &PTXResidentDriver{Operator: &TrainingF32{dualModule: &PTXModule{handle: 1, functions: map[string]CUfunction{"training_adamw_scaled": 22}}}}
	}
	b := make([]train.ResidentBuffer, 6)
	for i := range b {
		size := 257 * 4
		if i == 5 {
			size = 16
		}
		b[i] = &ptxResidentBuffer{buffer: &Buffer{Ptr: CUdeviceptr(4096 * (i + 1)), Size: size}}
	}
	p, err := train.NewAdamWStep(.01, .9, .999, 1e-8, .01, .9, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	calls := 0
	cuLaunchKernel = func(fn CUfunction, gx, gy, gz, bx, by, bz, shared uint32, stream uintptr, args, extra unsafe.Pointer) CUresult {
		calls++
		a := unsafe.Slice((*unsafe.Pointer)(args), 15)
		if fn != 22 || gx != 2 || gy != 1 || gz != 1 || bx != 256 || by != 1 || bz != 1 || *(*uint32)(a[6]) != 257 {
			t.Fatal("launch shape")
		}
		for i := 0; i < 6; i++ {
			if *(*CUdeviceptr)(a[i]) != b[i].(*ptxResidentBuffer).buffer.Ptr {
				t.Fatal("buffer ABI", i)
			}
		}
		values := []float32{p.LearningRate, p.Beta1, p.Beta2, p.Epsilon, p.WeightDecay, p.Bias1, p.Bias2, p.EMADecay}
		for i, v := range values {
			if *(*float32)(a[i+7]) != v {
				t.Fatal("constant ABI", i)
			}
		}
		return CUDA_SUCCESS
	}
	validate := func(d *PTXResidentDriver, x []train.ResidentBuffer, p train.AdamWStep, n int) error {
		return d.ValidateScaledUpdate(context.Background(), x[0], x[1], x[2], x[3], x[4], x[5], n, p)
	}
	update := func(d *PTXResidentDriver) error {
		return d.UpdateScaled(context.Background(), b[0], b[1], b[2], b[3], b[4], b[5], 257, p)
	}
	d := makeDriver()
	if err = validate(d, b, p, 257); err != nil || calls != 0 {
		t.Fatal("preflight", err)
	}
	if err = update(d); err != nil || calls != 1 {
		t.Fatal(err)
	}
	for i := 0; i < 6; i++ {
		for j := 0; j < i; j++ {
			x := append([]train.ResidentBuffer(nil), b...)
			x[i] = x[j]
			if validate(d, x, p, 257) == nil {
				t.Fatal("alias", i, j)
			}
		}
	}
	for _, n := range []int{0, 1<<26 + 1} {
		if validate(d, b, p, n) == nil {
			t.Fatal("extent")
		}
	}
	q := p
	q.GradientScale = .5
	if validate(d, b, q, 257) == nil {
		t.Fatal("double scale")
	}
	clip := b[5].(*ptxResidentBuffer).buffer
	clip.Size = 12
	if validate(d, b, p, 257) == nil {
		t.Fatal("short clip")
	}
	clip.Size = 16
	for _, bad := range []*PTXResidentDriver{nil, {}, {Operator: &TrainingF32{}}} {
		if validate(bad, b, p, 257) == nil {
			t.Fatal("owner")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if d.UpdateScaled(ctx, b[0], b[1], b[2], b[3], b[4], b[5], 257, p) == nil || calls != 1 {
		t.Fatal("cancel")
	}
	for _, syncFail := range []bool{false, true} {
		d = makeDriver()
		cuLaunchKernel = func(CUfunction, uint32, uint32, uint32, uint32, uint32, uint32, uint32, uintptr, unsafe.Pointer, unsafe.Pointer) CUresult {
			calls++
			if !syncFail {
				return 201
			}
			return CUDA_SUCCESS
		}
		cuCtxSynchronize = func() CUresult {
			if syncFail {
				return 201
			}
			return CUDA_SUCCESS
		}
		if update(d) == nil || d.Operator.poison == nil {
			t.Fatal("uncertain failure")
		}
		before := calls
		if update(d) == nil || calls != before {
			t.Fatal("retry")
		}
	}
}
