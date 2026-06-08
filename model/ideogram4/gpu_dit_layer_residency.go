package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

type ditLayerGPUResidency struct {
	qkv   *nvidia.GPUFP8E4M3Linear
	o     *nvidia.GPUFP8E4M3Linear
	w1    *nvidia.GPUFP8E4M3Linear
	w2    *nvidia.GPUFP8E4M3Linear
	w3    *nvidia.GPUFP8E4M3Linear
	adaln *nvidia.GPUFP8E4M3Linear
}

func uploadDiTLayerGPU(l DiTLayer) (*ditLayerGPUResidency, error) {
	if !gpuFP8Enabled() {
		return nil, nil
	}
	if !nvidia.Available() {
		return nil, fmt.Errorf("nvidia runtime unavailable")
	}
	r := &ditLayerGPUResidency{}
	upload := func(dst **nvidia.GPUFP8E4M3Linear, name string, lin *FP8Linear) error {
		if lin == nil {
			return fmt.Errorf("nil %s linear", name)
		}
		w, err := nvidia.UploadFP8E4M3Linear(lin.weight.Weight, lin.weight.Scale, lin.weight.Bias, lin.weight.OutDim, lin.weight.InDim)
		if err != nil {
			return fmt.Errorf("upload %s: %w", name, err)
		}
		*dst = w
		return nil
	}
	if err := upload(&r.qkv, "qkv", l.QKV); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.o, "o", l.O); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.w1, "w1", l.W1); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.w2, "w2", l.W2); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.w3, "w3", l.W3); err != nil {
		r.Free()
		return nil, err
	}
	if err := upload(&r.adaln, "adaln", l.AdaLN); err != nil {
		r.Free()
		return nil, err
	}
	return r, nil
}

func (r *ditLayerGPUResidency) Free() {
	if r == nil {
		return
	}
	for _, w := range []*nvidia.GPUFP8E4M3Linear{r.qkv, r.o, r.w1, r.w2, r.w3, r.adaln} {
		w.Free()
	}
	r.qkv, r.o, r.w1, r.w2, r.w3, r.adaln = nil, nil, nil, nil, nil, nil
}

func (r *ditLayerGPUResidency) gemv(name string, gpuW *nvidia.GPUFP8E4M3Linear, cpuW *FP8Linear, x, out []float32) error {
	if r != nil && gpuW != nil {
		if err := nvidia.GemvFP8E4M3(out, x, gpuW); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU %s GEMV: %w", name, err)
		}
	}
	return cpuW.weight.GemvTo(x, out)
}

func (r *ditLayerGPUResidency) QKV(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.qkv
	}
	return r.gemv("qkv", w, l.QKV, x, out)
}
func (r *ditLayerGPUResidency) O(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.o
	}
	return r.gemv("o", w, l.O, x, out)
}
func (r *ditLayerGPUResidency) W1(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w1
	}
	return r.gemv("w1", w, l.W1, x, out)
}
func (r *ditLayerGPUResidency) W2(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w2
	}
	return r.gemv("w2", w, l.W2, x, out)
}
func (r *ditLayerGPUResidency) W3(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w3
	}
	return r.gemv("w3", w, l.W3, x, out)
}
func (r *ditLayerGPUResidency) AdaLN(l DiTLayer, x, out []float32) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.adaln
	}
	return r.gemv("adaln", w, l.AdaLN, x, out)
}

func (r *ditLayerGPUResidency) gemm(name string, gpuW *nvidia.GPUFP8E4M3Linear, cpuW *FP8Linear, x, out []float32, batch int) error {
	if r != nil && gpuW != nil {
		if err := nvidia.GemmFP8E4M3(out, x, batch, gpuW); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU %s GEMM: %w", name, err)
		}
	}
	return cpuW.ApplyBatch(x, out, batch)
}

func (r *ditLayerGPUResidency) QKVBatch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.qkv
	}
	return r.gemm("qkv", w, l.QKV, x, out, batch)
}
func (r *ditLayerGPUResidency) OBatch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.o
	}
	return r.gemm("o", w, l.O, x, out, batch)
}
func (r *ditLayerGPUResidency) W1Batch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w1
	}
	return r.gemm("w1", w, l.W1, x, out, batch)
}
func (r *ditLayerGPUResidency) W2Batch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w2
	}
	return r.gemm("w2", w, l.W2, x, out, batch)
}
func (r *ditLayerGPUResidency) W3Batch(l DiTLayer, x, out []float32, batch int) error {
	var w *nvidia.GPUFP8E4M3Linear
	if r != nil {
		w = r.w3
	}
	return r.gemm("w3", w, l.W3, x, out, batch)
}

func (r *ditLayerGPUResidency) W1W3Batch(l DiTLayer, x, outW1, outW3 []float32, batch int) error {
	if r != nil && r.w1 != nil && r.w3 != nil {
		if err := nvidia.Gemm2FP8E4M3SameInput(outW1, outW3, x, batch, r.w1, r.w3); err == nil {
			return nil
		} else if gpuFP8Strict() {
			return fmt.Errorf("DiT layer GPU w1+w3 GEMM2: %w", err)
		}
	}
	if err := l.W1.ApplyBatch(x, outW1, batch); err != nil {
		return err
	}
	return l.W3.ApplyBatch(x, outW3, batch)
}
