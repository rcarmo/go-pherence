package k3

import (
	"fmt"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanBackend offloads compute ops to the PowerVR BXM-4-64 via Vulkan 1.3 compute.
// Each call uploads host slices, dispatches, and downloads results back.
// This round-trip cost means smaller ops may be slower than SIMD; the benefit
// shows for large batched GEMV and attention where the GPU's memory bandwidth wins.
type VulkanBackend struct{}

func (VulkanBackend) Name() string { return TierVulkan.String() }

// alloc allocates a VkBuf of the given byte size or returns an error.
func alloc(sizeBytes int) (*vk.VkBuf, error) {
	b, err := vk.VkBufAlloc(sizeBytes)
	if err != nil {
		return nil, fmt.Errorf("vulkan alloc %d bytes: %w", sizeBytes, err)
	}
	return b, nil
}

func (VulkanBackend) GemvF32(out, x, w []float32, inDim, outDim int) error {
	outBuf, err := alloc(len(out) * 4)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	xBuf, err := alloc(len(x) * 4)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	wBuf, err := alloc(len(w) * 4)
	if err != nil {
		return err
	}
	defer wBuf.Free()
	if err := xBuf.UploadChecked(x); err != nil {
		return err
	}
	if err := wBuf.UploadChecked(w); err != nil {
		return err
	}
	if err := vk.VkGemvF32(outBuf, xBuf, wBuf, inDim, outDim); err != nil {
		return err
	}
	return outBuf.DownloadChecked(out)
}

func (VulkanBackend) RMSNormF32(x, w []float32, eps float32) error {
	xBuf, err := alloc(len(x) * 4)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	wBuf, err := alloc(len(w) * 4)
	if err != nil {
		return err
	}
	defer wBuf.Free()
	if err := xBuf.UploadChecked(x); err != nil {
		return err
	}
	if err := wBuf.UploadChecked(w); err != nil {
		return err
	}
	if err := vk.VkRMSNormF32(xBuf, wBuf, len(x), eps); err != nil {
		return err
	}
	return xBuf.DownloadChecked(x)
}

func (VulkanBackend) RMSNormNoScaleF32(x []float32, eps float32) error {
	xBuf, err := alloc(len(x) * 4)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	if err := xBuf.UploadChecked(x); err != nil {
		return err
	}
	if err := vk.VkRMSNormNoScaleF32(xBuf, len(x), eps); err != nil {
		return err
	}
	return xBuf.DownloadChecked(x)
}

func (VulkanBackend) SiLUMulF32(dst, gate, up []float32) error {
	dstBuf, err := alloc(len(dst) * 4)
	if err != nil {
		return err
	}
	defer dstBuf.Free()
	gateBuf, err := alloc(len(gate) * 4)
	if err != nil {
		return err
	}
	defer gateBuf.Free()
	upBuf, err := alloc(len(up) * 4)
	if err != nil {
		return err
	}
	defer upBuf.Free()
	if err := gateBuf.UploadChecked(gate); err != nil {
		return err
	}
	if err := upBuf.UploadChecked(up); err != nil {
		return err
	}
	if err := vk.VkSiLUMulF32(dstBuf, gateBuf, upBuf, len(dst)); err != nil {
		return err
	}
	return dstBuf.DownloadChecked(dst)
}

func (VulkanBackend) GELUTanhMulF32(dst, gate, up []float32) error {
	// VkGELUTanhMulF32 mutates gate in-place (gate = gelu(gate)*up).
	// We use a separate dst → copy back to dst after.
	gateBuf, err := alloc(len(gate) * 4)
	if err != nil {
		return err
	}
	defer gateBuf.Free()
	upBuf, err := alloc(len(up) * 4)
	if err != nil {
		return err
	}
	defer upBuf.Free()
	if err := gateBuf.UploadChecked(gate); err != nil {
		return err
	}
	if err := upBuf.UploadChecked(up); err != nil {
		return err
	}
	if err := vk.VkGELUTanhMulF32(gateBuf, upBuf, len(gate)); err != nil {
		return err
	}
	return gateBuf.DownloadChecked(dst)
}

func (VulkanBackend) RoPEPartialF32(x, freqs []float32, pos, nHeads, headDim, rotHalf int) error {
	xBuf, err := alloc(len(x) * 4)
	if err != nil {
		return err
	}
	defer xBuf.Free()
	freqBuf, err := alloc(len(freqs) * 4)
	if err != nil {
		return err
	}
	defer freqBuf.Free()
	if err := xBuf.UploadChecked(x); err != nil {
		return err
	}
	if err := freqBuf.UploadChecked(freqs); err != nil {
		return err
	}
	if err := vk.VkRoPEPartialF32(xBuf, freqBuf, pos, nHeads, headDim, rotHalf); err != nil {
		return err
	}
	return xBuf.DownloadChecked(x)
}

func (VulkanBackend) AttentionScoresF32(out, q, kCache []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	outBuf, err := alloc(len(out) * 4)
	if err != nil {
		return err
	}
	defer outBuf.Free()
	qBuf, err := alloc(len(q) * 4)
	if err != nil {
		return err
	}
	defer qBuf.Free()
	kBuf, err := alloc(len(kCache) * 4)
	if err != nil {
		return err
	}
	defer kBuf.Free()
	if err := qBuf.UploadChecked(q); err != nil {
		return err
	}
	if err := kBuf.UploadChecked(kCache); err != nil {
		return err
	}
	if err := vk.VkAttentionScoresF32(outBuf, qBuf, kBuf, seqLen, nHeads, nKVHeads, headDim, scale); err != nil {
		return err
	}
	return outBuf.DownloadChecked(out)
}
