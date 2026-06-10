package ideogram4

import (
	"fmt"
	"math"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuHiddenResidentEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_HIDDEN_RESIDENT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func (m *DiTModel) forwardLayersHiddenResident(hidden []float32, adaln []float32, rope *MRoPE) error {
	if m == nil || !nvidia.Available() || rope == nil || len(m.Layers) == 0 {
		return fmt.Errorf("hidden-resident DiT unavailable")
	}
	cfg := m.Config
	emb := cfg.EmbDim
	if emb <= 0 || len(hidden)%emb != 0 {
		return fmt.Errorf("hidden-resident DiT bad hidden shape")
	}
	tokens := len(hidden) / emb
	normEps := float32(cfg.NormEps)
	if normEps <= 0 {
		normEps = 1e-5
	}
	heads, headDim := cfg.NumHeads, cfg.HeadDim
	scaleAttn := float32(1 / math.Sqrt(float64(headDim)))
	hBuf, err := nvidia.Malloc(len(hidden))
	if err != nil {
		return err
	}
	defer hBuf.Free()
	normedBuf, err := nvidia.Malloc(len(hidden))
	if err != nil {
		return err
	}
	defer normedBuf.Free()
	if err := hBuf.Upload(hidden); err != nil {
		return err
	}
	tableLen := tokens * (headDim / 2)
	cosBuf, err := nvidia.Malloc(tableLen)
	if err != nil {
		return err
	}
	defer cosBuf.Free()
	sinBuf, err := nvidia.Malloc(tableLen)
	if err != nil {
		return err
	}
	defer sinBuf.Free()
	if err := cosBuf.Upload(rope.cos[:tableLen]); err != nil {
		return err
	}
	if err := sinBuf.Upload(rope.sin[:tableLen]); err != nil {
		return err
	}
	mod := make([]float32, 4*emb)
	for i := range m.Layers {
		l := &m.Layers[i]
		layerGPU, err := l.uploadGPU()
		if err != nil {
			if gpuFP8Strict() {
				return err
			}
			return fmt.Errorf("hidden-resident layer %d upload: %w", i, err)
		}
		freeLayer := !l.cacheAnyGPUResidency()
		if err := layerGPU.AdaLN(*l, adaln, mod); err != nil {
			if freeLayer {
				layerGPU.Free()
			}
			return err
		}
		scaleMSA := mod[0:emb]
		gateMSA := mod[emb : 2*emb]
		scaleMLP := mod[2*emb : 3*emb]
		gateMLP := mod[3*emb : 4*emb]
		transformAdaLNMod(mod, emb)
		if err := layerGPU.FullLayerIslandsBuffer(*l, hBuf, normedBuf, cosBuf, sinBuf, scaleMSA, gateMSA, scaleMLP, gateMLP, tokens, heads, headDim, scaleAttn, normEps); err != nil {
			if freeLayer {
				layerGPU.Free()
			}
			return fmt.Errorf("hidden-resident layer %d: %w", i, err)
		}
		if freeLayer {
			layerGPU.Free()
		}
	}
	return hBuf.Download(hidden)
}
