package model

import (
	"fmt"
	"os"

	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

// cpuTokenState owns the request-local output, KV and scratch buffers used by
// the legacy sequential CPU generatePreparedEmbeddings loop. It stays package
// private until a resumable decode session adopts it directly.
type cpuTokenState struct {
	output []int

	kvCacheK     [][]float32
	kvCacheV     [][]float32
	compressedKV []*kv.CompressedKVCache

	attnScoresScratch []float32
	attnOutScratch    []float32

	hidden          []float32
	scratchResidual []float32
	scratchQ        []float32
	scratchK        []float32
	scratchV        []float32
	scratchO        []float32
	scratchMlp      []float32
	scratchGate     []float32
	scratchUp       []float32
	scratchDown     []float32

	scratchPLIGate []float32
	scratchPLIProj []float32
	pliProjBuf     []float32
	pliSlices      [][]float32

	maxSequence int
	position    int

	captureFinalHidden func(pos int, hidden []float32)
	skipFinalDecode    bool
}

// newCPUTokenStateForLegacyGenerate mirrors the exact shape/capacity checks of
// the current sequential generatePreparedEmbeddings allocation path so future
// resumable decode work can reuse the same request-owned state without moving
// token math yet.
func newCPUTokenStateForLegacyGenerate(m *LlamaModel, prepared []int, maxTokens int) (*cpuTokenState, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if maxTokens < 0 {
		return nil, fmt.Errorf("maxTokens=%d must be >= 0", maxTokens)
	}
	cfg := m.Config
	maxInt := int(^uint(0) >> 1)
	if maxTokens > maxInt-len(prepared) {
		return nil, fmt.Errorf("output capacity overflow: prepared=%d maxTokens=%d", len(prepared), maxTokens)
	}
	if cfg.NumLayers < 0 || len(m.Layers) < cfg.NumLayers {
		return nil, fmt.Errorf("layers=%d, want at least %d", len(m.Layers), cfg.NumLayers)
	}
	outCap := len(prepared) + maxTokens
	maxSequence := outCap
	if maxSequence < 1 {
		maxSequence = 1
	}
	st := &cpuTokenState{
		output:      make([]int, len(prepared), outCap),
		kvCacheK:    make([][]float32, cfg.NumLayers),
		kvCacheV:    make([][]float32, cfg.NumLayers),
		maxSequence: maxSequence,
		position:    len(prepared),
	}
	copy(st.output, prepared)

	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	inter := cfg.Intermediate
	if h <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || inter < 0 {
		return nil, fmt.Errorf("invalid CPU token state dims hidden=%d heads=%d kvHeads=%d headDim=%d intermediate=%d", h, numHeads, numKVHeads, headDim, inter)
	}

	if m.EnableTurboQuant || os.Getenv("TURBO_QUANT") == "1" || m.TurboQuantConfig != nil {
		tqCfg := kv.DefaultTurboQuantConfig()
		if m.TurboQuantConfig != nil {
			tqCfg = *m.TurboQuantConfig
		}
		if m.TurboQuantStates == nil {
			m.TurboQuantStates = make(map[int]*kv.TurboQuantState)
		}
		getTQ := func(layerHeadDim int) *kv.TurboQuantState {
			if tq := m.TurboQuantStates[layerHeadDim]; tq != nil {
				return tq
			}
			tq := kv.NewTurboQuantState(layerHeadDim, cfg.NumLayers, tqCfg)
			m.TurboQuantStates[layerHeadDim] = tq
			return tq
		}
		loaderDebugf("  TurboQuant: %d-bit keys, %d-bit values, window=%d\n",
			tqCfg.KeyBits, tqCfg.ValueBits, tqCfg.ResidualWindow)

		st.compressedKV = make([]*kv.CompressedKVCache, cfg.NumLayers)
		for l := range st.compressedKV {
			layerHD, err := m.LayerHeadDim(l)
			if err != nil {
				return nil, err
			}
			layerKVHeads := gemmacfg.LayerKVHeads(cfg, l)
			layerKVDim, err := m.LayerKVDim(l)
			if err != nil || layerKVHeads < 0 {
				if err != nil {
					return nil, err
				}
				return nil, fmt.Errorf("layer %d kv heads=%d", l, layerKVHeads)
			}
			tq := getTQ(layerHD)
			st.compressedKV[l] = kv.NewCompressedKVCache(layerKVDim, layerKVHeads, layerHD, tq, tq.IsProtectedLayer(l))
		}
	} else {
		for l := range st.kvCacheK {
			layerKVDim, err := m.LayerKVDim(l)
			if err != nil {
				return nil, err
			}
			cacheCap, okCap := checkedProduct(maxSequence, layerKVDim)
			if !okCap {
				return nil, fmt.Errorf("layer %d KV capacity overflow: seq=%d dim=%d", l, maxSequence, layerKVDim)
			}
			st.kvCacheK[l] = make([]float32, 0, cacheCap)
			st.kvCacheV[l] = make([]float32, 0, cacheCap)
		}
	}

	maxHeadDim := headDim
	for i := range m.Layers {
		layerHD, err := m.LayerHeadDim(i)
		if err != nil {
			return nil, err
		}
		if layerHD > maxHeadDim {
			maxHeadDim = layerHD
		}
	}
	attnOutDim, okAttnOutDim := checkedProduct(numHeads, maxHeadDim)
	if !okAttnOutDim || attnOutDim <= 0 {
		return nil, fmt.Errorf("attention output scratch overflow: heads=%d headDim=%d", numHeads, maxHeadDim)
	}
	st.attnScoresScratch = make([]float32, maxSequence)
	st.attnOutScratch = make([]float32, attnOutDim)

	scQDim, okScQ := checkedProduct(numHeads, maxHeadDim)
	scKVDim, okScKV := checkedProduct(cfg.NumKVHeads, maxHeadDim)
	if !okScQ || !okScKV {
		return nil, fmt.Errorf("layer scratch overflow: qHeads=%d kvHeads=%d headDim=%d", numHeads, cfg.NumKVHeads, maxHeadDim)
	}
	scInter := inter
	for l := 0; l < cfg.NumLayers; l++ {
		lhd, err := m.LayerHeadDim(l)
		if err != nil {
			return nil, err
		}
		lkvh := gemmacfg.LayerKVHeads(cfg, l)
		q, okQ := checkedProduct(numHeads, lhd)
		kvDim, okKV := checkedProduct(lkvh, lhd)
		if !okQ || !okKV {
			return nil, fmt.Errorf("layer %d scratch overflow: qHeads=%d kvHeads=%d headDim=%d", l, numHeads, lkvh, lhd)
		}
		if q > scQDim {
			scQDim = q
		}
		if kvDim > scKVDim {
			scKVDim = kvDim
		}
		if li := m.layerInterFor(&m.Layers[l]); li > scInter {
			scInter = li
		}
	}
	if scQDim < 1 {
		scQDim = 1
	}
	if scKVDim < 1 {
		scKVDim = 1
	}
	if scInter < 1 {
		scInter = 1
	}
	st.hidden = make([]float32, h)
	st.scratchResidual = make([]float32, h)
	st.scratchQ = make([]float32, scQDim)
	st.scratchK = make([]float32, scKVDim)
	st.scratchV = make([]float32, scKVDim)
	st.scratchO = make([]float32, h)
	st.scratchMlp = make([]float32, h)
	st.scratchGate = make([]float32, scInter)
	st.scratchUp = make([]float32, scInter)
	st.scratchDown = make([]float32, h)
	if cfg.HiddenPerLayer > 0 {
		st.scratchPLIGate = make([]float32, cfg.HiddenPerLayer)
		st.scratchPLIProj = make([]float32, h)
		if m.PerLayerModelProj != nil {
			if td, ok := checkedProduct(cfg.NumLayers, cfg.HiddenPerLayer); ok {
				st.pliProjBuf = make([]float32, td)
				st.pliSlices = make([][]float32, cfg.NumLayers)
			}
		}
	}
	return st, nil
}
