package model

import (
	"encoding/json"
	"fmt"

	"github.com/rcarmo/go-pherence/backends/mlx"
	simdq4 "github.com/rcarmo/go-pherence/backends/simd/quant/q4"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"

	"github.com/rcarmo/go-pherence/runtime/kv"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"

	"math"
	"os"
	"time"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/tensor"
)

// LoadLlama loads a LLaMA-style model from safetensors + config.json.
// ForceOnTheFly controls whether quantized models keep INT4 packed weights.
// Set to true before LoadLlama when using GPU forward pass (GPU Q4 GEMV needs packed weights).
var ForceOnTheFly bool

func LoadLlama(dir string) (model *LlamaModel, err error) {
	defer func() {
		if r := recover(); r != nil {
			model = nil
			if e, ok := r.(error); ok {
				err = fmt.Errorf("load llama %s: %w", dir, e)
			} else {
				err = fmt.Errorf("load llama %s: %v", dir, r)
			}
		}
	}()

	// Load config
	var cfg LlamaConfig
	cfgData, err := loaderconfig.ReadModelConfig(dir, &cfg)
	if err != nil {
		return nil, err
	}
	// Gemma4 and Qwen3.5/Qwen3.6: text config may be nested under text_config.
	if normalized, ok := gemmacfg.NormalizeTextConfig(cfgData, cfg); ok {
		cfg = normalized
	} else if cfg.HiddenSize == 0 {
		var nested struct {
			TextConfig LlamaConfig `json:"text_config"`
			ModelType  string      `json:"model_type"`
		}
		if err := json.Unmarshal(cfgData, &nested); err == nil && nested.TextConfig.HiddenSize > 0 {
			// Preserve top-level model_type when the nested config omits it.
			outerType := nested.ModelType
			cfg = nested.TextConfig
			if cfg.ModelType == "" {
				cfg.ModelType = outerType + "_text"
			}
		}
	}
	if cfg.AttentionLogitSoftcapping < 0 {
		return nil, fmt.Errorf("invalid attn_logit_softcapping %g: must be non-negative", cfg.AttentionLogitSoftcapping)
	}
	if err := validateGemmaAttentionConfig(cfg); err != nil {
		return nil, err
	}
	if cfg.RMSNormEps == 0 {
		cfg.RMSNormEps = 1e-5
	}
	// hidden_act fallback: some models use "hidden_act" instead of "hidden_activation"
	if cfg.HiddenAct == "" {
		var actFallback struct {
			HiddenAct string `json:"hidden_act"`
		}
		if err := json.Unmarshal(cfgData, &actFallback); err == nil && actFallback.HiddenAct != "" {
			cfg.HiddenAct = actFallback.HiddenAct
		}
	}
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	// Gemma4: infer sliding_window_pattern from layer_types
	if len(cfg.LayerTypes) > 0 && cfg.SlidingWindowPattern == 0 {
		for i, lt := range cfg.LayerTypes {
			if lt == "full_attention" {
				cfg.SlidingWindowPattern = i + 1
				break
			}
		}
	}

	// Detect tensor prefix (Gemma4 uses "language_model.model.")
	cfg.TensorPrefix = ""
	if cfg.ModelType == "gemma4_text" || cfg.ModelType == "gemma4" {
		cfg.TensorPrefix = "language_model."
	}

	// Detect unsupported Model Optimizer / NVFP4-style quantization before
	// falling through into normal/MLX/GPTQ tensor loading. This keeps FP4
	// checkpoints from failing later with misleading missing-tensor errors.
	if quantMeta, err := loaderconfig.ParseQuantizationMetadata(cfgData); err == nil && quantMeta.UnsupportedFP4 {
		return nil, fmt.Errorf("unsupported FP4/NVFP4 quantization: quant_algo=%q quant_method=%q", quantMeta.Algo, quantMeta.Method)
	}
	if qwenMTP, err := loaderconfig.ParseQwenNativeMTPMetadata(cfgData); err == nil && qwenMTP.HasNativeMTP && (qwenMTP.ModelType == "qwen3_5" || qwenMTP.ModelType == "qwen3_5_text") {
		return nil, fmt.Errorf("unsupported Qwen3.5/Qwen3.6 native MTP architecture: model_type=%q architecture=%q mtp_num_hidden_layers=%d linear_attention=%v requires qwen3_5 base support", qwenMTP.ModelType, qwenMTP.Architecture, qwenMTP.MTPNumHiddenLayers, qwenMTP.HasLinearAttention)
	}

	// Try sharded first, then single file
	f, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if os.Getenv("GO_PHERENCE_EAGER_LOAD") == "1" {
		if ef, ok := f.(weights.EagerSource); ok {
			t0 := time.Now()
			bytes, err := ef.EagerLoad()
			if err != nil {
				return nil, fmt.Errorf("eager load safetensors: %w", err)
			}
			loaderDebugf("  Eager loaded %.1f MB of mmap'd weights in %.2fs\n", float64(bytes)/(1024*1024), time.Since(t0).Seconds())
		}
	}

	// Try loading quantization config (GPTQ)
	var qc struct {
		Bits      int  `json:"bits"`
		GroupSize int  `json:"group_size"`
		Sym       bool `json:"sym"`
	}
	if ok, err := loaderconfig.ReadQuantizeConfig(dir, &qc); err == nil && ok && qc.Bits > 0 {
		cfg.QuantBits = qc.Bits
		cfg.QuantGroup = qc.GroupSize
		cfg.QuantSym = qc.Sym
		cfg.QuantFormat = "gptq"
		loaderDebugf("  GPTQ: %d-bit, group=%d, sym=%v\n", qc.Bits, qc.GroupSize, qc.Sym)
	}

	// Try MLX quantization from config.json
	if cfg.QuantBits == 0 {
		var mlxCfg struct {
			Quantization struct {
				Bits      int `json:"bits"`
				GroupSize int `json:"group_size"`
			} `json:"quantization"`
		}
		if err := json.Unmarshal(cfgData, &mlxCfg); err == nil && mlxCfg.Quantization.Bits > 0 {
			cfg.QuantBits = mlxCfg.Quantization.Bits
			cfg.QuantGroup = mlxCfg.Quantization.GroupSize
			cfg.QuantSym = false // MLX uses bias, not symmetric
			cfg.QuantFormat = "mlx"
			loaderDebugf("  MLX: %d-bit, group=%d\n", cfg.QuantBits, cfg.QuantGroup)
		}
	}

	reapCfg, err := LoadREAPConfig(dir)
	if err != nil {
		return nil, err
	}
	if reapCfg != nil {
		loaderDebugf("  REAP: static expert pruning enabled prune_ratio=%.2f default_active=%d layer_masks=%d\n", reapCfg.PruneRatio, len(reapCfg.DefaultActive), len(reapCfg.LayerActiveNumeric))
	}

	m := &LlamaModel{Config: cfg, REAP: reapCfg}
	if cfg.IsOrthrus() {
		loaderDebugf("  Orthrus: block_size=%d mask_token_id=%d (baseline Qwen3 path; diffusion tensors ignored)\n", cfg.OrthrusBlockSize, cfg.OrthrusMaskTokenID)
	}
	h := cfg.HiddenSize
	// For large models (>2B params), skip pre-transpose to save memory
	large := cfg.HiddenSize >= 3000
	m.Large = large
	m.Quantized = cfg.QuantBits > 0
	// Heuristic: dequant at load if enough RAM, on-the-fly for very large models
	// F32 dequant needs ~4× model_params bytes. For 7B = ~28GB.
	// On-the-fly keeps INT4 packed (~4GB for 7B) but inference is 20× slower.
	// If GPU is available and model is quantized, keep INT4 packed for GPU upload
	// OnTheFlyQuant: keep INT4 packed (for GPU upload) vs dequant-at-load (fast CPU).
	// Set ForceOnTheFly=true before LoadLlama when using GPU forward pass.
	onTheFly := ForceOnTheFly && cfg.QuantBits > 0
	m.OnTheFlyQuant = onTheFly

	prefix := cfg.TensorPrefix
	load := func(name string, shape []int) *tensor.Tensor {
		data, actualShape, err := f.GetFloat32(prefix + name)
		if err != nil && prefix != "" {
			data, actualShape, err = f.GetFloat32(name)
		}
		if err != nil {
			panic(fmt.Sprintf("load %s: %v", name, err))
		}
		// Use actual shape from safetensors if it matches element count
		if len(actualShape) > 0 {
			n := 1
			for _, d := range actualShape {
				n *= d
			}
			if n == len(data) {
				shape = actualShape
			}
		}
		return tensor.FromFloat32(data, shape)
	}
	loadT := func(name string, shape []int) *tensor.Tensor {
		if large {
			return load(name, shape) // keep original layout, use NT path
		}
		return load(name, shape).Transpose2D()
	}

	// loadQW loads raw GPTQ quantized weight without dequantization.
	loadQW := func(name string, outDim, inDim int) *QuantWeight {
		qw, _, err := f.GetInt32(prefix + name + ".qweight")
		if err != nil && prefix != "" {
			qw, _, err = f.GetInt32(name + ".qweight")
		}
		if err != nil {
			panic(fmt.Sprintf("loadQW %s.qweight: %v", name, err))
		}
		gIdx, _, err := f.GetInt32(name + ".g_idx")
		if err != nil {
			panic(fmt.Sprintf("loadQW %s.g_idx: %v", name, err))
		}
		scRaw, scDtype, _, err := f.GetRaw(name + ".scales")
		if err != nil {
			panic(fmt.Sprintf("loadQW %s.scales: %v", name, err))
		}
		var scales []float32
		if scDtype == "F16" {
			n := len(scRaw) / 2
			scales = make([]float32, n)
			for i := 0; i < n; i++ {
				h := uint16(scRaw[i*2]) | uint16(scRaw[i*2+1])<<8
				scales[i] = simdq4.Float16ToFloat32(h)
			}
		} else {
			scales, _, err = f.GetFloat32(name + ".scales")
			if err != nil {
				panic(fmt.Sprintf("loadQW %s.scales: %v", name, err))
			}
		}
		if err := simdq4.ValidateSym(qw, gIdx, scales, inDim, outDim); err != nil {
			panic(fmt.Sprintf("loadQW %s GPTQ validation: %v", name, err))
		}
		return &QuantWeight{QWeight: qw, GIdx: gIdx, Scales: scales, InDim: inDim, OutDim: outDim}
	}
	_ = loadQW

	// loadMLXW loads an MLX affine quantized weight.
	loadMLXW := func(name string, outDim, inDim int) *mlx.QuantWeight {
		qw, err := mlx.LoadWeight(f, prefix+name, outDim, inDim, cfg.QuantGroup, cfg.QuantBits)
		if err != nil && prefix != "" {
			qw, err = mlx.LoadWeight(f, name, outDim, inDim, cfg.QuantGroup, cfg.QuantBits)
		}
		if err != nil {
			panic(fmt.Sprintf("loadMLXW %s: %v", name, err))
		}
		return qw
	}
	_ = loadMLXW

	// loadQ loads a GPTQ quantized weight, dequantizes to F32.
	// name is the base name (e.g. "model.layers.0.mlp.gate_proj")
	// shape is [outFeatures, inFeatures] (the original weight shape)
	loadQ := func(name string, outDim, inDim int) *tensor.Tensor {
		qw, _, err := f.GetInt32(name + ".qweight")
		if err != nil {
			panic(fmt.Sprintf("loadQ %s.qweight: %v", name, err))
		}
		gIdx, _, err := f.GetInt32(name + ".g_idx")
		if err != nil {
			panic(fmt.Sprintf("loadQ %s.g_idx: %v", name, err))
		}
		// Scales are F16, load raw and convert
		scRaw, scDtype, _, err := f.GetRaw(name + ".scales")
		if err != nil {
			panic(fmt.Sprintf("loadQ %s.scales: %v", name, err))
		}
		var scales []float32
		if scDtype == "F16" {
			n := len(scRaw) / 2
			scales = make([]float32, n)
			for i := 0; i < n; i++ {
				h := uint16(scRaw[i*2]) | uint16(scRaw[i*2+1])<<8
				scales[i] = simdq4.Float16ToFloat32(h)
			}
		} else {
			scales, _, err = f.GetFloat32(name + ".scales")
			if err != nil {
				panic(fmt.Sprintf("loadQ %s.scales: %v", name, err))
			}
		}

		var data []float32
		if cfg.QuantSym {
			if err := simdq4.ValidateSym(qw, gIdx, scales, inDim, outDim); err != nil {
				panic(fmt.Sprintf("loadQ %s GPTQ validation: %v", name, err))
			}
			data = simdq4.DequantSym(qw, gIdx, scales, inDim, outDim)
		} else {
			qz, _, err := f.GetInt32(name + ".qzeros")
			if err != nil {
				panic(fmt.Sprintf("loadQ %s.qzeros: %v", name, err))
			}
			if err := simdq4.Validate(qw, qz, gIdx, scales, inDim, outDim, false); err != nil {
				panic(fmt.Sprintf("loadQ %s GPTQ validation: %v", name, err))
			}
			data = simdq4.Dequant(qw, qz, gIdx, scales, inDim, outDim, false)
		}
		// data is [outDim, inDim] row-major
		t := tensor.FromFloat32(data, []int{outDim, inDim})
		if !large {
			t = t.Transpose2D() // pre-transpose for NN path
		}
		return t
	}
	_ = loadQ

	// Load embeddings — MLX may quantize these
	var embedMLX *mlx.QuantWeight
	if cfg.QuantFormat == "mlx" {
		// Try to load quantized embedding, dequantize for lookup, and retain the
		// packed handle for tied compact LM-head GPU execution.
		if emb, err := mlx.LoadWeight(f, prefix+"model.embed_tokens", cfg.VocabSize, h, cfg.QuantGroup, cfg.QuantBits); err == nil {
			embedMLX = emb
			data := mlx.Dequant(emb)
			m.EmbedTokens = tensor.FromFloat32(data, []int{cfg.VocabSize, h})
		} else {
			m.EmbedTokens = load("model.embed_tokens.weight", []int{cfg.VocabSize, h})
		}
	} else {
		m.EmbedTokens = load("model.embed_tokens.weight", []int{cfg.VocabSize, h})
	}
	m.Norm = load("model.norm.weight", []int{h})

	// LM head: often tied to embed_tokens. MLX may quantize it too.
	if cfg.QuantFormat == "mlx" {
		if lm, err := mlx.LoadWeight(f, prefix+"lm_head", cfg.VocabSize, h, cfg.QuantGroup, cfg.QuantBits); err == nil {
			m.LMHeadMLX = lm
			data := mlx.Dequant(lm)
			m.LMHead = tensor.FromFloat32(data, []int{cfg.VocabSize, h})
		} else {
			m.LMHead = m.EmbedTokens // tied weights
			if embedMLX != nil {
				m.LMHeadMLX = embedMLX
			}
		}
	} else if _, _, err := f.GetFloat32("lm_head.weight"); err == nil {
		m.LMHead = load("lm_head.weight", []int{cfg.VocabSize, h})
	} else {
		m.LMHead = m.EmbedTokens // tied weights
		if embedMLX != nil {
			m.LMHeadMLX = embedMLX
		}
	}

	kvDim := cfg.HeadDim * cfg.NumKVHeads

	// tryLoad checks if a tensor exists
	tryLoad := func(name string) bool {
		_, _, _, err := f.GetRaw(prefix + name)
		if err != nil && prefix != "" {
			_, _, _, err = f.GetRaw(name)
		}
		return err == nil
	}
	_ = tryLoad

	m.Layers = make([]LlamaLayer, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		p := fmt.Sprintf("model.layers.%d", l)
		var layer LlamaLayer

		// Per-layer Q/K/V/O dimensions (Gemma4: varies by layer type)
		qDimL := h      // Q output = numHeads * headDim
		kvDimL := kvDim // K/V output = layerKVHeads * headDim
		oDimIn := h     // O input = numHeads * headDim
		if len(cfg.LayerTypes) > l {
			lt := cfg.LayerTypes[l]
			var lhd int
			if lt == "full_attention" && cfg.GlobalHeadDim > 0 {
				lhd = cfg.GlobalHeadDim
			} else {
				lhd = cfg.HeadDim
			}
			layerKVHeads := gemmacfg.LayerKVHeads(cfg, l)
			qDimL = cfg.NumHeads * lhd
			kvDimL = layerKVHeads * lhd
			oDimIn = qDimL
		}

		// Check if this layer uses MoE (switch_mlp format)
		isMoELayer := cfg.NumExperts > 0 && tryLoad(p+".mlp.switch_mlp.gate_proj.weight")

		hasVProj := tryLoad(p + ".self_attn.v_proj.weight")
		if cfg.QuantFormat == "mlx" && onTheFly {
			kw := loadMLXW(p+".self_attn.k_proj", kvDimL, h)
			vw := kw
			if hasVProj || !cfg.AttentionKEqV {
				vw = loadMLXW(p+".self_attn.v_proj", kvDimL, h)
			}
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QWm:       loadMLXW(p+".self_attn.q_proj", qDimL, h),
				KWm:       kw,
				VWm:       vw,
				OWm:       loadMLXW(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateWm = loadMLXW(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpWm = loadMLXW(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownWm = loadMLXW(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else if cfg.QuantFormat == "mlx" {
			// MLX dequant-at-load
			loadMLXDeq := func(name string, outDim, inDim int) *tensor.Tensor {
				qw := loadMLXW(name, outDim, inDim)
				data := mlx.Dequant(qw)
				// Use actual dims from loaded weight (may differ from caller's hint)
				if large {
					return tensor.FromFloat32(data, []int{qw.OutDim, qw.InDim})
				}
				return tensor.FromFloat32(data, []int{qw.OutDim, qw.InDim}).Transpose2D()
			}
			kw := loadMLXDeq(p+".self_attn.k_proj", kvDimL, h)
			vw := kw
			if hasVProj || !cfg.AttentionKEqV {
				vw = loadMLXDeq(p+".self_attn.v_proj", kvDimL, h)
			}
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW:        loadMLXDeq(p+".self_attn.q_proj", qDimL, h),
				KW:        kw,
				VW:        vw,
				OW:        loadMLXDeq(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateW = loadMLXDeq(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpW = loadMLXDeq(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownW = loadMLXDeq(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else if cfg.QuantBits > 0 && onTheFly {
			kwq := loadQW(p+".self_attn.k_proj", kvDimL, h)
			vwq := kwq
			if hasVProj || !cfg.AttentionKEqV {
				vwq = loadQW(p+".self_attn.v_proj", kvDimL, h)
			}
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QWq:       loadQW(p+".self_attn.q_proj", qDimL, h),
				KWq:       kwq,
				VWq:       vwq,
				OWq:       loadQW(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateWq = loadQW(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpWq = loadQW(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownWq = loadQW(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else if cfg.QuantBits > 0 {
			kw := loadQ(p+".self_attn.k_proj", kvDimL, h)
			vw := kw
			if hasVProj || !cfg.AttentionKEqV {
				vw = loadQ(p+".self_attn.v_proj", kvDimL, h)
			}
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW:        loadQ(p+".self_attn.q_proj", qDimL, h),
				KW:        kw,
				VW:        vw,
				OW:        loadQ(p+".self_attn.o_proj", h, oDimIn),
			}
			if !isMoELayer {
				layer.GateW = loadQ(p+".mlp.gate_proj", cfg.Intermediate, h)
				layer.UpW = loadQ(p+".mlp.up_proj", cfg.Intermediate, h)
				layer.DownW = loadQ(p+".mlp.down_proj", h, cfg.Intermediate)
			}
		} else {
			layer = LlamaLayer{
				InputNorm: load(p+".input_layernorm.weight", []int{h}),
				PostNorm:  load(p+".post_attention_layernorm.weight", []int{h}),
				QW:        loadT(p+".self_attn.q_proj.weight", []int{qDimL, h}),
				KW:        loadT(p+".self_attn.k_proj.weight", []int{kvDimL, h}),
				VW:        loadT(p+".self_attn.v_proj.weight", []int{kvDimL, h}),
				OW:        loadT(p+".self_attn.o_proj.weight", []int{h, oDimIn}),
			}
			if !isMoELayer {
				layer.GateW = loadT(p+".mlp.gate_proj.weight", []int{cfg.Intermediate, h})
				layer.UpW = loadT(p+".mlp.up_proj.weight", []int{cfg.Intermediate, h})
				layer.DownW = loadT(p+".mlp.down_proj.weight", []int{h, cfg.Intermediate})
			}
		}
		// Optional Q/K/V biases (Qwen2 has these, LLaMA doesn't)
		if tryLoad(p + ".self_attn.q_proj.bias") {
			layer.QB = load(p+".self_attn.q_proj.bias", []int{qDimL})
			layer.KB = load(p+".self_attn.k_proj.bias", []int{kvDimL})
			layer.VB = load(p+".self_attn.v_proj.bias", []int{kvDimL})
		}
		// Optional pre/post FFN norms (Gemma3 has these)
		if tryLoad(p + ".pre_feedforward_layernorm.weight") {
			layer.PreFFNNorm = load(p+".pre_feedforward_layernorm.weight", []int{h})
			layer.PostFFNNorm = load(p+".post_feedforward_layernorm.weight", []int{h})
		}
		// Optional QK-Norm (Qwen3/Gemma3 have these)
		layerHD := qDimL / cfg.NumHeads // per-layer head dim
		if tryLoad(p + ".self_attn.q_norm.weight") {
			layer.QNorm = load(p+".self_attn.q_norm.weight", []int{layerHD})
			layer.KNorm = load(p+".self_attn.k_norm.weight", []int{layerHD})
		}

		// MoE: load router and expert weights from switch_mlp format
		if cfg.NumExperts > 0 && cfg.MoEIntermediate > 0 {
			moePath := p + ".mlp"
			if tryLoad(moePath + ".gate.weight") {
				layer.IsMoE = true
				// Router gate: [numExperts, hidden] — load as MLX quantized
				if cfg.QuantFormat == "mlx" && onTheFly {
					layer.RouterW = loadMLXW(moePath+".gate", cfg.NumExperts, h)
				}
				// Expert weights: switch_mlp format [numExperts, moeInter, packed]
				moeI := cfg.MoEIntermediate
				expGate, err := LoadSwitchMLXExperts(f, moePath+".switch_mlp.gate_proj", cfg.NumExperts, moeI, h, cfg.QuantGroup, cfg.QuantBits)
				if err == nil {
					layer.ExpertGateW = expGate
				} else {
					loaderDebugf("  MoE layer %d gate_proj: %v\n", l, err)
				}
				expUp, err := LoadSwitchMLXExperts(f, moePath+".switch_mlp.up_proj", cfg.NumExperts, moeI, h, cfg.QuantGroup, cfg.QuantBits)
				if err == nil {
					layer.ExpertUpW = expUp
				} else {
					loaderDebugf("  MoE layer %d up_proj: %v\n", l, err)
				}
				expDown, err := LoadSwitchMLXExperts(f, moePath+".switch_mlp.down_proj", cfg.NumExperts, h, moeI, cfg.QuantGroup, cfg.QuantBits)
				if err == nil {
					layer.ExpertDownW = expDown
				} else {
					loaderDebugf("  MoE layer %d down_proj: %v\n", l, err)
				}
				// Clear the non-MoE MLP weights (they don't apply)
				layer.GateWm = nil
				layer.UpWm = nil
				layer.DownWm = nil
				layer.GateW = nil
				layer.UpW = nil
				layer.DownW = nil
			}
		}
		// Gemma4: per-layer properties
		if len(cfg.LayerTypes) > l {
			lt := cfg.LayerTypes[l]
			// Head dim: global layers use GlobalHeadDim, sliding use HeadDim
			if lt == "full_attention" && cfg.GlobalHeadDim > 0 {
				layer.HeadDimLocal = cfg.GlobalHeadDim
			} else {
				layer.HeadDimLocal = cfg.HeadDim
			}
			// KV sharing: first N layers have own K/V, rest share
			firstKVShared := cfg.NumLayers - cfg.NumKVSharedLayers
			if l < firstKVShared || cfg.NumKVSharedLayers == 0 {
				layer.HasKV = true
			} else {
				layer.HasKV = false
				// Find the source layer (same layer type, in the first M layers)
				for src := 0; src < firstKVShared; src++ {
					if cfg.LayerTypes[src] == lt {
						layer.KVSourceLayer = src
					}
				}
			}
		} else {
			layer.HeadDimLocal = cfg.HeadDim
			layer.HasKV = true
		}

		// Layer scalar (Gemma4)
		layer.LayerScalar = 1.0
		if tryLoad(p + ".layer_scalar") {
			d := load(p+".layer_scalar", []int{1})
			layer.LayerScalar = d.Data()[0]
		}

		// V norm (Gemma4)
		if tryLoad(p + ".self_attn.v_norm.weight") {
			layer.VNorm = load(p+".self_attn.v_norm.weight", []int{layer.HeadDimLocal})
		}

		// Per-layer input gating weights (Gemma4)
		if cfg.HiddenPerLayer > 0 {
			hpl := cfg.HiddenPerLayer
			if cfg.QuantFormat == "mlx" && cfg.QuantBits > 0 {
				if qw, err := mlx.LoadWeight(f, prefix+p+".per_layer_input_gate", hpl, h, cfg.QuantGroup, cfg.QuantBits); err == nil {
					layer.PLIGate = mlx.Dequant(qw)
					qw2, err := mlx.LoadWeight(f, prefix+p+".per_layer_projection", h, hpl, cfg.QuantGroup, cfg.QuantBits)
					if err != nil {
						panic(fmt.Sprintf("load MLX %s.per_layer_projection: %v", p, err))
					}
					layer.PLIProj = mlx.Dequant(qw2)
					if tryLoad(p + ".post_per_layer_input_norm.weight") {
						layer.PLIPostNorm = load(p+".post_per_layer_input_norm.weight", []int{h}).Data()
					}
				}
			} else if tryLoad(p + ".per_layer_input_gate.weight") {
				layer.PLIGate = load(p+".per_layer_input_gate.weight", nil).Data()
				layer.PLIProj = load(p+".per_layer_projection.weight", nil).Data()
				layer.PLIPostNorm = load(p+".post_per_layer_input_norm.weight", []int{h}).Data()
			}
		}

		m.Layers[l] = layer
	}

	// Gemma3: norm formula is (1 + weight) — confirmed in mlx-lm gemma3_text.py line 111
	// Gemma4 inherits from Gemma3n which uses raw weight (NOT 1+w)
	if cfg.ModelType == "gemma3_text" {
		for l := range m.Layers {
			for _, norm := range []*tensor.Tensor{
				m.Layers[l].InputNorm, m.Layers[l].PostNorm,
				m.Layers[l].PreFFNNorm, m.Layers[l].PostFFNNorm,
				m.Layers[l].QNorm, m.Layers[l].KNorm,
			} {
				if norm != nil {
					d := norm.Data()
					for i := range d {
						d[i] += 1.0
					}
				}
			}
		}
		nd := m.Norm.Data()
		for i := range nd {
			nd[i] += 1.0
		}
	}

	// Gemma4: load model-level per-layer projection weights
	if cfg.HiddenPerLayer > 0 {
		hpl := cfg.HiddenPerLayer
		totalDim := cfg.NumLayers * hpl
		// per_layer_model_projection: [totalDim, hidden] BF16 (not quantized)
		if tryLoad("model.per_layer_model_projection.weight") {
			m.PerLayerModelProj = load("model.per_layer_model_projection.weight", []int{totalDim, h}).Data()
		}
		// per_layer_projection_norm: [hpl]
		if tryLoad("model.per_layer_projection_norm.weight") {
			m.PerLayerProjNorm = load("model.per_layer_projection_norm.weight", []int{hpl}).Data()
		}
		// embed_tokens_per_layer: [vocabPerLayer, totalDim] quantized
		vpl := cfg.VocabPerLayer
		if vpl == 0 {
			vpl = 262144
		} // default for Gemma4
		if cfg.QuantFormat == "mlx" && cfg.QuantBits > 0 {
			if qw, err := mlx.LoadWeight(f, prefix+"model.embed_tokens_per_layer", vpl, totalDim, cfg.QuantGroup, cfg.QuantBits); err == nil {
				m.EmbedPerLayer = mlx.Dequant(qw)
				loaderDebugf("  Loaded per-layer embedding: [%d, %d]\n", vpl, totalDim)
			}
		} else if tryLoad("model.embed_tokens_per_layer.weight") {
			m.EmbedPerLayer = load("model.embed_tokens_per_layer.weight", []int{vpl, totalDim}).Data()
		}
		m.PerLayerInputScale = 0.7071067811865476 // 2^-0.5
		m.PerLayerProjScale = float32(1.0 / math.Sqrt(float64(h)))
		m.EmbedPerLayerScale = float32(math.Sqrt(float64(hpl)))
	}

	// Pre-compute RoPE frequencies
	m.precomputeRoPE()
	m.precomputeGemma4RoPE()
	if cfg.ModelType == "gemma4_text" {
		loaderDebugf("  RoPE: SWA half=%d (theta=10k), Full half=%d (theta=1M, partial=0.25)\n", m.RopeHalfSWA, m.RopeHalfFull)
	}

	return m, nil
}

// Generate produces tokens autoregressively.
func (m *LlamaModel) mvQ(out, x []float32, qw *QuantWeight) bool {
	if qw == nil {
		return false
	}
	return simdq4.GemvSymTo(out, x, qw.QWeight, qw.GIdx, qw.Scales, qw.InDim, qw.OutDim)
}

func (m *LlamaModel) mv(out, x, w []float32, inDim, outDim int) {
	if m.Large {
		gemvNTParallel(out, x, w, inDim, outDim)
	} else {
		gemvParallel(out, x, w, inDim, outDim)
	}
}

// PreparedGenerateTokens returns the token sequence that Generate will actually
// process after model-specific BOS/chat-template wrapping. It is useful for
// callers that need accurate prompt/generation accounting.
func (m *LlamaModel) PreparedGenerateTokens(tokenIDs []int) []int {
	return append([]int(nil), m.prepareGenerateTokens(tokenIDs)...)
}

func (m *LlamaModel) prepareGenerateTokens(tokenIDs []int) []int {
	cfg := m.Config

	// BOS token for Gemma
	if cfg.BOSTokenID > 0 && (cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text") {
		tokenIDs = append([]int{cfg.BOSTokenID}, tokenIDs...)
	}
	// Gemma4 instruct chat template: <bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
	if cfg.ModelType == "gemma4_text" && m.Tok != nil {
		turnStart, turnEnd := -1, -1
		newlineID := -1
		for id, tok := range m.Tok.InvVocab {
			if tok == "<|turn>" {
				turnStart = id
			}
			if tok == "<turn|>" {
				turnEnd = id
			}
			if tok == "\n" {
				newlineID = id
			}
		}
		if turnStart >= 0 && turnEnd >= 0 && newlineID >= 0 {
			user := m.Tok.Encode("user")
			mdl := m.Tok.Encode("model")
			wrapped := []int{cfg.BOSTokenID, turnStart}
			wrapped = append(wrapped, user...)
			wrapped = append(wrapped, newlineID)
			wrapped = append(wrapped, tokenIDs[1:]...) // skip BOS
			wrapped = append(wrapped, turnEnd)
			wrapped = append(wrapped, newlineID)
			wrapped = append(wrapped, turnStart)
			wrapped = append(wrapped, mdl...)
			wrapped = append(wrapped, newlineID)
			tokenIDs = wrapped
		}
	}
	// Qwen3/Qwen3-MoE instruct chat template: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
	if (cfg.ModelType == "qwen3" || cfg.ModelType == "qwen3_moe") && m.Tok != nil {
		imStart, imEnd, nlID := -1, -1, -1
		for id, tok := range m.Tok.InvVocab {
			if tok == "<|im_start|>" {
				imStart = id
			}
			if tok == "<|im_end|>" {
				imEnd = id
			}
			if tok == "\n" || tok == "\u010a" || tok == "Ċ" {
				nlID = id
			}
		}
		if imStart >= 0 && imEnd >= 0 && nlID >= 0 {
			user := m.Tok.Encode("user")
			assistant := m.Tok.Encode("assistant")
			wrapped := []int{imStart}
			wrapped = append(wrapped, user...)
			wrapped = append(wrapped, nlID)
			wrapped = append(wrapped, tokenIDs...)
			wrapped = append(wrapped, imEnd, nlID, imStart)
			wrapped = append(wrapped, assistant...)
			wrapped = append(wrapped, nlID)
			tokenIDs = wrapped
		}
	}
	return tokenIDs
}

func (m *LlamaModel) Generate(tokenIDs []int, maxTokens int) []int {
	return m.generatePrepared(m.prepareGenerateTokens(tokenIDs), maxTokens)
}

func (m *LlamaModel) generatePrepared(tokenIDs []int, maxTokens int) []int {
	cfg := m.Config

	if maxTokens < 0 {
		return append([]int(nil), tokenIDs...)
	}
	maxInt := int(^uint(0) >> 1)
	if maxTokens > maxInt-len(tokenIDs) || cfg.NumLayers < 0 || len(m.Layers) < cfg.NumLayers {
		return append([]int(nil), tokenIDs...)
	}
	outCap := len(tokenIDs) + maxTokens
	output := make([]int, len(tokenIDs), outCap)
	copy(output, tokenIDs)

	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	inter := cfg.Intermediate
	if h <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || inter < 0 {
		return output
	}

	// Allocate KV cache (with optional TurboQuant compression)
	kvCacheK := make([][]float32, cfg.NumLayers) // [layers][seqLen * layerKVDim]
	kvCacheV := make([][]float32, cfg.NumLayers)
	var compressedKV []*kv.CompressedKVCache
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

		compressedKV = make([]*kv.CompressedKVCache, cfg.NumLayers)
		for l := range compressedKV {
			layerHD, err := m.LayerHeadDim(l)
			if err != nil {
				return output
			}
			layerKVHeads := gemmacfg.LayerKVHeads(cfg, l)
			layerKVDim, err := m.LayerKVDim(l)
			if err != nil || layerKVHeads < 0 {
				return output
			}
			tq := getTQ(layerHD)
			compressedKV[l] = kv.NewCompressedKVCache(layerKVDim, layerKVHeads, layerHD, tq, tq.IsProtectedLayer(l))
		}
	} else {
		seqCap := len(tokenIDs) + maxTokens
		if seqCap < 1 {
			seqCap = 1
		}
		for l := range kvCacheK {
			layerKVDim, err := m.LayerKVDim(l)
			if err != nil {
				return output
			}
			// Size the cache to the full sequence so growth never reallocates.
			cacheCap, okCap := checkedProduct(seqCap, layerKVDim)
			if !okCap {
				return output
			}
			kvCacheK[l] = make([]float32, 0, cacheCap)
			kvCacheV[l] = make([]float32, 0, cacheCap)
		}
	}

	// Reusable CPU decode scratch for GQA attention.
	maxHeadDim := headDim
	for i := range m.Layers {
		layerHD, err := m.LayerHeadDim(i)
		if err != nil {
			return output
		}
		if layerHD > maxHeadDim {
			maxHeadDim = layerHD
		}
	}
	maxSeqLen := len(tokenIDs) + maxTokens
	if maxSeqLen < 1 {
		maxSeqLen = 1
	}
	attnOutDim, okAttnOutDim := checkedProduct(numHeads, maxHeadDim)
	if !okAttnOutDim || attnOutDim <= 0 {
		return output
	}
	attnScoresScratch := make([]float32, maxSeqLen)
	attnOutScratch := make([]float32, attnOutDim)

	// Batched CPU prefill: process all prompt tokens together so each weight
	// matrix is read once for the whole prompt instead of once per token. Only
	// engaged on the validated subset (plain KV caches, non-MoE, no Gemma4
	// per-layer inputs); otherwise the sequential loop below handles the prompt.
	startStep := 0
	if compressedKV == nil && maxTokens >= 1 && m.prefillCPUEligible(len(tokenIDs)) {
		if lastHidden, ok := m.prefillCPU(tokenIDs, kvCacheK, kvCacheV); ok {
			_, _, maxIdx, err := m.finishCPUDecodeStep(lastHidden)
			if err != nil {
				panic(err)
			}
			output = append(output, maxIdx)
			startStep = len(tokenIDs)
		}
	}

	// Reusable per-token / per-layer scratch buffers. Each is sized for the
	// widest layer and sliced to the prefix a layer needs; reusing them avoids
	// re-allocating ~10 buffers per layer per generated token.
	scQDim, okScQ := checkedProduct(numHeads, maxHeadDim)
	scKVDim, okScKV := checkedProduct(cfg.NumKVHeads, maxHeadDim)
	if !okScQ || !okScKV {
		return output
	}
	scInter := inter
	for l := 0; l < cfg.NumLayers; l++ {
		lhd, err := m.LayerHeadDim(l)
		if err != nil {
			return output
		}
		lkvh := gemmacfg.LayerKVHeads(cfg, l)
		q, okQ := checkedProduct(numHeads, lhd)
		kv, okKV := checkedProduct(lkvh, lhd)
		if !okQ || !okKV {
			return output
		}
		if q > scQDim {
			scQDim = q
		}
		if kv > scKVDim {
			scKVDim = kv
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
	hidden := make([]float32, h)
	scratchResidual := make([]float32, h)
	scratchQ := make([]float32, scQDim)
	scratchK := make([]float32, scKVDim)
	scratchV := make([]float32, scKVDim)
	scratchO := make([]float32, h)
	scratchMlp := make([]float32, h)
	scratchGate := make([]float32, scInter)
	scratchUp := make([]float32, scInter)
	scratchDown := make([]float32, h)
	var scratchPLIGate, scratchPLIProj []float32
	var pliProjBuf []float32
	var pliSlices [][]float32
	if cfg.HiddenPerLayer > 0 {
		scratchPLIGate = make([]float32, cfg.HiddenPerLayer)
		scratchPLIProj = make([]float32, h)
		if m.PerLayerModelProj != nil {
			if td, ok := checkedProduct(cfg.NumLayers, cfg.HiddenPerLayer); ok {
				pliProjBuf = make([]float32, td)
				pliSlices = make([][]float32, cfg.NumLayers)
			}
		}
	}

	// Process prompt + generate
	for step := startStep; step < len(tokenIDs)+maxTokens-1; step++ {
		var tokID int
		if step < len(tokenIDs) {
			tokID = tokenIDs[step]
		} else {
			tokID = output[len(output)-1]
		}

		// Embed single token using the same helper exposed for verifier/MTP paths.
		if err := m.ScaledTokenEmbeddingInto(hidden, tokID); err != nil {
			panic(err)
		}

		pos := step

		if debugOpHook != nil {
			debugOpHook("cpu", step, 0, "embed_scaled", hidden)
		}

		// Gemma4: compute per-layer inputs for this token using the same helper
		// exposed for verifier/MTP paths.
		perLayerInputs, err := m.Gemma4PerLayerInputsInto(pliProjBuf, pliSlices, hidden, tokID)
		if err != nil {
			panic(err)
		}
		if perLayerInputs != nil {
			if debugCPUPerLayerInputsOverrideHook != nil {
				debugCPUPerLayerInputsOverrideHook(step, perLayerInputs)
			}
			if debugOpHook != nil && len(perLayerInputs) > 0 {
				debugOpHook("cpu", step, 0, "pli0_input", perLayerInputs[0])
			}
		}

		for l := 0; l < cfg.NumLayers; l++ {
			layer := &m.Layers[l]
			if debugCPUHiddenInOverrideHook != nil {
				debugCPUHiddenInOverrideHook(step, l, hidden)
			}
			residual := scratchResidual
			copy(residual, hidden)
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "hidden_in", hidden)
			}

			// RMS Norm (BF16 for Gemma3)
			if cfg.ModelType == "gemma3_text" {
				simd.RMSNormBF16(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
			} else {
				rmsNormInPlace(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "normed", hidden)
			}

			// BF16 embed scaling was already applied above

			// Q, K, V projections (single token: [1, h] @ [h, dim])
			layerHeadDim, err := m.LayerHeadDim(l)
			if err != nil {
				return output
			}
			layerKVHeads := gemmacfg.LayerKVHeads(cfg, l)
			qDim, okQDim := checkedProduct(numHeads, layerHeadDim)
			layerKVDim, okKVDim := checkedProduct(layerKVHeads, layerHeadDim)
			if layerKVHeads <= 0 || !okQDim || !okKVDim {
				return output
			}
			q := scratchQ[:qDim]

			// Always compute Q
			if layer.QWq != nil {
				if !m.mvQ(q, hidden, layer.QWq) {
					return output
				}
			} else if layer.QWm != nil {
				if !mlx.GemvParallel(q, hidden, layer.QWm) {
					return output
				}
			} else if layer.QWGGUF != nil {
				if !gemvGGUFTo(q, hidden, layer.QWGGUF, h, qDim) {
					return output
				}
			} else {
				m.mv(q, hidden, layer.QW.Data(), h, qDim)
			}

			// K, V: only compute for HasKV layers; shared layers reuse source KV cache
			var k, v []float32
			if layer.HasKV {
				k = scratchK[:layerKVDim]
				v = scratchV[:layerKVDim]
				if layer.KWq != nil {
					if !m.mvQ(k, hidden, layer.KWq) {
						return output
					}
					if cfg.AttentionKEqV && (layer.VWq == nil || layer.VWq == layer.KWq) {
						copy(v, k)
					} else if layer.VWq != nil {
						if !m.mvQ(v, hidden, layer.VWq) {
							return output
						}
					} else {
						return output
					}
				} else if layer.KWm != nil {
					if !mlx.GemvParallel(k, hidden, layer.KWm) {
						return output
					}
					if cfg.AttentionKEqV && (layer.VWm == nil || layer.VWm == layer.KWm) {
						copy(v, k)
					} else if layer.VWm != nil {
						if !mlx.GemvParallel(v, hidden, layer.VWm) {
							return output
						}
					} else {
						return output
					}
				} else if layer.KWGGUF != nil {
					if !gemvGGUFTo(k, hidden, layer.KWGGUF, h, layerKVDim) {
						return output
					}
					if cfg.AttentionKEqV && (layer.VWGGUF == nil || layer.VWGGUF == layer.KWGGUF) {
						copy(v, k)
					} else if layer.VWGGUF != nil {
						if !gemvGGUFTo(v, hidden, layer.VWGGUF, h, layerKVDim) {
							return output
						}
					} else {
						return output
					}
				} else {
					if layer.KW == nil {
						return output
					}
					m.mv(k, hidden, layer.KW.Data(), h, layerKVDim)
					if cfg.AttentionKEqV && (layer.VW == nil || layer.VW == layer.KW) {
						copy(v, k)
					} else if layer.VW != nil {
						m.mv(v, hidden, layer.VW.Data(), h, layerKVDim)
					} else {
						return output
					}
				}
			}

			if cfg.ModelType == "gemma3_text" {
				simd.ToBF16(q)
				if k != nil {
					simd.ToBF16(k)
					simd.ToBF16(v)
				}
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "q", q)
				if k != nil {
					debugOpHook("cpu", step, l, "k", k)
					debugOpHook("cpu", step, l, "v", v)
				}
			}

			// Add bias if present (Qwen2)
			if layer.QB != nil {
				qb := layer.QB.Data()
				simd.VecAdd(q, q, qb)
				if k != nil {
					kb, vb := layer.KB.Data(), layer.VB.Data()
					simd.VecAdd(k, k, kb)
					simd.VecAdd(v, v, vb)
				}
			}

			// Select norm function
			normFn := rmsNormInPlace
			if cfg.ModelType == "gemma3_text" {
				normFn = rmsNormBF16
			}

			// V norm (Gemma4: RMSNormNoScale — normalize without weight)
			if cfg.ModelType == "gemma4_text" && v != nil {
				eps := float32(cfg.RMSNormEps)
				for head := 0; head < layerKVHeads; head++ {
					simd.RMSNormNoScale(v[head*layerHeadDim:(head+1)*layerHeadDim], eps)
				}
			} else if layer.VNorm != nil && v != nil {
				vnorm := layer.VNorm.Data()
				for head := 0; head < layerKVHeads; head++ {
					normFn(v[head*layerHeadDim:(head+1)*layerHeadDim], vnorm, float32(cfg.RMSNormEps))
				}
			}

			// QK-Norm (Qwen3/Gemma3/4): RMSNorm each head of Q and K separately
			if layer.QNorm != nil {
				qNorm := layer.QNorm.Data()
				for head := 0; head < numHeads; head++ {
					normFn(q[head*layerHeadDim:(head+1)*layerHeadDim], qNorm, float32(cfg.RMSNormEps))
				}
				if k != nil {
					if layer.KNorm == nil {
						return output
					}
					kNorm := layer.KNorm.Data()
					for head := 0; head < layerKVHeads; head++ {
						normFn(k[head*layerHeadDim:(head+1)*layerHeadDim], kNorm, float32(cfg.RMSNormEps))
					}
				}
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "q_qknorm", q)
				if k != nil {
					debugOpHook("cpu", step, l, "k_qknorm", k)
					debugOpHook("cpu", step, l, "v_attn", v)
				}
			}

			// RoPE on Q (always) and K (only if HasKV).
			if cfg.ModelType == "gemma4_text" {
				freqs, rotHalf := m.ensureGemma4RoPE(l, pos)
				applyRoPEPartial(q, freqs, pos, numHeads, layerHeadDim, rotHalf)
				if k != nil {
					applyRoPEPartial(k, freqs, pos, layerKVHeads, layerHeadDim, rotHalf)
				}
			} else {
				freqs := m.ensureRoPE(pos)
				applyRoPE(q, freqs, pos, numHeads, layerHeadDim)
				if k != nil {
					applyRoPE(k, freqs, pos, layerKVHeads, layerHeadDim)
				}
			}

			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "q_attn", q)
				if k != nil {
					debugOpHook("cpu", step, l, "k_attn", k)
					debugOpHook("cpu", step, l, "v_attn", v)
				}
			}

			// KV cache: append for HasKV layers, reuse source for shared layers
			kvLayer := l
			if !layer.HasKV {
				kvLayer = layer.KVSourceLayer
			}
			if k != nil {
				if compressedKV != nil {
					compressedKV[kvLayer].Append(k, v)
				} else {
					kvCacheK[kvLayer] = append(kvCacheK[kvLayer], k...)
					kvCacheV[kvLayer] = append(kvCacheV[kvLayer], v...)
				}
			}

			// Attention: Q against cached K, V (may be from source layer)
			seqLen := pos + 1
			attnSeqLen := seqLen
			attnKVOffset := 0
			// SWA layers: restrict attention to sliding_window most recent positions
			if cfg.SlidingWindow > 0 && len(cfg.LayerTypes) > l && cfg.LayerTypes[l] == "sliding_attention" {
				if seqLen > cfg.SlidingWindow {
					attnSeqLen = cfg.SlidingWindow
					attnKVOffset = seqLen - cfg.SlidingWindow
				}
			}
			var attnOut []float32
			var kCache, vCache []float32
			if compressedKV != nil {
				kCache = compressedKV[kvLayer].GetK()
				vCache = compressedKV[kvLayer].GetV()
			} else {
				kCache = kvCacheK[kvLayer]
				vCache = kvCacheV[kvLayer]
			}
			attnOut = attnOutScratch[:qDim]
			attnScores := attnScoresScratch[:attnSeqLen]
			scale := attentionScale(cfg, layerHeadDim)
			gqaAttentionHeadsParallelSoftcap(attnOut, attnScores, q, kCache[attnKVOffset*layerKVHeads*layerHeadDim:], vCache[attnKVOffset*layerKVHeads*layerHeadDim:], attnSeqLen, numHeads, layerKVHeads, layerHeadDim, scale, attentionLogitSoftcap(cfg))
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "attn", attnOut)
			}

			// Output projection
			oOut := scratchO
			if layer.OWq != nil {
				if !m.mvQ(oOut, attnOut, layer.OWq) {
					return output
				}
			} else if layer.OWm != nil {
				if !mlx.GemvParallel(oOut, attnOut, layer.OWm) {
					return output
				}
			} else if layer.OWGGUF != nil {
				if !gemvGGUFTo(oOut, attnOut, layer.OWGGUF, qDim, h) {
					return output
				}
			} else {
				m.mv(oOut, attnOut, layer.OW.Data(), qDim, h)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "o", oOut)
			}

			// Gemma3: post-attn norm BEFORE residual add
			if layer.PreFFNNorm != nil {
				// Gemma3 pattern: norm(attn_output), then add residual
				rmsNormInPlace(oOut, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
				for i := range hidden {
					hidden[i] = residual[i] + oOut[i]
				}
				copy(residual, hidden)
			} else {
				// Qwen/LLaMA pattern: add residual, then norm
				simd.VecAdd(hidden, residual, oOut)
				copy(residual, hidden)
				rmsNormInPlace(hidden, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
			}

			// MLP input: preFFNNorm for Gemma3, postNorm already applied for Qwen
			mlpInput := hidden
			if layer.PreFFNNorm != nil {
				mlpInput = scratchMlp
				copy(mlpInput, hidden)
				if cfg.ModelType == "gemma3_text" {
					simd.RMSNormBF16(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
					// mlpInput is already BF16 from RMSNormBF16
				} else {
					rmsNormInPlace(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
				}
			}

			layerInter := inter
			if layer.GateWq != nil && layer.GateWq.OutDim > 0 {
				layerInter = layer.GateWq.OutDim
			} else if layer.GateWm != nil && layer.GateWm.OutDim > 0 {
				layerInter = layer.GateWm.OutDim
			} else if layer.GateW != nil {
				s := layer.GateW.Shape()
				if len(s) >= 2 {
					if m.Large {
						layerInter = s[0]
					} else {
						layerInter = s[1]
					}
				} else if len(s) == 1 && s[0] > 0 {
					layerInter = s[0]
				}
			}

			if debugCPUMLPInputOverrideHook != nil {
				debugCPUMLPInputOverrideHook(step, l, mlpInput)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "mlp_input", mlpInput)
			}

			// MLP: gate * up → SiLU → down (or MoE for expert layers)
			var down []float32
			if layer.IsMoE && layer.ExpertGateW != nil {
				// MoE forward: router → top-k experts → weighted sum, with optional
				// REAP static expert masks applied before top-k selection.
				down = moeForwardWithREAP(mlpInput, layer, cfg, m.REAP, l)
			} else {
				gate := scratchGate[:layerInter]
				up := scratchUp[:layerInter]
				if layer.GateWq != nil {
					if !m.mvQ(gate, mlpInput, layer.GateWq) || !m.mvQ(up, mlpInput, layer.UpWq) {
						return output
					}
				} else if layer.GateWm != nil {
					if !mlx.GemvParallel(gate, mlpInput, layer.GateWm) {
						return output
					}
					if !mlx.GemvParallel(up, mlpInput, layer.UpWm) {
						return output
					}
				} else if layer.GateWGGUF != nil {
					if !gemvGGUFTo(gate, mlpInput, layer.GateWGGUF, h, layerInter) || !gemvGGUFTo(up, mlpInput, layer.UpWGGUF, h, layerInter) {
						return output
					}
				} else {
					m.mv(gate, mlpInput, layer.GateW.Data(), h, layerInter)
					m.mv(up, mlpInput, layer.UpW.Data(), h, layerInter)
				}

				if cfg.ModelType == "gemma3_text" {
					simd.ToBF16(gate)
					simd.ToBF16(up)
				}
				if debugOpHook != nil {
					debugOpHook("cpu", step, l, "gate_pre", gate)
					debugOpHook("cpu", step, l, "up", up)
				}
				// Activation(gate) * up
				if cfg.HiddenAct == "gelu_pytorch_tanh" {
					if cfg.ModelType == "gemma4_text" {
						ggmlGELUMulInPlace(gate, up)
					} else {
						simd.GELUTanhMul(gate, gate, up)
						simd.ToBF16(gate)
					}
				} else {
					simd.VecSiLUMul(gate, gate, up)
				}
				if debugOpHook != nil {
					debugOpHook("cpu", step, l, "gate_act", gate)
				}

				// Down projection
				down = scratchDown
				if layer.DownWq != nil {
					if !m.mvQ(down, gate, layer.DownWq) {
						return output
					}
				} else if layer.DownWm != nil {
					if !mlx.GemvParallel(down, gate, layer.DownWm) {
						return output
					}
				} else if layer.DownWGGUF != nil {
					if !gemvGGUFTo(down, gate, layer.DownWGGUF, layerInter, h) {
						return output
					}
				} else {
					m.mv(down, gate, layer.DownW.Data(), layerInter, h)
				}
			}

			// BF16 down projection output for Gemma3
			if cfg.ModelType == "gemma3_text" {
				simd.ToBF16(down)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "down", down)
			}

			// Post-FFN norm (Gemma3)
			if layer.PostFFNNorm != nil {
				if cfg.ModelType == "gemma3_text" {
					rmsNormBF16(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
				} else {
					rmsNormInPlace(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
				}
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "down_postffn", down)
			}

			// Residual
			simd.VecAdd(hidden, residual, down)
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "hidden_post_ffn", hidden)
			}

			// Per-layer input gating (Gemma4)
			if layer.PLIGate != nil && perLayerInputs != nil && l < len(perLayerInputs) {
				hpl := cfg.HiddenPerLayer
				pli := perLayerInputs[l]
				// gate = gelu(per_layer_input_gate(h)) * per_layer_input → [hiddenPerLayer]
				gate2 := scratchPLIGate[:hpl]
				gemvNT(gate2, hidden, layer.PLIGate, h, hpl)
				if cfg.ModelType == "gemma4_text" {
					ggmlGELUMulInPlace(gate2, pli)
				} else {
					simd.GELUTanhMul(gate2, gate2, pli)
				}
				// proj = per_layer_projection(gate) → [hidden]
				proj2 := scratchPLIProj
				gemvNT(proj2, gate2, layer.PLIProj, hpl, h)
				// norm
				rmsNormInPlace(proj2, layer.PLIPostNorm, float32(cfg.RMSNormEps))
				// residual add
				simd.VecAdd(hidden, hidden, proj2)
			}
			if debugOpHook != nil {
				debugOpHook("cpu", step, l, "hidden_post_pli", hidden)
			}
			// Layer scalar (Gemma4)
			if layer.LayerScalar != 1.0 {
				simd.VecScale(hidden, hidden, layer.LayerScalar)
			}
			if cfg.ModelType == "gemma3_text" {
				simd.ToBF16(hidden)
			}
			if debugLayerHook != nil {
				debugLayerHook("cpu", step, l, hidden)
			}

		}

		// LM head: logits = final_norm(hidden) @ lm_head^T (greedy: take argmax)
		if step >= len(tokenIDs)-1 {
			finalActivation, logits, maxIdx, err := m.finishCPUDecodeStep(hidden)
			if err != nil {
				panic(err)
			}
			if debugLogitsHook != nil {
				debugLogitsHook("cpu", step, finalActivation, logits)
			}
			output = append(output, maxIdx)
		}
	}
	return output
}
