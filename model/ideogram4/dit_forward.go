package ideogram4

import (
	"fmt"
	"math"
)

// imagePositionOffset matches the reference IMAGE_POSITION_OFFSET, keeping image
// (t,h,w) positions disjoint from text positions in the shared MRoPE space.
const imagePositionOffset = 65536

// DiTGlobals holds the non-layer parameters of an Ideogram4 transformer.
type DiTGlobals struct {
	InputProj   *FP8Linear // in_channels -> emb (bias)
	LLMCondProj *FP8Linear // llm_features -> emb (bias)
	TimeIn      *FP8Linear // emb -> emb (bias)
	TimeOut     *FP8Linear // emb -> emb (bias)
	AdaLNProj   *FP8Linear // emb -> adaln_dim (bias)
	FinalAdaLN  *FP8Linear // adaln_dim -> emb (bias)
	FinalLinear *FP8Linear // emb -> in_channels (bias)

	LLMCondNorm    []float32 // RMSNorm over llm_features_dim (eps 1e-6)
	ImageIndicator []float32 // embed_image_indicator [2, emb]
}

// DiTModel is a fully-loaded Ideogram4 transformer ready for a native velocity
// forward pass.
type DiTModel struct {
	Config  Config
	Globals DiTGlobals
	Layers  []DiTLayer
}

// LoadDiTModel assembles a DiTModel (globals, per-layer FP8 linears, RMSNorm
// weights, and the image-indicator embedding) from a tensor source.
func LoadDiTModel(cfg Config, src CombinedTensorSource) (*DiTModel, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	emb := cfg.EmbDim
	loadLin := func(prefix string) (*FP8Linear, error) {
		spec, ok := LinearSpecForPrefix(cfg, prefix)
		if !ok {
			return nil, fmt.Errorf("ideogram4 DiT invalid linear %q", prefix)
		}
		return LoadFP8Linear(src, spec)
	}
	loadNorm := func(name string, n int) ([]float32, error) {
		b, dt, sh, err := src.GetRaw(name)
		if err != nil {
			return nil, fmt.Errorf("ideogram4 DiT norm %q: %w", name, err)
		}
		total := 1
		for _, d := range sh {
			total *= d
		}
		if total != n {
			return nil, fmt.Errorf("ideogram4 DiT norm %q len=%d want=%d", name, total, n)
		}
		return decodeFloatVec(b, dt, n)
	}

	var g DiTGlobals
	var err error
	if g.InputProj, err = loadLin("input_proj"); err != nil {
		return nil, err
	}
	if g.LLMCondProj, err = loadLin("llm_cond_proj"); err != nil {
		return nil, err
	}
	if g.TimeIn, err = loadLin("t_embedding.mlp_in"); err != nil {
		return nil, err
	}
	if g.TimeOut, err = loadLin("t_embedding.mlp_out"); err != nil {
		return nil, err
	}
	if g.AdaLNProj, err = loadLin("adaln_proj"); err != nil {
		return nil, err
	}
	if g.FinalAdaLN, err = loadLin("final_layer.adaln_modulation"); err != nil {
		return nil, err
	}
	if g.FinalLinear, err = loadLin("final_layer.linear"); err != nil {
		return nil, err
	}
	if g.LLMCondNorm, err = loadNorm("llm_cond_norm.weight", cfg.LLMFeaturesDim); err != nil {
		return nil, err
	}
	if g.ImageIndicator, err = loadNorm("embed_image_indicator.weight", 2*emb); err != nil {
		return nil, err
	}

	layers := make([]DiTLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		lp := fmt.Sprintf("layers.%d", i)
		var l DiTLayer
		if l.QKV, err = loadLin(lp + ".attention.qkv"); err != nil {
			return nil, err
		}
		if l.O, err = loadLin(lp + ".attention.o"); err != nil {
			return nil, err
		}
		if l.W1, err = loadLin(lp + ".feed_forward.w1"); err != nil {
			return nil, err
		}
		if l.W2, err = loadLin(lp + ".feed_forward.w2"); err != nil {
			return nil, err
		}
		if l.W3, err = loadLin(lp + ".feed_forward.w3"); err != nil {
			return nil, err
		}
		if l.AdaLN, err = loadLin(lp + ".adaln_modulation"); err != nil {
			return nil, err
		}
		if l.NormQ, err = loadNorm(lp+".attention.norm_q.weight", cfg.HeadDim); err != nil {
			return nil, err
		}
		if l.NormK, err = loadNorm(lp+".attention.norm_k.weight", cfg.HeadDim); err != nil {
			return nil, err
		}
		if l.AttnN1, err = loadNorm(lp+".attention_norm1.weight", emb); err != nil {
			return nil, err
		}
		if l.AttnN2, err = loadNorm(lp+".attention_norm2.weight", emb); err != nil {
			return nil, err
		}
		if l.FfnN1, err = loadNorm(lp+".ffn_norm1.weight", emb); err != nil {
			return nil, err
		}
		if l.FfnN2, err = loadNorm(lp+".ffn_norm2.weight", emb); err != nil {
			return nil, err
		}
		layers[i] = l
	}
	return &DiTModel{Config: cfg, Globals: g, Layers: layers}, nil
}

// sinusoidalEmbedding matches the reference _sinusoidal_embedding:
// freq = exp(arange(half) * -log(scale)/(half-1)); emb = cat[sin(t*freq), cos(t*freq)].
func sinusoidalEmbedding(t float64, dim int, scale float64) []float32 {
	out := make([]float32, dim)
	half := dim / 2
	logScale := math.Log(scale) / float64(half-1)
	for i := 0; i < half; i++ {
		freq := math.Exp(float64(i) * -logScale)
		ang := t * freq
		out[i] = float32(math.Sin(ang))
		out[half+i] = float32(math.Cos(ang))
	}
	return out
}

// adalnInput computes SiLU(adaln_proj(t_embedding(timestep))) per the reference.
// t_embedding: scaled = 1e4*timestep; emb = sinusoidal(scaled, scale=1e4);
// emb = SiLU(mlp_in(emb)); t_cond = mlp_out(emb).
func (m *DiTModel) adalnInput(timestep float32) ([]float32, error) {
	emb := m.Config.EmbDim
	scaled := 1e4 * float64(timestep)
	sin := sinusoidalEmbedding(scaled, emb, 1e4)
	mid := make([]float32, emb)
	if err := m.Globals.TimeIn.Apply(sin, mid); err != nil {
		return nil, err
	}
	for i := range mid {
		mid[i] = siluScalar(mid[i])
	}
	tcond := make([]float32, emb)
	if err := m.Globals.TimeOut.Apply(mid, tcond); err != nil {
		return nil, err
	}
	adaln := make([]float32, m.Config.AdaLNDim)
	if err := m.Globals.AdaLNProj.Apply(tcond, adaln); err != nil {
		return nil, err
	}
	for i := range adaln {
		adaln[i] = siluScalar(adaln[i])
	}
	return adaln, nil
}

// Velocity runs the full DiT forward and returns the predicted velocity for the
// image tokens: [imageTokens, in_channels].
func (m *DiTModel) Velocity(latents []float32, gridH, gridW int, textFeatures []float32, timestep float32) ([]float32, error) {
	if m == nil {
		return nil, ErrRuntimeNotImplemented
	}
	cfg := m.Config
	emb := cfg.EmbDim
	imgTokens := gridH * gridW
	if imgTokens <= 0 || len(latents) != imgTokens*cfg.InChannels {
		return nil, fmt.Errorf("ideogram4 DiT latents=%d want %d*%d", len(latents), imgTokens, cfg.InChannels)
	}
	if cfg.LLMFeaturesDim <= 0 || len(textFeatures)%cfg.LLMFeaturesDim != 0 {
		return nil, fmt.Errorf("ideogram4 DiT text features=%d not divisible by %d", len(textFeatures), cfg.LLMFeaturesDim)
	}
	textTokens := len(textFeatures) / cfg.LLMFeaturesDim

	adaln, err := m.adalnInput(timestep)
	if err != nil {
		return nil, err
	}

	totalTokens := textTokens + imgTokens
	hidden := make([]float32, totalTokens*emb)
	eps := float32(1e-6)
	// text tokens: llm_cond_proj(RMSNorm(features)) + indicator[0].
	normedFeat := make([]float32, cfg.LLMFeaturesDim)
	for t := 0; t < textTokens; t++ {
		feat := textFeatures[t*cfg.LLMFeaturesDim : (t+1)*cfg.LLMFeaturesDim]
		rmsNormWeightedTo(normedFeat, feat, m.Globals.LLMCondNorm, eps)
		row := hidden[t*emb : (t+1)*emb]
		if err := m.Globals.LLMCondProj.Apply(normedFeat, row); err != nil {
			return nil, err
		}
		for i := 0; i < emb; i++ {
			row[i] += m.Globals.ImageIndicator[i] // index 0 = llm token
		}
	}
	// image tokens: input_proj(latents) + indicator[1].
	for t := 0; t < imgTokens; t++ {
		lat := latents[t*cfg.InChannels : (t+1)*cfg.InChannels]
		row := hidden[(textTokens+t)*emb : (textTokens+t+1)*emb]
		if err := m.Globals.InputProj.Apply(lat, row); err != nil {
			return nil, err
		}
		for i := 0; i < emb; i++ {
			row[i] += m.Globals.ImageIndicator[emb+i] // index 1 = image token
		}
	}

	// positions: text (i,i,i); image (0,h,w)+offset.
	positions := make([][3]int, 0, totalTokens)
	for t := 0; t < textTokens; t++ {
		positions = append(positions, [3]int{t, t, t})
	}
	for r := 0; r < gridH; r++ {
		for c := 0; c < gridW; c++ {
			positions = append(positions, [3]int{imagePositionOffset, r + imagePositionOffset, c + imagePositionOffset})
		}
	}
	rope, err := BuildMRoPEPositions(cfg, positions)
	if err != nil {
		return nil, err
	}

	for i := range m.Layers {
		if err := m.Layers[i].ForwardLayer(cfg, hidden, adaln, rope); err != nil {
			return nil, fmt.Errorf("ideogram4 DiT layer %d: %w", i, err)
		}
	}

	// final layer over image tokens: scale = 1 + final_adaln(SiLU(adaln));
	// out = final_linear(LayerNorm_noaffine(h) * scale).
	condAct := make([]float32, cfg.AdaLNDim)
	for i := range adaln {
		condAct[i] = siluScalar(adaln[i])
	}
	scale := make([]float32, emb)
	if err := m.Globals.FinalAdaLN.Apply(condAct, scale); err != nil {
		return nil, err
	}
	for i := range scale {
		scale[i] = 1 + scale[i]
	}
	normed := make([]float32, emb)
	modBuf := make([]float32, emb)
	velocity := make([]float32, imgTokens*cfg.InChannels)
	for t := 0; t < imgTokens; t++ {
		row := hidden[(textTokens+t)*emb : (textTokens+t+1)*emb]
		layerNormNoAffine(normed, row, 1e-6)
		for i := 0; i < emb; i++ {
			modBuf[i] = normed[i] * scale[i]
		}
		if err := m.Globals.FinalLinear.Apply(modBuf, velocity[t*cfg.InChannels:(t+1)*cfg.InChannels]); err != nil {
			return nil, err
		}
	}
	return velocity, nil
}

// layerNormNoAffine computes a non-affine LayerNorm (mean/var over the row).
func layerNormNoAffine(dst, x []float32, eps float32) {
	if gpuNormEnabled() {
		if err := layerNormNoAffineGPU(dst, x, eps); err == nil || gpuNormStrict() {
			return
		}
	}
	layerNormNoAffineCPU(dst, x, eps)
}

func layerNormNoAffineCPU(dst, x []float32, eps float32) {
	n := len(x)
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(n)
	var variance float32
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(n)
	inv := float32(1 / math.Sqrt(float64(variance)+float64(eps)))
	for i := 0; i < n; i++ {
		dst[i] = (x[i] - mean) * inv
	}
}
