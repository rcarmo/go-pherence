package ideogram4

import (
	"fmt"
	"math"
)

// DiTGlobals holds the non-layer FP8 linears of an Ideogram4 transformer.
type DiTGlobals struct {
	InputProj   *FP8Linear // in_channels -> emb
	LLMCondProj *FP8Linear // llm_features -> emb
	TimeIn      *FP8Linear // emb -> emb
	TimeOut     *FP8Linear // emb -> emb
	AdaLNProj   *FP8Linear // emb -> adaln_dim
	FinalAdaLN  *FP8Linear // adaln_dim -> 2*emb
	FinalLinear *FP8Linear // emb -> in_channels
}

// DiTGlobalsFromSet extracts the global linears from a loaded set.
func DiTGlobalsFromSet(set map[string]*FP8Linear) (DiTGlobals, error) {
	get := func(key string) (*FP8Linear, error) {
		lin, ok := set[key]
		if !ok || lin == nil {
			return nil, fmt.Errorf("ideogram4 DiT missing global %q", key)
		}
		return lin, nil
	}
	var g DiTGlobals
	var err error
	if g.InputProj, err = get("input_proj"); err != nil {
		return DiTGlobals{}, err
	}
	if g.LLMCondProj, err = get("llm_cond_proj"); err != nil {
		return DiTGlobals{}, err
	}
	if g.TimeIn, err = get("t_embedding.mlp_in"); err != nil {
		return DiTGlobals{}, err
	}
	if g.TimeOut, err = get("t_embedding.mlp_out"); err != nil {
		return DiTGlobals{}, err
	}
	if g.AdaLNProj, err = get("adaln_proj"); err != nil {
		return DiTGlobals{}, err
	}
	if g.FinalAdaLN, err = get("final_layer.adaln_modulation"); err != nil {
		return DiTGlobals{}, err
	}
	if g.FinalLinear, err = get("final_layer.linear"); err != nil {
		return DiTGlobals{}, err
	}
	return g, nil
}

// DiTModel is a fully-loaded Ideogram4 transformer (globals + layers) ready for
// a native velocity forward pass.
type DiTModel struct {
	Config  Config
	Globals DiTGlobals
	Layers  []DiTLayer
}

// LoadDiTModel assembles a DiTModel from a loaded FP8 linear set.
func LoadDiTModel(cfg Config, set map[string]*FP8Linear) (*DiTModel, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	globals, err := DiTGlobalsFromSet(set)
	if err != nil {
		return nil, err
	}
	layers := make([]DiTLayer, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		l, err := DiTLayerFromSet(set, i)
		if err != nil {
			return nil, err
		}
		layers[i] = l
	}
	return &DiTModel{Config: cfg, Globals: globals, Layers: layers}, nil
}

// timestepEmbedding builds a standard sinusoidal embedding of dimension dim for
// a scalar timestep t in [0,1] (scaled by 1000 to match diffusion convention).
func timestepEmbedding(t float32, dim int) []float32 {
	out := make([]float32, dim)
	half := dim / 2
	scaled := float64(t) * 1000
	for i := 0; i < half; i++ {
		freq := math.Exp(-math.Log(10000) * float64(i) / float64(half))
		ang := scaled * freq
		out[i] = float32(math.Cos(ang))
		out[half+i] = float32(math.Sin(ang))
	}
	return out
}

// Velocity runs the full DiT forward and returns the predicted velocity for the
// image tokens: shape [imageTokens, in_channels].
//
// Inputs:
//   - latents:      [imageTokens, in_channels] image latent tokens (row-major)
//   - gridH, gridW: latent grid layout of those tokens
//   - textFeatures: [textTokens, llm_features_dim] Qwen3-VL conditioning
//   - timestep:     scalar diffusion time in [0,1]
//
// The text and image tokens form a single joint self-attention sequence
// (Ideogram4 uses one qkv/o per block), with adaLN conditioning derived from
// the timestep embedding.
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

	// --- conditioning vector from timestep embedding ---
	sin := timestepEmbedding(timestep, emb)
	tmid := make([]float32, emb)
	if err := m.Globals.TimeIn.Apply(sin, tmid); err != nil {
		return nil, err
	}
	for i := range tmid {
		tmid[i] = siluScalar(tmid[i])
	}
	temb := make([]float32, emb)
	if err := m.Globals.TimeOut.Apply(tmid, temb); err != nil {
		return nil, err
	}
	for i := range temb {
		temb[i] = siluScalar(temb[i])
	}
	cond := make([]float32, cfg.AdaLNDim)
	if err := m.Globals.AdaLNProj.Apply(temb, cond); err != nil {
		return nil, err
	}

	// --- token embeddings ---
	totalTokens := textTokens + imgTokens
	hidden := make([]float32, totalTokens*emb)
	eps := float32(cfg.NormEps)
	if eps <= 0 {
		eps = 1e-6
	}
	normedFeat := make([]float32, cfg.LLMFeaturesDim)
	for t := 0; t < textTokens; t++ {
		feat := textFeatures[t*cfg.LLMFeaturesDim : (t+1)*cfg.LLMFeaturesDim]
		layerNormTo(normedFeat, feat, eps)
		if err := m.Globals.LLMCondProj.Apply(normedFeat, hidden[t*emb:(t+1)*emb]); err != nil {
			return nil, err
		}
	}
	for t := 0; t < imgTokens; t++ {
		lat := latents[t*cfg.InChannels : (t+1)*cfg.InChannels]
		if err := m.Globals.InputProj.Apply(lat, hidden[(textTokens+t)*emb:(textTokens+t+1)*emb]); err != nil {
			return nil, err
		}
	}

	// --- positions: text prefix sequential temporal, image as 2D grid ---
	positions := make([][3]int, 0, totalTokens)
	for t := 0; t < textTokens; t++ {
		positions = append(positions, [3]int{t, 0, 0})
	}
	for r := 0; r < gridH; r++ {
		for c := 0; c < gridW; c++ {
			positions = append(positions, [3]int{0, r, c})
		}
	}
	rope, err := BuildMRoPEPositions(cfg, positions)
	if err != nil {
		return nil, err
	}

	// --- transformer blocks ---
	for i := range m.Layers {
		if err := m.Layers[i].ForwardLayer(cfg, hidden, cond, rope); err != nil {
			return nil, fmt.Errorf("ideogram4 DiT layer %d: %w", i, err)
		}
	}

	// --- final layer over image tokens ---
	finalMod := make([]float32, 2*emb)
	condAct := make([]float32, cfg.AdaLNDim)
	for i := range cond {
		condAct[i] = siluScalar(cond[i])
	}
	if err := m.Globals.FinalAdaLN.Apply(condAct, finalMod); err != nil {
		return nil, err
	}
	shift, scale := finalMod[0:emb], finalMod[emb:2*emb]
	normed := make([]float32, emb)
	modBuf := make([]float32, emb)
	velocity := make([]float32, imgTokens*cfg.InChannels)
	for t := 0; t < imgTokens; t++ {
		row := hidden[(textTokens+t)*emb : (textTokens+t+1)*emb]
		layerNormTo(normed, row, eps)
		modulate(modBuf, normed, shift, scale)
		if err := m.Globals.FinalLinear.Apply(modBuf, velocity[t*cfg.InChannels:(t+1)*cfg.InChannels]); err != nil {
			return nil, err
		}
	}
	return velocity, nil
}
