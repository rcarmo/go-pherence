package hunyuan3d

import "fmt"

// AddClassAndPosition builds ViT token embeddings from patch embeddings.
// patches=[patchTokens,dim], classToken=[dim], position=[patchTokens+1,dim], dst=[patchTokens+1,dim].
func AddClassAndPosition(dst, patches, classToken, position []float32, patchTokens, dim int) error {
	if patchTokens < 0 || dim <= 0 || len(dst) < (patchTokens+1)*dim || len(patches) < patchTokens*dim || len(classToken) < dim || len(position) < (patchTokens+1)*dim {
		return fmt.Errorf("hunyuan3d vit embed: invalid buffers/dims")
	}
	for d := 0; d < dim; d++ {
		dst[d] = classToken[d] + position[d]
	}
	for t := 0; t < patchTokens; t++ {
		out := dst[(t+1)*dim : (t+2)*dim]
		patch := patches[t*dim : (t+1)*dim]
		pos := position[(t+1)*dim : (t+2)*dim]
		for d := 0; d < dim; d++ {
			out[d] = patch[d] + pos[d]
		}
	}
	return nil
}

type ViTBlockWeights struct {
	Norm1Weight []float32
	QKVWeight   []float32
	QKVBias     []float32
	ProjWeight  []float32
	ProjBias    []float32
	Norm2Weight []float32
	FC1Weight   []float32
	FC1Bias     []float32
	FC2Weight   []float32
	FC2Bias     []float32
}

type ViTBlockConfig struct {
	Tokens int
	Dim    int
	Heads  int
	MLPDim int
	Eps    float32
}

// ViTBlockFloat32 is a CPU reference transformer encoder block using the shared
// Hunyuan3D primitive layer. x and dst are [tokens,dim].
func ViTBlockFloat32(dst, x []float32, cfg ViTBlockConfig, w ViTBlockWeights, scratch []float32) error {
	if err := validateViTBlock(cfg); err != nil {
		return err
	}
	tokens, dim, heads, headDim, mlpDim := cfg.Tokens, cfg.Dim, cfg.Heads, cfg.Dim/cfg.Heads, cfg.MLPDim
	needX := tokens * dim
	if len(dst) < needX || len(x) < needX {
		return fmt.Errorf("hunyuan3d vit block: short x/dst")
	}
	if len(w.Norm1Weight) < dim || len(w.QKVWeight) < 3*dim*dim || len(w.ProjWeight) < dim*dim || len(w.Norm2Weight) < dim || len(w.FC1Weight) < mlpDim*dim || len(w.FC2Weight) < dim*mlpDim {
		return fmt.Errorf("hunyuan3d vit block: short weights")
	}
	needScratch := needX + tokens*3*dim + tokens*dim + tokens*mlpDim + tokens*dim
	if len(scratch) < needScratch {
		return fmt.Errorf("hunyuan3d vit block: short scratch %d want %d", len(scratch), needScratch)
	}
	norm1 := scratch[:needX]
	qkv := scratch[needX : needX+tokens*3*dim]
	attn := scratch[needX+tokens*3*dim : needX+tokens*4*dim]
	mlp := scratch[needX+tokens*4*dim : needX+tokens*4*dim+tokens*mlpDim]
	norm2 := scratch[needX+tokens*4*dim+tokens*mlpDim : needScratch]
	if err := RMSNormFloat32(norm1, x, w.Norm1Weight, tokens, dim, cfg.Eps); err != nil {
		return err
	}
	if err := LinearFloat32(qkv, norm1, w.QKVWeight, w.QKVBias, tokens, dim, 3*dim); err != nil {
		return err
	}
	// Deinterleave [tokens,3*dim] into q/k/v [tokens,heads,headDim].
	q := make([]float32, tokens*heads*headDim)
	k := make([]float32, tokens*heads*headDim)
	v := make([]float32, tokens*heads*headDim)
	for t := 0; t < tokens; t++ {
		row := qkv[t*3*dim : (t+1)*3*dim]
		copy(q[t*dim:(t+1)*dim], row[:dim])
		copy(k[t*dim:(t+1)*dim], row[dim:2*dim])
		copy(v[t*dim:(t+1)*dim], row[2*dim:])
	}
	if err := AttentionFloat32(attn, q, k, v, tokens, tokens, heads, headDim, 0); err != nil {
		return err
	}
	if err := LinearFloat32(dst, attn, w.ProjWeight, w.ProjBias, tokens, dim, dim); err != nil {
		return err
	}
	for i := 0; i < needX; i++ {
		dst[i] += x[i]
	}
	if err := RMSNormFloat32(norm2, dst, w.Norm2Weight, tokens, dim, cfg.Eps); err != nil {
		return err
	}
	if err := LinearFloat32(mlp, norm2, w.FC1Weight, w.FC1Bias, tokens, dim, mlpDim); err != nil {
		return err
	}
	GELUTanhInPlace(mlp)
	if err := LinearFloat32(norm2, mlp, w.FC2Weight, w.FC2Bias, tokens, mlpDim, dim); err != nil {
		return err
	}
	for i := 0; i < needX; i++ {
		dst[i] += norm2[i]
	}
	return nil
}

func validateViTBlock(cfg ViTBlockConfig) error {
	if cfg.Tokens <= 0 || cfg.Dim <= 0 || cfg.Heads <= 0 || cfg.MLPDim <= 0 || cfg.Dim%cfg.Heads != 0 {
		return fmt.Errorf("invalid Hunyuan3D ViT block config: %+v", cfg)
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	return nil
}
