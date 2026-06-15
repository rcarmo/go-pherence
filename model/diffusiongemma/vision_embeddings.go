package diffusiongemma

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type VisionWeights struct {
	Plan    VisionTensorPlan `json:"plan"`
	Globals []TensorBinding  `json:"globals"`
	Layers  []LayerWeights   `json:"layers"`
	shards  *safetensors.ShardedFile
	infos   map[string]safetensors.TensorInfo
}

type ImageEmbeddingResult struct {
	Embeddings []float32 `json:"-"`
	Shape      [3]int    `json:"shape"` // [images, soft_tokens, text_hidden]
}

func OpenVisionWeights(modelDir string, shape Shape) (*VisionWeights, error) {
	plan, ok, err := VisionTensorPlanFromModelDir(modelDir, shape)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("DiffusionGemma safetensors index not found in %s", modelDir)
	}
	if !plan.Ready {
		return nil, fmt.Errorf("DiffusionGemma vision tensor plan is incomplete: missing %v", plan.Missing)
	}
	shards, err := safetensors.OpenSharded(filepath.Join(modelDir, "model.safetensors.index.json"))
	if err != nil {
		return nil, err
	}
	infos := shards.TensorInfos()
	out := &VisionWeights{Plan: plan, shards: shards, infos: infos}
	for _, h := range plan.Globals {
		b, err := bindTensorHandle(h, infos)
		if err != nil {
			_ = out.Close()
			return nil, err
		}
		out.Globals = append(out.Globals, b)
	}
	for _, lp := range plan.Layers {
		lw := LayerWeights{Layer: lp.Layer, Type: lp.Type}
		for _, h := range lp.Handles {
			b, err := bindTensorHandle(h, infos)
			if err != nil {
				_ = out.Close()
				return nil, err
			}
			lw.Bindings = append(lw.Bindings, b)
		}
		out.Layers = append(out.Layers, lw)
	}
	return out, nil
}

func (w *VisionWeights) Close() error {
	if w == nil || w.shards == nil {
		return nil
	}
	err := w.shards.Close()
	w.shards = nil
	return err
}

func (w *VisionWeights) RawTensor(name string) ([]byte, string, []int, error) {
	if w == nil || w.shards == nil {
		return nil, "", nil, fmt.Errorf("nil DiffusionGemma vision weights")
	}
	return w.shards.GetRaw(name)
}

func (w *VisionWeights) RawBF16Tensor(name string) ([]uint16, []int, error) {
	raw, dtype, shape, err := w.RawTensor(name)
	if err != nil {
		return nil, nil, err
	}
	if dtype != "BF16" {
		return nil, nil, nil
	}
	want, ok := tensorElementCount(shape)
	if !ok || want <= 0 {
		return nil, nil, fmt.Errorf("DiffusionGemma vision tensor %q invalid BF16 shape %v", name, shape)
	}
	needBytes, ok := checked.MulInt(want, 2)
	if !ok || len(raw) < needBytes {
		return nil, nil, fmt.Errorf("DiffusionGemma vision tensor %q BF16 bytes=%d want at least %d for shape %v", name, len(raw), needBytes, shape)
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), want), shape, nil
}

func (w *VisionWeights) CachedFloatTensor(name string) (FloatTensor, error) {
	raw, dtype, shape, err := w.RawTensor(name)
	if err != nil {
		return FloatTensor{}, err
	}
	n, ok := tensorElementCount(shape)
	if !ok || n <= 0 {
		return FloatTensor{}, fmt.Errorf("DiffusionGemma vision tensor %q invalid shape %v", name, shape)
	}
	out := make([]float32, n)
	if err := decodeFloatRowTo(out, raw, dtype); err != nil {
		return FloatTensor{}, err
	}
	return FloatTensor{Data: out, Shape: append([]int(nil), shape...), DType: dtype}, nil
}

func ComputeImageEmbeddings(pre Gemma4ImagePreprocessResult, weights *VisionWeights, shape Shape) (ImageEmbeddingResult, error) {
	return ComputeImageEmbeddingsWithTowerPrefix(pre, weights, shape, nil)
}

func ComputeImageEmbeddingsWithTowerPrefix(pre Gemma4ImagePreprocessResult, weights *VisionWeights, shape Shape, towerPrefix []VisionLayerF32) (ImageEmbeddingResult, error) {
	vision, patches, err := computeImagePatchHidden(pre, weights, shape)
	if err != nil {
		return ImageEmbeddingResult{}, err
	}
	if err := ApplyVisionTowerPrefixF32(vision, patches, shape, towerPrefix); err != nil {
		return ImageEmbeddingResult{}, err
	}
	return projectVisionPatchesToImageEmbeddings(vision, patches, weights, shape)
}

func ComputeImageEmbeddingsWithStreamingTowerPrefix(pre Gemma4ImagePreprocessResult, weights *VisionWeights, shape Shape, prefixLayers int) (ImageEmbeddingResult, error) {
	vision, patches, err := computeImagePatchHidden(pre, weights, shape)
	if err != nil {
		return ImageEmbeddingResult{}, err
	}
	if err := ApplyVisionTowerStreamingPrefixF32(vision, patches, shape, weights, prefixLayers); err != nil {
		return ImageEmbeddingResult{}, err
	}
	return projectVisionPatchesToImageEmbeddings(vision, patches, weights, shape)
}

func ComputeImageEmbeddingsWithFullStreamingTower(pre Gemma4ImagePreprocessResult, weights *VisionWeights, shape Shape) (ImageEmbeddingResult, error) {
	if shape.VisionLayers <= 0 {
		return ImageEmbeddingResult{}, fmt.Errorf("DiffusionGemma full streaming vision tower requires positive vision layers, got %d", shape.VisionLayers)
	}
	patches, err := imagePatchCount(pre, shape)
	if err != nil {
		return ImageEmbeddingResult{}, err
	}
	if max := maxFullStreamingVisionPatches(); max > 0 && patches > max {
		return ImageEmbeddingResult{}, fmt.Errorf("DiffusionGemma full streaming vision tower patch count %d exceeds guarded CPU scaffold limit %d; set GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES to an explicit higher value for reference validation", patches, max)
	}
	return ComputeImageEmbeddingsWithStreamingTowerPrefix(pre, weights, shape, shape.VisionLayers)
}

func imagePatchCount(pre Gemma4ImagePreprocessResult, shape Shape) (int, error) {
	if pre.Width <= 0 || pre.Height <= 0 || shape.PatchSize <= 0 || pre.Width%shape.PatchSize != 0 || pre.Height%shape.PatchSize != 0 {
		return 0, fmt.Errorf("DiffusionGemma image patch count invalid image=%dx%d patch=%d", pre.Width, pre.Height, shape.PatchSize)
	}
	return (pre.Width / shape.PatchSize) * (pre.Height / shape.PatchSize), nil
}

func MaxFullStreamingVisionPatches() int { return maxFullStreamingVisionPatches() }

func fullStreamingVisionPatchLimitOverridden() bool {
	overridden, _ := fullStreamingVisionPatchLimitOverrideState()
	return overridden
}

func fullStreamingVisionPatchLimitOverrideState() (overridden, valid bool) {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES"))
	if v == "" {
		return false, false
	}
	_, err := strconv.Atoi(v)
	return true, err == nil
}

func maxFullStreamingVisionPatches() int {
	const fallback = 64
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES"))
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func computeImagePatchHidden(pre Gemma4ImagePreprocessResult, weights *VisionWeights, shape Shape) ([]float32, int, error) {
	if weights == nil {
		return nil, 0, fmt.Errorf("DiffusionGemma image embeddings missing vision weights")
	}
	if pre.Shape[0] != 1 || pre.Shape[1] != 3 || pre.Width <= 0 || pre.Height <= 0 {
		return nil, 0, fmt.Errorf("DiffusionGemma image embeddings invalid pixel shape %v", pre.Shape)
	}
	if shape.PatchSize <= 0 || pre.Width%shape.PatchSize != 0 || pre.Height%shape.PatchSize != 0 {
		return nil, 0, fmt.Errorf("DiffusionGemma image embeddings image %dx%d not divisible by patch size %d", pre.Width, pre.Height, shape.PatchSize)
	}
	patchW, patchH := pre.Width/shape.PatchSize, pre.Height/shape.PatchSize
	patches := patchW * patchH
	hidden := shape.VisionHiddenSize
	textHidden := shape.TextHiddenSize
	softTokens := shape.VisionSoftTokens
	if patches <= 0 || hidden <= 0 || textHidden <= 0 || softTokens <= 0 {
		return nil, 0, fmt.Errorf("DiffusionGemma image embeddings invalid shape patches=%d hidden=%d text_hidden=%d soft_tokens=%d", patches, hidden, textHidden, softTokens)
	}
	patchVecLen := 3 * shape.PatchSize * shape.PatchSize
	proj, projShape, err := weights.RawBF16Tensor("model.encoder.vision_tower.patch_embedder.input_proj.weight")
	if err != nil {
		return nil, 0, err
	}
	if proj == nil || len(projShape) != 2 || projShape[0] != hidden || projShape[1] != patchVecLen {
		return nil, 0, fmt.Errorf("DiffusionGemma patch projection shape %v want [%d,%d] BF16", projShape, hidden, patchVecLen)
	}
	pos, err := weights.CachedFloatTensor("model.encoder.vision_tower.patch_embedder.position_embedding_table")
	if err != nil {
		return nil, 0, err
	}
	if len(pos.Shape) != 3 || pos.Shape[2] != hidden || pos.Shape[1] < patches {
		return nil, 0, fmt.Errorf("DiffusionGemma position embedding shape %v incompatible with patches=%d hidden=%d", pos.Shape, patches, hidden)
	}
	stdBias, err := weights.CachedFloatTensor("model.encoder.vision_tower.std_bias")
	if err != nil {
		return nil, 0, err
	}
	stdScale, err := weights.CachedFloatTensor("model.encoder.vision_tower.std_scale")
	if err != nil {
		return nil, 0, err
	}
	if len(stdBias.Data) != hidden || len(stdScale.Data) != hidden {
		return nil, 0, fmt.Errorf("DiffusionGemma std scale/bias shape mismatch bias=%v scale=%v hidden=%d", stdBias.Shape, stdScale.Shape, hidden)
	}
	patchHidden, ok := checked.MulInt(patches, hidden)
	if !ok {
		return nil, 0, fmt.Errorf("DiffusionGemma image embedding patch buffer overflow")
	}
	vision := make([]float32, patchHidden)
	patchVec := make([]float32, patchVecLen)
	for py := 0; py < patchH; py++ {
		for px := 0; px < patchW; px++ {
			patchIndex := py*patchW + px
			gemma4FlattenPatchBCHW(patchVec, pre.PixelValues, pre.Width, pre.Height, shape.PatchSize, px, py)
			row := vision[patchIndex*hidden : (patchIndex+1)*hidden]
			if !bf16GemvNarrow(row, patchVec, proj, hidden, patchVecLen) {
				return nil, 0, fmt.Errorf("DiffusionGemma patch projection rejected patch=%d", patchIndex)
			}
			posRow := pos.Data[patchIndex*hidden : (patchIndex+1)*hidden]
			for i := 0; i < hidden; i++ {
				row[i] = (row[i]+posRow[i])*stdScale.Data[i] + stdBias.Data[i]
			}
		}
	}
	return vision, patches, nil
}

func ApplyVisionTowerPrefixF32(vision []float32, patches int, shape Shape, towerPrefix []VisionLayerF32) error {
	if len(towerPrefix) == 0 {
		return nil
	}
	if err := validateVisionTowerPrefixShape(vision, patches, shape); err != nil {
		return err
	}
	headDim := shape.VisionHiddenSize / shape.VisionHeads
	return RunVisionTowerF32(vision, patches, shape.VisionHiddenSize, shape.VisionHeads, headDim, towerPrefix)
}

func ApplyVisionTowerStreamingPrefixF32(vision []float32, patches int, shape Shape, weights *VisionWeights, count int) error {
	if count == 0 {
		return nil
	}
	if err := validateVisionTowerPrefixShape(vision, patches, shape); err != nil {
		return err
	}
	return RunVisionTowerF32StreamingPrefix(vision, patches, shape, weights, count)
}

func validateVisionTowerPrefixShape(vision []float32, patches int, shape Shape) error {
	if shape.VisionHiddenSize <= 0 || shape.VisionHeads <= 0 || len(vision) != patches*shape.VisionHiddenSize {
		return fmt.Errorf("DiffusionGemma vision tower prefix invalid hidden len=%d patches=%d hidden=%d heads=%d", len(vision), patches, shape.VisionHiddenSize, shape.VisionHeads)
	}
	if shape.VisionHiddenSize%shape.VisionHeads != 0 {
		return fmt.Errorf("DiffusionGemma vision hidden=%d not divisible by heads=%d", shape.VisionHiddenSize, shape.VisionHeads)
	}
	return nil
}

func projectVisionPatchesToImageEmbeddings(vision []float32, patches int, weights *VisionWeights, shape Shape) (ImageEmbeddingResult, error) {
	softTokens := shape.VisionSoftTokens
	hidden := shape.VisionHiddenSize
	textHidden := shape.TextHiddenSize
	if softTokens <= 0 || patches < softTokens || patches%softTokens != 0 {
		return ImageEmbeddingResult{}, fmt.Errorf("DiffusionGemma image embeddings cannot pool patches=%d to soft_tokens=%d", patches, softTokens)
	}
	proj, projShape, err := weights.RawBF16Tensor("model.encoder.embed_vision.embedding_projection.weight")
	if err != nil {
		return ImageEmbeddingResult{}, err
	}
	if proj == nil || len(projShape) != 2 || projShape[0] != textHidden || projShape[1] != hidden {
		return ImageEmbeddingResult{}, fmt.Errorf("DiffusionGemma embed_vision projection shape %v want [%d,%d] BF16", projShape, textHidden, hidden)
	}
	pooledVision, err := PoolVisionPatchesToSoftTokensF32(vision, patches, hidden, softTokens)
	if err != nil {
		return ImageEmbeddingResult{}, err
	}
	outLen, ok := checked.MulInt(softTokens, textHidden)
	if !ok {
		return ImageEmbeddingResult{}, fmt.Errorf("DiffusionGemma image embedding output overflow")
	}
	out := make([]float32, outLen)
	projected := make([]float32, textHidden)
	for tok := 0; tok < softTokens; tok++ {
		pooled := pooledVision[tok*hidden : (tok+1)*hidden]
		if !bf16GemvNarrow(projected, pooled, proj, textHidden, hidden) {
			return ImageEmbeddingResult{}, fmt.Errorf("DiffusionGemma embed_vision projection rejected soft token=%d", tok)
		}
		copy(out[tok*textHidden:(tok+1)*textHidden], projected)
	}
	return ImageEmbeddingResult{Embeddings: out, Shape: [3]int{1, softTokens, textHidden}}, nil
}

func gemma4FlattenPatchBCHW(dst, src []float32, width, height, patchSize, patchX, patchY int) {
	_ = height
	pixels := width * height
	i := 0
	for c := 0; c < 3; c++ {
		base := c * pixels
		for y := 0; y < patchSize; y++ {
			row := base + (patchY*patchSize+y)*width + patchX*patchSize
			copy(dst[i:i+patchSize], src[row:row+patchSize])
			i += patchSize
		}
	}
}

func InsertImageEmbeddings(tokenEmbeddings []float32, tokenIDs []int, imageEmbeddings ImageEmbeddingResult, imageTokenID int, hidden int) (int, error) {
	if hidden <= 0 || len(tokenEmbeddings) != len(tokenIDs)*hidden {
		return 0, fmt.Errorf("DiffusionGemma insert image embeddings shape mismatch tokens=%d hidden=%d embeddings=%d", len(tokenIDs), hidden, len(tokenEmbeddings))
	}
	if imageEmbeddings.Shape[2] != hidden || imageEmbeddings.Shape[0] != 1 {
		return 0, fmt.Errorf("DiffusionGemma image embedding shape %v incompatible with hidden=%d", imageEmbeddings.Shape, hidden)
	}
	needed := imageEmbeddings.Shape[1]
	if len(imageEmbeddings.Embeddings) != needed*hidden {
		return 0, fmt.Errorf("DiffusionGemma image embeddings length=%d want %d", len(imageEmbeddings.Embeddings), needed*hidden)
	}
	used := 0
	for i, id := range tokenIDs {
		if id != imageTokenID {
			continue
		}
		if used >= needed {
			return 0, fmt.Errorf("DiffusionGemma prompt has more image tokens than embeddings: used=%d available=%d", used+1, needed)
		}
		copy(tokenEmbeddings[i*hidden:(i+1)*hidden], imageEmbeddings.Embeddings[used*hidden:(used+1)*hidden])
		used++
	}
	if used != needed {
		return 0, fmt.Errorf("DiffusionGemma prompt image token count=%d want %d", used, needed)
	}
	return used, nil
}
