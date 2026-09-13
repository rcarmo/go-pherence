package omnivoice

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const (
	checkpointFloatDType = "F16"
	checkpointIndexName  = "model.safetensors.index.json"
	checkpointFileName   = "model.safetensors"
)

// TensorSpec is the expected checkpoint header entry for one tensor.
type TensorSpec struct {
	DType string  `json:"dtype"`
	Shape []int64 `json:"shape"`
}

// TensorIssue describes a shape or dtype mismatch for one tensor.
type TensorIssue struct {
	Name          string  `json:"name"`
	Problem       string  `json:"problem"`
	ExpectedDType string  `json:"expected_dtype,omitempty"`
	ActualDType   string  `json:"actual_dtype,omitempty"`
	ExpectedShape []int64 `json:"expected_shape,omitempty"`
	ActualShape   []int64 `json:"actual_shape,omitempty"`
}

// CheckpointMetadata summarizes checkpoint-header validation against the
// expected OmniVoice/Qwen3 layout.
type CheckpointMetadata struct {
	Path                string         `json:"path,omitempty"`
	Valid               bool           `json:"valid"`
	TensorCount         int            `json:"tensor_count"`
	ExpectedTensorCount int            `json:"expected_tensor_count"`
	DataBytes           int64          `json:"data_bytes"`
	DTypes              map[string]int `json:"dtypes,omitempty"`
	Missing             []string       `json:"missing,omitempty"`
	Unexpected          []string       `json:"unexpected,omitempty"`
	Issues              []TensorIssue  `json:"issues,omitempty"`
}

// ExpectedShapes returns the expected checkpoint tensor shapes for the supplied
// OmniVoice/Qwen3 config.
func ExpectedShapes(config Config) map[string][]int64 {
	specs, err := expectedTensorSpecs(config)
	if err != nil {
		return nil
	}
	out := make(map[string][]int64, len(specs))
	for name, spec := range specs {
		out[name] = append([]int64(nil), spec.Shape...)
	}
	return out
}

// ValidateTensorInfos validates a safetensors tensor-info map against the
// expected OmniVoice/Qwen3 checkpoint layout.
func ValidateTensorInfos(config Config, infos map[string]safetensors.TensorInfo) CheckpointMetadata {
	meta := CheckpointMetadata{
		TensorCount: len(infos),
		DTypes:      map[string]int{},
	}
	for _, info := range infos {
		meta.DTypes[info.DType]++
		if n := tensorByteLen(info); n > meta.DataBytes {
			meta.DataBytes = n
		}
	}
	specs, err := expectedTensorSpecs(config)
	if err != nil {
		meta.Issues = []TensorIssue{{Problem: err.Error()}}
		return meta
	}
	meta.ExpectedTensorCount = len(specs)

	for name, spec := range specs {
		info, ok := infos[name]
		if !ok {
			meta.Missing = append(meta.Missing, name)
			continue
		}
		actualShape := intsToInt64s(info.Shape)
		if (spec.DType == checkpointFloatDType && info.DType != "F16" && info.DType != "F32" && info.DType != "BF16") || (spec.DType != checkpointFloatDType && info.DType != spec.DType) {
			meta.Issues = append(meta.Issues, TensorIssue{
				Name:          name,
				Problem:       "dtype mismatch",
				ExpectedDType: spec.DType,
				ActualDType:   info.DType,
				ExpectedShape: append([]int64(nil), spec.Shape...),
				ActualShape:   actualShape,
			})
		}
		if !sameShape(actualShape, spec.Shape) {
			meta.Issues = append(meta.Issues, TensorIssue{
				Name:          name,
				Problem:       "shape mismatch",
				ExpectedDType: spec.DType,
				ActualDType:   info.DType,
				ExpectedShape: append([]int64(nil), spec.Shape...),
				ActualShape:   actualShape,
			})
		}
	}
	for name := range infos {
		if _, ok := specs[name]; !ok {
			meta.Unexpected = append(meta.Unexpected, name)
		}
	}
	sort.Strings(meta.Missing)
	sort.Strings(meta.Unexpected)
	sort.Slice(meta.Issues, func(i, j int) bool {
		if meta.Issues[i].Name == meta.Issues[j].Name {
			return meta.Issues[i].Problem < meta.Issues[j].Problem
		}
		return meta.Issues[i].Name < meta.Issues[j].Name
	})
	meta.Valid = len(meta.Missing) == 0 && len(meta.Unexpected) == 0 && len(meta.Issues) == 0
	return meta
}

// ValidateCheckpoint opens a safetensors checkpoint header and validates its
// tensor metadata against the supplied OmniVoice config. path may be a model
// directory, a single .safetensors file, or a sharded index JSON.
func ValidateCheckpoint(path string, config Config) (CheckpointMetadata, error) {
	infos, resolved, err := tensorInfosFromPath(path)
	if err != nil {
		return CheckpointMetadata{}, err
	}
	meta := ValidateTensorInfos(config, infos)
	meta.Path = resolved
	return meta, nil
}

func expectedTensorSpecs(config Config) (map[string]TensorSpec, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	hidden := int64(config.LLMConfig.HiddenSize)
	intermediate := int64(config.LLMConfig.IntermediateSize)
	headDim := int64(config.LLMConfig.HeadDim)
	numLayers := config.LLMConfig.NumHiddenLayers
	qOut, ok := checkedMulInt64(int64(config.LLMConfig.NumAttentionHeads), headDim)
	if !ok {
		return nil, fmt.Errorf("q projection dimension overflow")
	}
	kvOut, ok := checkedMulInt64(int64(config.LLMConfig.NumKeyValueHeads), headDim)
	if !ok {
		return nil, fmt.Errorf("kv projection dimension overflow")
	}
	audioRows, ok := checkedMulInt64(int64(config.AudioVocabSize), int64(config.NumAudioCodebook))
	if !ok {
		return nil, fmt.Errorf("audio embedding dimension overflow")
	}

	out := make(map[string]TensorSpec, 5+numLayers*11)
	out["codebook_layer_offsets"] = TensorSpec{DType: "I64", Shape: []int64{int64(config.NumAudioCodebook)}}
	out["audio_embeddings.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{audioRows, hidden}}
	out["audio_heads.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{audioRows, hidden}}
	out["llm.embed_tokens.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{int64(config.LLMConfig.VocabSize), hidden}}
	out["llm.norm.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{hidden}}

	for layer := 0; layer < numLayers; layer++ {
		prefix := fmt.Sprintf("llm.layers.%d.", layer)
		out[prefix+"input_layernorm.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{hidden}}
		out[prefix+"mlp.down_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{hidden, intermediate}}
		out[prefix+"mlp.gate_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{intermediate, hidden}}
		out[prefix+"mlp.up_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{intermediate, hidden}}
		out[prefix+"post_attention_layernorm.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{hidden}}
		out[prefix+"self_attn.k_norm.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{headDim}}
		out[prefix+"self_attn.k_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{kvOut, hidden}}
		out[prefix+"self_attn.o_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{hidden, qOut}}
		out[prefix+"self_attn.q_norm.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{headDim}}
		out[prefix+"self_attn.q_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{qOut, hidden}}
		out[prefix+"self_attn.v_proj.weight"] = TensorSpec{DType: checkpointFloatDType, Shape: []int64{kvOut, hidden}}
	}
	return out, nil
}

func tensorInfosFromPath(path string) (map[string]safetensors.TensorInfo, string, error) {
	if strings.TrimSpace(path) == "" {
		return nil, "", fmt.Errorf("omnivoice: empty checkpoint path")
	}
	fi, err := os.Stat(path)
	if err != nil {
		return nil, "", fmt.Errorf("omnivoice: stat checkpoint %s: %w", path, err)
	}
	if fi.IsDir() {
		infos, err := safetensors.TensorInfosFrom(path, "")
		if err != nil {
			return nil, "", fmt.Errorf("omnivoice: open checkpoint in %s: %w", path, err)
		}
		indexPath := filepath.Join(path, checkpointIndexName)
		if _, err := os.Stat(indexPath); err == nil {
			return infos, indexPath, nil
		}
		return infos, filepath.Join(path, checkpointFileName), nil
	}
	if strings.HasSuffix(path, checkpointIndexName) {
		sf, err := safetensors.OpenSharded(path)
		if err != nil {
			return nil, "", fmt.Errorf("omnivoice: open sharded checkpoint %s: %w", path, err)
		}
		defer sf.Close()
		return sf.TensorInfos(), path, nil
	}
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, "", fmt.Errorf("omnivoice: open checkpoint %s: %w", path, err)
	}
	defer f.Close()
	return f.TensorInfos(), path, nil
}

func checkedMulInt64(a, b int64) (int64, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > int64(^uint64(0)>>1)/b {
		return 0, false
	}
	return a * b, true
}

func intsToInt64s(in []int) []int64 {
	out := make([]int64, len(in))
	for i, v := range in {
		out[i] = int64(v)
	}
	return out
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func tensorByteLen(info safetensors.TensorInfo) int64 {
	if info.DataOffsets[1] < info.DataOffsets[0] {
		return 0
	}
	return int64(info.DataOffsets[1])
}
