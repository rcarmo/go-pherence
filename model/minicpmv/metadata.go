package minicpmv

import (
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type MetadataOptions struct {
	SafetensorsPath string `json:"safetensors_path,omitempty"`
}

type Metadata struct {
	ModelDir        string                            `json:"model_dir"`
	SafetensorsPath string                            `json:"safetensors_path,omitempty"`
	Config          config.MiniCPMVConfig             `json:"config"`
	Summary         config.MiniCPMVSummary            `json:"summary"`
	Processor       *config.MiniCPMVProcessorConfig   `json:"processor,omitempty"`
	Tokenizer       *config.MiniCPMVTokenizerMetadata `json:"tokenizer,omitempty"`
	Generation      *config.MiniCPMVGenerationConfig  `json:"generation,omitempty"`
	SpecialTokenIDs *SpecialTokenIDs                  `json:"special_token_ids,omitempty"`
	Tensors         *TensorInventory                  `json:"tensors,omitempty"`
	TensorReadiness *TensorReadiness                  `json:"tensor_readiness,omitempty"`
	ShapeValidation TensorShapeValidation             `json:"shape_validation,omitempty"`
	RuntimePlan     RuntimePlan                       `json:"runtime_plan"`
	VisionPlan      VisionExecutionPlan               `json:"vision_plan"`
	AudioPlan       AudioExecutionPlan                `json:"audio_plan"`
	SlicePlan       SlicePlan                         `json:"slice_plan"`
	ResamplerPlan   *ResamplerTensorPlan              `json:"resampler_plan,omitempty"`
}

func LoadMetadata(modelDir string) (Metadata, error) {
	return LoadMetadataWithOptions(modelDir, MetadataOptions{})
}

func LoadMetadataWithOptions(modelDir string, opts MetadataOptions) (Metadata, error) {
	cfg, err := config.ReadMiniCPMVConfig(modelDir)
	if err != nil {
		return Metadata{}, err
	}
	summary := cfg.MiniCPMVSummary()
	meta := Metadata{ModelDir: modelDir, SafetensorsPath: opts.SafetensorsPath, Config: cfg, Summary: summary}
	if proc, ok, err := config.ReadMiniCPMVProcessorConfig(modelDir); err != nil {
		return meta, err
	} else if ok {
		meta.Processor = &proc
	}
	if tok, ok, err := config.ReadMiniCPMVTokenizerMetadata(modelDir); err != nil {
		return meta, err
	} else if ok {
		meta.Tokenizer = &tok
		if ids, err := ResolveSpecialTokenIDs(summary, &tok); err == nil {
			meta.SpecialTokenIDs = &ids
		}
	} else if ids, err := ResolveSpecialTokenIDs(summary, nil); err == nil {
		meta.SpecialTokenIDs = &ids
	}
	if gen, ok, err := config.ReadMiniCPMVGenerationConfig(modelDir); err != nil {
		return meta, err
	} else if ok {
		meta.Generation = &gen
	}
	var tensorNames []string
	if names, err := safetensors.NamesFrom(modelDir, opts.SafetensorsPath); err == nil {
		tensorNames = names
		inv := SummarizeTensors(names)
		meta.Tensors = &inv
		readiness := TensorReadinessFromInventory(inv)
		meta.TensorReadiness = &readiness
		if infos, err := safetensors.TensorInfosFrom(modelDir, opts.SafetensorsPath); err == nil {
			meta.ShapeValidation = ValidateTensorShapes(summary, infos)
		}
		needsKV := summary.VisionHiddenSize > 0 && summary.HiddenSize > 0 && summary.VisionHiddenSize != summary.HiddenSize
		plan := BuildResamplerTensorPlan(names, needsKV)
		meta.ResamplerPlan = &plan
	} else if opts.SafetensorsPath != "" {
		return meta, err
	} else if inv, ok, err := TensorInventoryFromModelDir(modelDir); err != nil {
		return meta, err
	} else if ok {
		meta.Tensors = &inv
		readiness := TensorReadinessFromInventory(inv)
		meta.TensorReadiness = &readiness
	}
	meta.VisionPlan = BuildVisionExecutionPlan(summary, meta.Tensors)
	meta.AudioPlan = BuildAudioExecutionPlan(summary, tensorNames)
	meta.SlicePlan = BuildSlicePlan(summary, summary.ImageSize, summary.ImageSize)
	meta.RuntimePlan = BuildRuntimePlan(summary, meta.Processor, meta.Tokenizer, meta.Tensors)
	return meta, nil
}

func (m Metadata) RequireRuntimeReady() error {
	return m.RuntimePlan.RequireReady()
}
