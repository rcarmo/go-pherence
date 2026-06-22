package minicpmv

import (
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

type Metadata struct {
	ModelDir        string                            `json:"model_dir"`
	Config          config.MiniCPMVConfig             `json:"config"`
	Summary         config.MiniCPMVSummary            `json:"summary"`
	Processor       *config.MiniCPMVProcessorConfig   `json:"processor,omitempty"`
	Tokenizer       *config.MiniCPMVTokenizerMetadata `json:"tokenizer,omitempty"`
	SpecialTokenIDs *SpecialTokenIDs                  `json:"special_token_ids,omitempty"`
	Tensors         *TensorInventory                  `json:"tensors,omitempty"`
	TensorReadiness *TensorReadiness                  `json:"tensor_readiness,omitempty"`
	ShapeValidation TensorShapeValidation             `json:"shape_validation,omitempty"`
	RuntimePlan     RuntimePlan                       `json:"runtime_plan"`
	VisionPlan      VisionExecutionPlan               `json:"vision_plan"`
	AudioPlan       AudioExecutionPlan                `json:"audio_plan"`
	ResamplerPlan   *ResamplerTensorPlan              `json:"resampler_plan,omitempty"`
}

func LoadMetadata(modelDir string) (Metadata, error) {
	cfg, err := config.ReadMiniCPMVConfig(modelDir)
	if err != nil {
		return Metadata{}, err
	}
	summary := cfg.MiniCPMVSummary()
	meta := Metadata{ModelDir: modelDir, Config: cfg, Summary: summary}
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
	var tensorNames []string
	if inv, ok, err := TensorInventoryFromModelDir(modelDir); err != nil {
		return meta, err
	} else if ok {
		meta.Tensors = &inv
		readiness := TensorReadinessFromInventory(inv)
		meta.TensorReadiness = &readiness
		if infos, err := safetensors.TensorInfosFrom(modelDir, ""); err == nil {
			meta.ShapeValidation = ValidateTensorShapes(summary, infos)
		}
		if names, err := safetensors.NamesFrom(modelDir, ""); err == nil {
			tensorNames = names
			needsKV := summary.VisionHiddenSize > 0 && summary.HiddenSize > 0 && summary.VisionHiddenSize != summary.HiddenSize
			plan := BuildResamplerTensorPlan(names, needsKV)
			meta.ResamplerPlan = &plan
		}
	}
	meta.VisionPlan = BuildVisionExecutionPlan(summary, meta.Tensors)
	meta.AudioPlan = BuildAudioExecutionPlan(summary, tensorNames)
	meta.RuntimePlan = BuildRuntimePlan(summary, meta.Processor, meta.Tokenizer, meta.Tensors)
	return meta, nil
}

func (m Metadata) RequireRuntimeReady() error {
	return m.RuntimePlan.RequireReady()
}
