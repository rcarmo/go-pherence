package omnivoice

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// LoadCodecEncoder loads only the non-HuBERT HiggsAudioV2 encoder tensors:
// DAC acoustic encoder, semantic encoder, fusion fc and RVQ quantizer.
func LoadCodecEncoder(path string) (*CodecWeights, error) {
	raw, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return nil, err
	}
	var config struct {
		ModelType    string `json:"model_type"`
		SampleRate   int    `json:"sample_rate"`
		CodebookSize int    `json:"codebook_size"`
		Acoustic     struct {
			Downsampling []int `json:"downsampling_ratios"`
			Upsampling   []int `json:"upsampling_ratios"`
		} `json:"acoustic_model_config"`
	}
	if err = json.Unmarshal(raw, &config); err != nil {
		return nil, err
	}
	if config.ModelType != "higgs_audio_v2_tokenizer" || config.SampleRate <= 0 || config.CodebookSize <= 0 {
		return nil, fmt.Errorf("omnivoice: unsupported codec config")
	}
	rates := append([]int(nil), config.Acoustic.Downsampling...)
	if len(rates) == 0 {
		rates = append(rates, config.Acoustic.Upsampling...)
	}
	f, err := safetensors.Open(filepath.Join(path, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c := &CodecWeights{Tensors: map[string][]float32{}, Shapes: map[string][]int{}, Rates: rates, SampleRate: config.SampleRate, CodebookSize: config.CodebookSize}
	for _, name := range f.Names() {
		wanted := strings.HasPrefix(name, "acoustic_encoder.") || strings.HasPrefix(name, "encoder_semantic.") || strings.HasPrefix(name, "fc.") || (strings.HasPrefix(name, "quantizer.quantizers.") && (strings.HasSuffix(name, "codebook.embed") || strings.Contains(name, ".project_in.") || strings.Contains(name, ".project_out.")))
		if !wanted {
			continue
		}
		data, shape, err := f.GetFloat32(name)
		if err != nil {
			return nil, err
		}
		c.Tensors[name] = data
		c.Shapes[name] = shape
		if strings.HasSuffix(name, "codebook.embed") {
			c.Quantizers++
		}
	}
	if c.Quantizers == 0 || len(c.Rates) == 0 {
		return nil, fmt.Errorf("omnivoice: empty codec encoder")
	}
	return c, nil
}
