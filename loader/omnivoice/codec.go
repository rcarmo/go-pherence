package omnivoice

import (
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/loader/safetensors"
	"os"
	"path/filepath"
	"strings"
)

// CodecWeights contains only tensors needed for HiggsAudioV2 waveform decode,
// not HuBERT, encoder, or unused quantizer input projections/statistics.
type CodecWeights struct {
	Tensors                              map[string][]float32
	Shapes                               map[string][]int
	Rates                                []int
	SampleRate, CodebookSize, Quantizers int
}

func LoadCodecDecoder(path string) (*CodecWeights, error) {
	raw, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return nil, err
	}
	var config struct {
		ModelType    string `json:"model_type"`
		SampleRate   int    `json:"sample_rate"`
		CodebookSize int    `json:"codebook_size"`
		Acoustic     struct {
			Rates []int `json:"upsampling_ratios"`
		} `json:"acoustic_model_config"`
	}
	if err = json.Unmarshal(raw, &config); err != nil {
		return nil, err
	}
	if config.ModelType != "higgs_audio_v2_tokenizer" || config.SampleRate <= 0 || config.CodebookSize <= 0 {
		return nil, fmt.Errorf("omnivoice: unsupported codec config")
	}
	f, err := safetensors.Open(filepath.Join(path, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c := &CodecWeights{Tensors: map[string][]float32{}, Shapes: map[string][]int{}, Rates: config.Acoustic.Rates, SampleRate: config.SampleRate, CodebookSize: config.CodebookSize}
	for _, name := range f.Names() {
		wanted := strings.HasPrefix(name, "acoustic_decoder.") || strings.HasPrefix(name, "fc2.") || (strings.HasPrefix(name, "quantizer.quantizers.") && (strings.HasSuffix(name, "codebook.embed") || strings.Contains(name, ".project_out.")))
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
		return nil, fmt.Errorf("omnivoice: empty codec decoder")
	}
	return c, nil
}
