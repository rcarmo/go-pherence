package minicpmv

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type FixtureExpectedSummary struct {
	ModelType         string `json:"model_type"`
	Architecture      string `json:"architecture"`
	TextModelType     string `json:"text_model_type"`
	VisionModelType   string `json:"vision_model_type"`
	AudioModelType    string `json:"audio_model_type"`
	HiddenSize        int    `json:"hidden_size"`
	VisionHiddenSize  int    `json:"vision_hidden_size"`
	AudioHiddenSize   int    `json:"audio_hidden_size"`
	NumQuery          int    `json:"num_query"`
	ResamplerGrid     int    `json:"resampler_grid"`
	ImagePatchTokenID int    `json:"image_patch_token_id"`
	AudioPatchTokenID int    `json:"audio_patch_token_id"`
	MetadataReady     bool   `json:"metadata_ready"`
	RuntimeReady      bool   `json:"runtime_ready"`
}

func LoadMiniCPMOFixtureExpectedSummary() (FixtureExpectedSummary, error) {
	paths := []string{
		filepath.Join(MiniCPMOFixturePath, "expected_summary.json"),
		filepath.Join("testdata", "minicpmo_fixture", "expected_summary.json"),
	}
	var lastErr error
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			lastErr = err
			continue
		}
		var out FixtureExpectedSummary
		if err := json.Unmarshal(b, &out); err != nil {
			return out, err
		}
		return out, nil
	}
	return FixtureExpectedSummary{}, lastErr
}
