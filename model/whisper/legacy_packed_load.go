package whisper

import (
	"context"
	"fmt"
	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
	"strconv"
	"strings"
)

func isEncoderFFNSourceName(name string) bool {
	prefix := "model.encoder.layers."
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	fields := strings.Split(strings.TrimPrefix(name, prefix), ".")
	if len(fields) != 3 || fields[2] != "weight" || (fields[1] != "fc1" && fields[1] != "fc2") {
		return false
	}
	layer, err := strconv.Atoi(fields[0])
	return err == nil && layer >= 0 && strconv.Itoa(layer) == fields[0]
}
func checkOriginalQ5FFNMetadata(file *legacy.File, c Config) error {
	if err := validatePCMConfig(c); err != nil {
		return err
	}
	if file == nil || c.EncoderLayers < 1 || c.EncoderDModel%32 != 0 || c.EncoderFFNDim%32 != 0 {
		return fmt.Errorf("original packed Whisper: Q5 geometry/source")
	}
	return checkOriginalQ5FFNTensors(file.Tensors(), c)
}
func checkOriginalQ5FFNTensors(inventory []legacy.Tensor, c Config) error {
	if err := validatePCMConfig(c); err != nil {
		return err
	}
	if c.EncoderLayers < 1 || c.EncoderDModel%32 != 0 || c.EncoderFFNDim%32 != 0 {
		return fmt.Errorf("original packed Whisper: Q5 geometry")
	}
	tensors := map[string]legacy.Tensor{}
	for _, t := range inventory {
		tensors[t.Name] = t
	}
	for layer := 0; layer < c.EncoderLayers; layer++ {
		for _, m := range []struct{ part, k, n int }{{0, c.EncoderDModel, c.EncoderFFNDim}, {2, c.EncoderFFNDim, c.EncoderDModel}} {
			name := fmt.Sprintf("encoder.blocks.%d.mlp.%d.weight", layer, m.part)
			t, ok := tensors[name]
			if !ok || t.Type != 6 || len(t.Shape) != 2 || t.Shape[0] != m.k || t.Shape[1] != m.n || t.Bytes != int64(m.k/32*m.n*22) {
				return fmt.Errorf("original packed Whisper: require Q5 tensor %s", name)
			}
		}
	}
	return nil
}

// LoadOriginalQ5ResidentChecked loads a pinned open original GGML file into an
// owned host-only decoder and resident Vulkan encoder, without materialising CPU
// encoder FC1/FC2 arrays. All required metadata is still validated first. Actual
// packed blocks are read/finite-checked during transactional native construction.
// Vulkan must already be explicitly initialised. file is caller-owned and used
// synchronously; neither the returned decoder nor encoder retains it.
// Model.Encoder is nil on success: CPU inference is rejected, never fallback.
// On construction cleanup failure, encoder is returned with error for Close.
func LoadOriginalQ5ResidentChecked(ctx context.Context, file *legacy.File, cfg Config) (model *Whisper, encoder *VulkanEncoder, err error) {
	if ctx == nil {
		return nil, nil, fmt.Errorf("original packed Whisper: nil context")
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, err
	}
	if err = checkOriginalQ5FFNMetadata(file, cfg); err != nil {
		return nil, nil, err
	}
	source, err := newLegacyGGMLSource(ctx, file, cfg)
	if err != nil {
		return nil, nil, err
	}
	model, err = loadModelSourceCheckedMode(ctx, source, cfg, true)
	if err != nil {
		return nil, nil, err
	}
	encoder, err = newVulkanEncoderPackedOnlyMode(ctx, model.Encoder, cfg.MaxLength, vk.NewVkF32Plan, vulkanLinearOriginalQ5MLP, file, true)
	if err != nil {
		return nil, encoder, err
	}
	model.Encoder = nil
	if err = model.ValidatePCMVulkanHostDecoder(ctx, encoder); err != nil {
		closeErr := encoder.Close()
		if closeErr != nil {
			return nil, encoder, fmt.Errorf("original packed Whisper: admission %v; cleanup %w", err, closeErr)
		}
		return nil, nil, err
	}
	return model, encoder, nil
}
