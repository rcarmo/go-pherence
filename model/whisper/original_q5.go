package whisper

import (
	"context"
	"fmt"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
	legacy "github.com/rcarmo/go-pherence/loader/whisperggml"
)

// LoadOriginalQ5Model opens a pinned whisper.cpp GGML Q5_0 checkpoint for the
// original-compatible resident Vulkan path. The HF config/generation/tokenizer
// documents are validated first; the GGML vocabulary must match the tokenizer
// text exactly. Encoder FFN values stay packed (packed-only load). The returned
// file is open and caller-owned: pass it to NewVulkanEncoderOriginalQ5, then
// Close it. No GPU allocation or inference happens here.
func LoadOriginalQ5Model(ctx context.Context, ggmlPath, ggmlSHA256 string, modelJSON, generationJSON []byte, tokenizer *Tokenizer) (*Whisper, *CheckedGenerationConfig, *legacy.File, error) {
	if ctx == nil || tokenizer == nil {
		return nil, nil, nil, fmt.Errorf("whisper original Q5: context and tokenizer required")
	}
	cfg, err := ParseModelConfigChecked(modelJSON)
	if err != nil {
		return nil, nil, nil, err
	}
	generation, err := ParseGenerationConfigChecked(generationJSON, cfg, tokenizer)
	if err != nil {
		return nil, nil, nil, err
	}
	file, err := legacy.Open(ctx, ggmlPath, ggmlSHA256)
	if err != nil {
		return nil, nil, nil, err
	}
	fail := func(err error) (*Whisper, *CheckedGenerationConfig, *legacy.File, error) {
		file.Close()
		return nil, nil, nil, err
	}
	vocab := file.Vocabulary()
	if len(vocab) != TokenEOT {
		return fail(fmt.Errorf("whisper original Q5: GGML vocabulary incomplete"))
	}
	for i := 0; i < TokenEOT; i++ {
		if tokenizer.decodeRaw([]int{i}) != string(vocab[i]) {
			return fail(fmt.Errorf("whisper original Q5: GGML/tokenizer vocabulary mismatch at %d", i))
		}
	}
	source, err := newLegacyGGMLSource(ctx, file, cfg)
	if err != nil {
		return fail(err)
	}
	if err := checkOriginalQ5FFNMetadata(file, cfg); err != nil {
		return fail(err)
	}
	model, err := loadModelSourceCheckedMode(ctx, source, cfg, true)
	if err != nil {
		return fail(err)
	}
	return model, generation, file, nil
}

// NewVulkanEncoderOriginalQ5 builds the resident original-compatible encoder:
// Q5_0 x Q8_1 MMQ projections, 48-query padded-extent flash attention and
// tanh-form GELU, with every decoder layer's cross K/V computed in the final
// plan. It then attaches the packed Q5_0 rows to the CPU decoder. Requires a
// prior VulkanInitIntegerDot. Use it with PCMTranscribeOptions
// OriginalDecoderCompatibility and OriginalWindowCompatibility. The host encoder
// weights may be released after success; file stays caller-owned.
func NewVulkanEncoderOriginalQ5(ctx context.Context, model *Whisper, file *legacy.File) (*VulkanEncoder, error) {
	if model == nil || model.Encoder == nil || model.Decoder == nil || file == nil {
		return nil, fmt.Errorf("whisper original Q5: model and GGML file required")
	}
	enc, err := newVulkanEncoderOriginalCross(ctx, model.Encoder, model.Config.MaxLength, vk.NewVkF32Plan, file, model.Decoder)
	if err != nil {
		if enc != nil {
			enc.Close()
		}
		return nil, err
	}
	if err := model.Decoder.attachOriginalQ5(ctx, file); err != nil {
		enc.Close()
		return nil, err
	}
	return enc, nil
}
