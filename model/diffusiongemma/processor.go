package diffusiongemma

import (
	"os"
	"path/filepath"
	"strings"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

type ProcessorMetadata struct {
	TokenizerClass    string                `json:"tokenizer_class,omitempty"`
	ProcessorClass    string                `json:"processor_class,omitempty"`
	ImageProcessor    string                `json:"image_processor,omitempty"`
	VideoProcessor    string                `json:"video_processor,omitempty"`
	Backend           string                `json:"backend,omitempty"`
	BOS               string                `json:"bos,omitempty"`
	EOS               string                `json:"eos,omitempty"`
	Pad               string                `json:"pad,omitempty"`
	Mask              string                `json:"mask,omitempty"`
	BOI               string                `json:"boi,omitempty"`
	EOI               string                `json:"eoi,omitempty"`
	Image             string                `json:"image,omitempty"`
	Think             string                `json:"think,omitempty"`
	BOT               string                `json:"bot,omitempty"`
	EOT               string                `json:"eot,omitempty"`
	BOC               string                `json:"boc,omitempty"`
	EOC               string                `json:"eoc,omitempty"`
	ImageSeqLength    int                   `json:"image_seq_length,omitempty"`
	PatchSize         int                   `json:"patch_size,omitempty"`
	PoolingKernelSize int                   `json:"pooling_kernel_size,omitempty"`
	DoConvertRGB      bool                  `json:"do_convert_rgb,omitempty"`
	DoResize          bool                  `json:"do_resize,omitempty"`
	DoRescale         bool                  `json:"do_rescale,omitempty"`
	DoNormalize       bool                  `json:"do_normalize,omitempty"`
	RescaleFactor     float32               `json:"rescale_factor,omitempty"`
	ImageMean         []float32             `json:"image_mean,omitempty"`
	ImageStd          []float32             `json:"image_std,omitempty"`
	ChatTemplateBytes int                   `json:"chat_template_bytes,omitempty"`
	ChatTemplate      *ChatTemplateMetadata `json:"chat_template,omitempty"`
}

type ChatTemplateMetadata struct {
	Path             string   `json:"path"`
	Bytes            int      `json:"bytes"`
	HasSystemRole    bool     `json:"has_system_role"`
	HasUserRole      bool     `json:"has_user_role"`
	HasAssistantRole bool     `json:"has_assistant_role"`
	HasToolSupport   bool     `json:"has_tool_support"`
	HasThinkingToken bool     `json:"has_thinking_token"`
	Markers          []string `json:"markers,omitempty"`
}

func ReadProcessorMetadata(modelDir string) (ProcessorMetadata, bool, error) {
	var out ProcessorMetadata
	seen := false
	if tok, ok, err := loaderconfig.ReadDiffusionGemmaTokenizerConfig(modelDir); err != nil {
		return out, false, err
	} else if ok {
		seen = true
		out.TokenizerClass = tok.TokenizerClass
		out.Backend = tok.Backend
		out.BOS = tok.BOSToken
		out.EOS = tok.EOSToken
		out.Pad = tok.PadToken
		out.Mask = tok.MaskToken
		out.BOI = tok.BOIToken
		out.EOI = tok.EOIToken
		out.Image = tok.ImageToken
		out.Think = tok.ThinkToken
		out.BOT = firstNonEmpty(tok.BOTToken, tok.SOTToken)
		out.EOT = tok.EOTToken
		out.BOC = firstNonEmpty(tok.BOCToken, tok.SOCToken)
		out.EOC = tok.EOCToken
	}
	if proc, ok, err := loaderconfig.ReadDiffusionGemmaProcessorConfig(modelDir); err != nil {
		return out, false, err
	} else if ok {
		seen = true
		out.ProcessorClass = proc.ProcessorClass
		out.ImageProcessor = proc.ImageProcessorType
		out.VideoProcessor = proc.VideoProcessorType
		out.ImageSeqLength = proc.ImageSeqLength
		out.PatchSize = proc.PatchSize
		out.PoolingKernelSize = proc.PoolingKernelSize
		out.DoConvertRGB = proc.DoConvertRGB
		out.DoResize = proc.DoResize
		out.DoRescale = proc.DoRescale
		out.DoNormalize = proc.DoNormalize
		out.RescaleFactor = proc.RescaleFactor
		out.ImageMean = append([]float32(nil), proc.ImageMean...)
		out.ImageStd = append([]float32(nil), proc.ImageStd...)
	}
	if meta, ok, err := ReadChatTemplateMetadata(modelDir); err != nil {
		return out, false, err
	} else if ok {
		seen = true
		out.ChatTemplateBytes = meta.Bytes
		out.ChatTemplate = &meta
	}
	return out, seen, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func ReadChatTemplateMetadata(modelDir string) (ChatTemplateMetadata, bool, error) {
	path := filepath.Join(modelDir, "chat_template.jinja")
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return ChatTemplateMetadata{}, false, nil
		}
		return ChatTemplateMetadata{}, false, err
	}
	s := string(b)
	contains := func(x string) bool { return strings.Contains(s, x) }
	meta := ChatTemplateMetadata{
		Path:             path,
		Bytes:            len(b),
		HasSystemRole:    contains("system"),
		HasUserRole:      contains("user"),
		HasAssistantRole: contains("assistant"),
		HasToolSupport:   contains("tools") || contains("tool_calls") || contains("function"),
		HasThinkingToken: contains("<|think|>") || contains("think"),
	}
	for _, marker := range []string{"<bos>", "<eos>", "<turn|>", "<channel|>", "<|think|>", "tools", "tool_calls"} {
		if contains(marker) {
			meta.Markers = append(meta.Markers, marker)
		}
	}
	return meta, true, nil
}
