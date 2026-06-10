package diffusiongemma

import (
	"os"
	"path/filepath"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

type ProcessorMetadata struct {
	TokenizerClass    string `json:"tokenizer_class,omitempty"`
	ProcessorClass    string `json:"processor_class,omitempty"`
	Backend           string `json:"backend,omitempty"`
	BOS               string `json:"bos,omitempty"`
	EOS               string `json:"eos,omitempty"`
	Pad               string `json:"pad,omitempty"`
	Mask              string `json:"mask,omitempty"`
	BOI               string `json:"boi,omitempty"`
	EOI               string `json:"eoi,omitempty"`
	Image             string `json:"image,omitempty"`
	Think             string `json:"think,omitempty"`
	BOT               string `json:"bot,omitempty"`
	EOT               string `json:"eot,omitempty"`
	BOC               string `json:"boc,omitempty"`
	EOC               string `json:"eoc,omitempty"`
	ImageSeqLength    int    `json:"image_seq_length,omitempty"`
	PatchSize         int    `json:"patch_size,omitempty"`
	ChatTemplateBytes int    `json:"chat_template_bytes,omitempty"`
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
		out.BOT = tok.BOTToken
		out.EOT = tok.EOTToken
		out.BOC = tok.BOCToken
		out.EOC = tok.EOCToken
	}
	if proc, ok, err := loaderconfig.ReadDiffusionGemmaProcessorConfig(modelDir); err != nil {
		return out, false, err
	} else if ok {
		seen = true
		out.ProcessorClass = proc.ProcessorClass
		out.ImageSeqLength = proc.ImageSeqLength
		out.PatchSize = proc.PatchSize
	}
	if b, err := os.ReadFile(filepath.Join(modelDir, "chat_template.jinja")); err == nil {
		seen = true
		out.ChatTemplateBytes = len(b)
	} else if !os.IsNotExist(err) {
		return out, false, err
	}
	return out, seen, nil
}
