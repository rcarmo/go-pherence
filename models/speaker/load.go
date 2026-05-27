package speaker

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// LoadECAPASafetensors loads ECAPA weights from a converted safetensors file.
//
// This is the intended bridge from SpeechBrain/pyannote-style checkpoints to the
// pure-Go diarization path: convert the source checkpoint once into these stable
// tensor names, then load it without bringing a Python runtime into production.
// Expected tensor names and shapes are:
//
//   - conv0.weight: [channels[0], num_mels, kernel_size]
//   - conv0.bias: [channels[0]]
//   - blocks.N.conv.weight: [channels[N+1], channels[N], kernel_size]
//   - blocks.N.conv.bias: [channels[N+1]]
//   - blocks.N.se_down.weight: [se_bottleneck, channels[N+1]]
//   - blocks.N.se_up.weight: [channels[N+1], se_bottleneck]
//   - pool.attn.weight: [attention_dim, channels[-1]]
//   - pool.attn.bias: [attention_dim]
//   - pool.out.weight: [attention_dim]
//   - pool.out.bias: [1]
//   - embed.weight: [embed_dim, channels[-1]*2]
//   - embed.bias: [embed_dim]
func LoadECAPASafetensors(path string, cfg Config) (*ECAPA, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if err := validateECAPAConfig(cfg); err != nil {
		return nil, err
	}

	e := NewECAPA(cfg)
	if e.Conv0Weight, err = getTensor(f, "conv0.weight", cfg.Channels[0], cfg.NumMels, cfg.KernelSize); err != nil {
		return nil, err
	}
	if e.Conv0Bias, err = getTensor(f, "conv0.bias", cfg.Channels[0]); err != nil {
		return nil, err
	}

	for i := range e.Blocks {
		inCh := cfg.Channels[i]
		outCh := cfg.Channels[i+1]
		b := &e.Blocks[i]
		prefix := fmt.Sprintf("blocks.%d", i)
		if b.ConvWeight, err = getTensor(f, prefix+".conv.weight", outCh, inCh, cfg.KernelSize); err != nil {
			return nil, err
		}
		if b.ConvBias, err = getTensor(f, prefix+".conv.bias", outCh); err != nil {
			return nil, err
		}
		if b.SEDown, err = getTensor(f, prefix+".se_down.weight", cfg.SEBottleneck, outCh); err != nil {
			return nil, err
		}
		if b.SEUp, err = getTensor(f, prefix+".se_up.weight", outCh, cfg.SEBottleneck); err != nil {
			return nil, err
		}
	}

	lastCh := cfg.Channels[len(cfg.Channels)-1]
	if e.PoolAttnWeight, err = getTensor(f, "pool.attn.weight", cfg.AttentionDim, lastCh); err != nil {
		return nil, err
	}
	if e.PoolAttnBias, err = getTensor(f, "pool.attn.bias", cfg.AttentionDim); err != nil {
		return nil, err
	}
	if e.PoolOutWeight, err = getTensor(f, "pool.out.weight", cfg.AttentionDim); err != nil {
		return nil, err
	}
	if e.PoolOutBias, err = getTensor(f, "pool.out.bias", 1); err != nil {
		return nil, err
	}
	if e.EmbedWeight, err = getTensor(f, "embed.weight", cfg.EmbedDim, lastCh*2); err != nil {
		return nil, err
	}
	if e.EmbedBias, err = getTensor(f, "embed.bias", cfg.EmbedDim); err != nil {
		return nil, err
	}

	return e, nil
}

func validateECAPAConfig(cfg Config) error {
	if cfg.SampleRate <= 0 || cfg.NumMels <= 0 || cfg.KernelSize <= 0 || cfg.EmbedDim <= 0 || cfg.SEBottleneck <= 0 || cfg.AttentionDim <= 0 {
		return fmt.Errorf("speaker: invalid ECAPA config: %+v", cfg)
	}
	if len(cfg.Channels) < 2 {
		return fmt.Errorf("speaker: ECAPA config needs at least two channel entries")
	}
	for i, ch := range cfg.Channels {
		if ch <= 0 {
			return fmt.Errorf("speaker: invalid ECAPA channel %d: %d", i, ch)
		}
	}
	return nil
}

func getTensor(f *safetensors.File, name string, wantShape ...int) ([]float32, error) {
	data, shape, err := f.GetFloat32(name)
	if err != nil {
		return nil, err
	}
	if !sameShape(shape, wantShape) {
		return nil, fmt.Errorf("speaker: tensor %s shape %v, want %v", name, shape, wantShape)
	}
	return data, nil
}

func sameShape(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
