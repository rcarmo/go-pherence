package speaker

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// SpeechBrainECAPA is the real ECAPA-TDNN topology used by
// speechbrain/spkrec-ecapa-voxceleb. It is separate from the older simplified
// ECAPA scaffold so converted checkpoint loading can validate the production
// tensor contract before the forward pass is implemented.
type SpeechBrainECAPA struct {
	Conv0  TDNNLayer // blocks.0, [1024, 80, 5]
	Blocks [3]SpeechBrainECAPABlock
	MFA    TDNNLayer // mfa, [3072, 3072, 1]
	ASP    SpeechBrainASP
	ASPBN  BatchNorm1D // asp_bn, 6144
	FC     Conv1D      // fc, [192, 6144, 1]
}

type SpeechBrainECAPABlock struct {
	TDNN1   TDNNLayer
	Res2Net []TDNNLayer // seven 128-channel 3x1 TDNN sub-blocks in current checkpoint
	TDNN2   TDNNLayer
	SE      SEBlock1D
}

type SpeechBrainASP struct {
	TDNN TDNNLayer // [128, 9216, 1]
	Conv Conv1D    // [3072, 128, 1]
}

type TDNNLayer struct {
	Conv Conv1D
	Norm BatchNorm1D
}

type Conv1D struct {
	Weight []float32
	Bias   []float32
	Shape  []int
}

type BatchNorm1D struct {
	Weight      []float32
	Bias        []float32
	RunningMean []float32
	RunningVar  []float32
}

type SEBlock1D struct {
	Conv1 Conv1D // [128, 1024, 1]
	Conv2 Conv1D // [1024, 128, 1]
}

// LoadSpeechBrainECAPASafetensors loads a converted SpeechBrain ECAPA
// safetensors file preserving SpeechBrain checkpoint key names. It validates the
// known spkrec-ecapa-voxceleb topology but does not yet implement forward.
func LoadSpeechBrainECAPASafetensors(path string) (*SpeechBrainECAPA, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := &SpeechBrainECAPA{}
	if m.Conv0, err = loadTDNNLayer(f, "blocks.0", []int{1024, 80, 5}, 1024); err != nil {
		return nil, err
	}
	for i := 1; i <= 3; i++ {
		b := &m.Blocks[i-1]
		prefix := fmt.Sprintf("blocks.%d", i)
		if b.TDNN1, err = loadTDNNLayer(f, prefix+".tdnn1", []int{1024, 1024, 1}, 1024); err != nil {
			return nil, err
		}
		b.Res2Net = make([]TDNNLayer, 7)
		for j := 0; j < 7; j++ {
			name := fmt.Sprintf("%s.res2net_block.blocks.%d", prefix, j)
			if b.Res2Net[j], err = loadTDNNLayer(f, name, []int{128, 128, 3}, 128); err != nil {
				return nil, err
			}
		}
		if b.TDNN2, err = loadTDNNLayer(f, prefix+".tdnn2", []int{1024, 1024, 1}, 1024); err != nil {
			return nil, err
		}
		if b.SE, err = loadSEBlock1D(f, prefix+".se_block"); err != nil {
			return nil, err
		}
	}
	if m.MFA, err = loadTDNNLayer(f, "mfa", []int{3072, 3072, 1}, 3072); err != nil {
		return nil, err
	}
	if m.ASP.TDNN, err = loadTDNNLayer(f, "asp.tdnn", []int{128, 9216, 1}, 128); err != nil {
		return nil, err
	}
	if m.ASP.Conv, err = loadConv1D(f, "asp.conv", 3072, 128, 1); err != nil {
		return nil, err
	}
	if m.ASPBN, err = loadBatchNorm1D(f, "asp_bn", 6144); err != nil {
		return nil, err
	}
	if m.FC, err = loadConv1D(f, "fc", 192, 6144, 1); err != nil {
		return nil, err
	}
	return m, nil
}

func loadTDNNLayer(f *safetensors.File, prefix string, convShape []int, normDim int) (TDNNLayer, error) {
	conv, err := loadConv1DShape(f, prefix+".conv", convShape)
	if err != nil {
		return TDNNLayer{}, err
	}
	norm, err := loadBatchNorm1D(f, prefix+".norm", normDim)
	if err != nil {
		return TDNNLayer{}, err
	}
	return TDNNLayer{Conv: conv, Norm: norm}, nil
}

func loadSEBlock1D(f *safetensors.File, prefix string) (SEBlock1D, error) {
	conv1, err := loadConv1D(f, prefix+".conv1", 128, 1024, 1)
	if err != nil {
		return SEBlock1D{}, err
	}
	conv2, err := loadConv1D(f, prefix+".conv2", 1024, 128, 1)
	if err != nil {
		return SEBlock1D{}, err
	}
	return SEBlock1D{Conv1: conv1, Conv2: conv2}, nil
}

func loadConv1D(f *safetensors.File, prefix string, dims ...int) (Conv1D, error) {
	return loadConv1DShape(f, prefix, dims)
}

func loadConv1DShape(f *safetensors.File, prefix string, dims []int) (Conv1D, error) {
	w, err := getTensor(f, prefix+".conv.weight", dims...)
	if err != nil {
		return Conv1D{}, err
	}
	b, err := getTensor(f, prefix+".conv.bias", dims[0])
	if err != nil {
		return Conv1D{}, err
	}
	return Conv1D{Weight: w, Bias: b, Shape: append([]int(nil), dims...)}, nil
}

func loadBatchNorm1D(f *safetensors.File, prefix string, dim int) (BatchNorm1D, error) {
	w, err := getTensor(f, prefix+".norm.weight", dim)
	if err != nil {
		return BatchNorm1D{}, err
	}
	b, err := getTensor(f, prefix+".norm.bias", dim)
	if err != nil {
		return BatchNorm1D{}, err
	}
	mean, err := getTensor(f, prefix+".norm.running_mean", dim)
	if err != nil {
		return BatchNorm1D{}, err
	}
	variance, err := getTensor(f, prefix+".norm.running_var", dim)
	if err != nil {
		return BatchNorm1D{}, err
	}
	return BatchNorm1D{Weight: w, Bias: b, RunningMean: mean, RunningVar: variance}, nil
}
