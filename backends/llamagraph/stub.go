//go:build !ggml || !cgo || !linux

package llamagraph

import "fmt"

const maxLayers = 0

type Config struct {
	NVocab, NEmbd, NHeads, NHeadsKV     int
	NLayers, NFF, NCtx                  int
	RopeBase, RmsEps                    float32
	RopeDims, NThreads                  int
	TokEmbdType                         int
	OutputType                          int
	WQType, WKType, WVType, WOType      []int
	FFNGateType, FFNUpType, FFNDownType []int
}

const (
	GGMLTypeF32  = 0
	GGMLTypeF16  = 1
	GGMLTypeQ4_0 = 2
	GGMLTypeQ4_1 = 3
	GGMLTypeQ4_K = 12
	GGMLTypeQ6_K = 14
	GGMLTypeQ2_K = 10
	GGMLTypeQ3_K = 11
	GGMLTypeQ8_K = 15
)

type Model struct{}

func New(cfg Config) (*Model, error) {
	return nil, fmt.Errorf("llamagraph support not built; rebuild with -tags ggml on a system with GGML headers/libraries")
}
func (m *Model) SetTokEmbd(data []byte)               {}
func (m *Model) SetOutputNorm(data []byte)            {}
func (m *Model) SetOutput(data []byte)                {}
func (m *Model) SetLayerAttnNorm(il int, data []byte) {}
func (m *Model) SetLayerWQ(il int, data []byte)       {}
func (m *Model) SetLayerWK(il int, data []byte)       {}
func (m *Model) SetLayerWV(il int, data []byte)       {}
func (m *Model) SetLayerWO(il int, data []byte)       {}
func (m *Model) SetLayerFFNNorm(il int, data []byte)  {}
func (m *Model) SetLayerFFNGate(il int, data []byte)  {}
func (m *Model) SetLayerFFNUp(il int, data []byte)    {}
func (m *Model) SetLayerFFNDown(il int, data []byte)  {}
func (m *Model) Decode(tokenID int) ([]float32, error) {
	return nil, fmt.Errorf("llamagraph support not built")
}
func (m *Model) Reset()     {}
func (m *Model) NPast() int { return 0 }
func (m *Model) Close()     {}
