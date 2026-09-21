//go:build !ggml || !cgo || !linux

package llamagraph

import "fmt"

type Model struct{}

func New(cfg Config) (*Model, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
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
func (m *Model) TieOutputEmbeddings()              {}
func (m *Model) SetLayerQNorm(il int, data []byte) {}
func (m *Model) SetLayerKNorm(il int, data []byte) {}
func (m *Model) SetMTPENorm(data []byte)           {}
func (m *Model) SetMTPHNorm(data []byte)           {}
func (m *Model) SetMTPEHProj(data []byte)          {}
func (m *Model) SetMTPSharedHeadNorm(data []byte)  {}
func (m *Model) Reset()                            {}
func (m *Model) NPast() int                        { return 0 }
func (m *Model) Close()                            {}
