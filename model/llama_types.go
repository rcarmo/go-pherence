package model

import (
	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/loader/gguf"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/common"
	"github.com/rcarmo/go-pherence/runtime/kv"
	"github.com/rcarmo/go-pherence/tensor"
)

// QuantWeight holds GPTQ INT4 weight data for on-the-fly dequantization.
type QuantWeight struct {
	QWeight []int32   // [inDim/8, outDim] packed
	GIdx    []int32   // [inDim] group index
	Scales  []float32 // [numGroups, outDim]
	InDim   int
	OutDim  int
}

type LlamaConfig = common.Config

// LlamaModel holds loaded weights for a LLaMA-style decoder.
type LlamaModel struct {
	Config LlamaConfig
	Tok    *tokenizer.Tokenizer // optional, for chat templates

	EmbedTokens     *tensor.Tensor    // [vocab, hidden]
	EmbedTokensGGUF *gguf.QuantMatrix // [vocab, hidden] quantized GGUF matrix

	EmbedPerLayer      []float32
	EmbedPerLayerGGUF  *gguf.QuantMatrix
	PerLayerModelProj  []float32
	PerLayerProjNorm   []float32
	PerLayerInputScale float32
	PerLayerProjScale  float32
	EmbedPerLayerScale float32
	Norm               *tensor.Tensor
	LMHead             *tensor.Tensor
	LMHeadMLX          *mlx.QuantWeight
	LMHeadGGUF         *gguf.QuantMatrix
	SuppressTokens     []int

	Layers []LlamaLayer

	RopeFreqs     []float32
	RopeFreqsSWA  []float32
	RopeFreqsFull []float32
	RopeHalfSWA   int
	RopeHalfFull  int
	Large         bool
	Quantized     bool
	OnTheFlyQuant bool

	EnableTurboQuant bool
	TurboQuantConfig *kv.TurboQuantConfig
	TurboQuantStates map[int]*kv.TurboQuantState
	REAP             *REAPConfig
}

type LlamaLayer struct {
	InputNorm     *tensor.Tensor
	PostNorm      *tensor.Tensor
	PreFFNNorm    *tensor.Tensor
	VNorm         *tensor.Tensor
	LayerScalar   float32
	HeadDimLocal  int
	HasKV         bool
	KVSourceLayer int

	PLIGate     []float32
	PLIProj     []float32
	PLIGateGGUF *gguf.QuantMatrix
	PLIProjGGUF *gguf.QuantMatrix
	PLIPostNorm []float32
	PostFFNNorm *tensor.Tensor

	QW, KW, VW, OW                 *tensor.Tensor
	QWGGUF, KWGGUF, VWGGUF, OWGGUF *gguf.QuantMatrix
	QB, KB, VB                     *tensor.Tensor
	QNorm, KNorm                   *tensor.Tensor

	QWq, KWq, VWq, OWq   *QuantWeight
	GateWq, UpWq, DownWq *QuantWeight

	QWm, KWm, VWm, OWm   *mlx.QuantWeight
	GateWm, UpWm, DownWm *mlx.QuantWeight

	GateW, UpW, DownW             *tensor.Tensor
	GateWGGUF, UpWGGUF, DownWGGUF *gguf.QuantMatrix

	IsMoE       bool
	RouterW     *mlx.QuantWeight
	ExpertGateW []*mlx.QuantWeight
	ExpertUpW   []*mlx.QuantWeight
	ExpertDownW []*mlx.QuantWeight
}
