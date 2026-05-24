//go:build !ggml || !cgo || !linux

package ggmlgraph

import (
	"fmt"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

type MulMat struct{ inDim, outDim int }

func NewMulMat(qtype int, raw []byte, inDim, outDim, threads int) (*MulMat, error) {
	return nil, fmt.Errorf("ggmlgraph support not built; rebuild with -tags ggml on a system with GGML headers/libraries")
}
func (m *MulMat) Close() {}
func (m *MulMat) Run(x []float32, out []float32) error {
	return fmt.Errorf("ggmlgraph support not built")
}

type QKV struct{ inDim, qDim, kvDim int }

func NewQKV(wq, wk, wv *gguf.QuantMatrix, threads int) (*QKV, error) {
	return nil, fmt.Errorf("ggmlgraph support not built")
}
func (g *QKV) Close()                         {}
func (g *QKV) Run(x, q, k, v []float32) error { return fmt.Errorf("ggmlgraph support not built") }

type MLP struct{ inDim, ffnDim, outDim int }

func NewMLP(wg, wu, wd *gguf.QuantMatrix, threads int) (*MLP, error) {
	return nil, fmt.Errorf("ggmlgraph support not built")
}
func (g *MLP) Close()                   {}
func (g *MLP) Run(x, y []float32) error { return fmt.Errorf("ggmlgraph support not built") }

type BackendMulMat struct{ inDim, outDim int }

func NewBackendMulMat(qtype int, raw []byte, inDim, outDim int) (*BackendMulMat, error) {
	return nil, fmt.Errorf("ggmlgraph support not built")
}
func (m *BackendMulMat) Close()                     {}
func (m *BackendMulMat) Run(x, out []float32) error { return fmt.Errorf("ggmlgraph support not built") }

type FFNBlock struct{ hidden int }

func NewFFNBlock(norm []float32, wg, wu, wd *gguf.QuantMatrix, eps float32, threads int) (*FFNBlock, error) {
	return nil, fmt.Errorf("ggmlgraph support not built")
}
func (g *FFNBlock) Close()                   {}
func (g *FFNBlock) Run(x, y []float32) error { return fmt.Errorf("ggmlgraph support not built") }
