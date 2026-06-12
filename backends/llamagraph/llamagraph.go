//go:build ggml && cgo && linux

// Package llamagraph implements a full GGML LLaMA decode-graph executor.
// It builds the complete per-token decode graph (embedding → N×(attn+FFN) →
// LM head) using libggml-base directly, mirroring llama.cpp's internal graph
// builder but driven by go-pherence's own GGUF loader.
//
// Performance target: match cmd/spacemit_llama (~12 tok/s decode on MilkV Jupiter K3)
// by using the same quantised GGML kernels through the same compute path.
package llamagraph

/*
#cgo CFLAGS:  -I/usr/include -I${SRCDIR}/csrc -O3
#cgo LDFLAGS: -lggml -lggml-base -lggml-cpu -lm -lstdc++

#include <stdlib.h>
#include "llamagraph.h"
#include "csrc/llamagraph.c"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

const maxLayers = C.GPLL_MAX_LAYERS

// Config holds all hyper-parameters and per-layer quantisation types needed
// at init time so we can allocate one contiguous weight buffer up front.
type Config struct {
	NVocab, NEmbd, NHeads, NHeadsKV int
	NLayers, NFF, NCtx              int
	RopeBase, RmsEps                float32
	RopeDims, NThreads              int

	// GGML type ints (ggml_type enum) per weight tensor.
	// Use the GGMLType* constants below.
	TokEmbdType int
	OutputType  int
	// Per-layer types (length must be >= NLayers)
	WQType, WKType, WVType, WOType          []int
	FFNGateType, FFNUpType, FFNDownType     []int
	// Output dimensions (0 = use default n_embd / n_embd_kv)
	WQOut, WKOut, WVOut, WOIn []int
	HasQKNorm bool  // Qwen3+ QK norm
}

// GGML type enum constants (mirror ggml_type).
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

// Model is a fully initialised GGML LLaMA model ready for single-token decode.
type Model struct {
	m      *C.gpll_model
	nVocab int
}

// New creates the model, allocates all GGML buffers, and returns it ready
// for SetLayer* / SetOutput* weight-loading calls.
func New(cfg Config) (*Model, error) {
	if cfg.NLayers > maxLayers {
		return nil, fmt.Errorf("NLayers %d exceeds GPLL_MAX_LAYERS %d", cfg.NLayers, maxLayers)
	}

	var wt C.gpll_weight_types
	wt.tok_embd = C.int(cfg.TokEmbdType)
	wt.output   = C.int(cfg.OutputType)
	for il := 0; il < cfg.NLayers; il++ {
		wt.wq[il]       = C.int(cfg.WQType[il])
		wt.wk[il]       = C.int(cfg.WKType[il])
		wt.wv[il]       = C.int(cfg.WVType[il])
		wt.wo[il]       = C.int(cfg.WOType[il])
		wt.ffn_gate[il] = C.int(cfg.FFNGateType[il])
		wt.ffn_up[il]   = C.int(cfg.FFNUpType[il])
		wt.ffn_down[il] = C.int(cfg.FFNDownType[il])
		if cfg.WQOut != nil { wt.wq_out[il] = C.int(cfg.WQOut[il]) }
		if cfg.WKOut != nil { wt.wk_out[il] = C.int(cfg.WKOut[il]) }
		if cfg.WVOut != nil { wt.wv_out[il] = C.int(cfg.WVOut[il]) }
		if cfg.WOIn  != nil { wt.wo_in[il]  = C.int(cfg.WOIn[il])  }
	}

	cm := C.gpll_init(
		C.int(cfg.NVocab), C.int(cfg.NEmbd), C.int(cfg.NHeads), C.int(cfg.NHeadsKV),
		C.int(cfg.NLayers), C.int(cfg.NFF), C.int(cfg.NCtx),
		C.float(cfg.RopeBase), C.float(cfg.RmsEps), C.int(cfg.RopeDims),
		C.int(cfg.NThreads), C.int(boolToInt(cfg.HasQKNorm)), &wt,
	)
	if cm == nil {
		return nil, fmt.Errorf("gpll_init returned nil")
	}
	if cfg.HasQKNorm {
		C.gpll_set_has_qk_norm(cm, 1)
	}
	return &Model{m: cm, nVocab: cfg.NVocab}, nil
}

func cptr(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return unsafe.Pointer(nil)
	}
	return unsafe.Pointer(&b[0])
}

// SetTokEmbd loads the embedding table bytes.
func (m *Model) SetTokEmbd(data []byte) {
	C.gpll_set_tok_embd(m.m, cptr(data), C.size_t(len(data)))
}

// SetOutputNorm loads the final RMSNorm weight (F32 bytes).
func (m *Model) SetOutputNorm(data []byte) {
	C.gpll_set_output_norm(m.m, cptr(data), C.size_t(len(data)))
}

// TieOutputEmbeddings makes output share tok_embd memory (tied embeddings).
func (m *Model) TieOutputEmbeddings() { C.gpll_tie_output_embeddings(m.m) }

// SetOutput loads the LM head weight bytes.
func (m *Model) SetOutput(data []byte) {
	C.gpll_set_output(m.m, cptr(data), C.size_t(len(data)))
}

// SetLayerAttnNorm loads layer il attention RMSNorm (F32 bytes).
func (m *Model) SetLayerAttnNorm(il int, data []byte) {
	C.gpll_set_attn_norm(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}

func (m *Model) SetLayerWQ(il int, data []byte) {
	C.gpll_set_wq(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerWK(il int, data []byte) {
	C.gpll_set_wk(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerWV(il int, data []byte) {
	C.gpll_set_wv(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerWO(il int, data []byte) {
	C.gpll_set_wo(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerFFNNorm(il int, data []byte) {
	C.gpll_set_ffn_norm(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerFFNGate(il int, data []byte) {
	C.gpll_set_ffn_gate(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerFFNUp(il int, data []byte) {
	C.gpll_set_ffn_up(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerFFNDown(il int, data []byte) {
	C.gpll_set_ffn_down(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
// SetLayerQNorm loads layer il Q norm (F32, head_dim).
func (m *Model) SetLayerQNorm(il int, data []byte) {
	C.gpll_set_q_norm(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}
func (m *Model) SetLayerKNorm(il int, data []byte) {
	C.gpll_set_k_norm(m.m, C.int(il), cptr(data), C.size_t(len(data)))
}

// MTP setters
func (m *Model) SetMTPENorm(data []byte)         { C.gpll_set_mtp_enorm(m.m, cptr(data), C.size_t(len(data))) }
func (m *Model) SetMTPHNorm(data []byte)         { C.gpll_set_mtp_hnorm(m.m, cptr(data), C.size_t(len(data))) }
func (m *Model) SetMTPEHProj(data []byte)        { C.gpll_set_mtp_eh_proj(m.m, cptr(data), C.size_t(len(data))) }
func (m *Model) SetMTPSharedHeadNorm(data []byte) { C.gpll_set_mtp_shared_head_norm(m.m, cptr(data), C.size_t(len(data))) }


// Decode runs one decode step for tokenID. Returns logits (len NVocab).
func (m *Model) Decode(tokenID int) ([]float32, error) {
	logits := make([]float32, m.nVocab)
	rc := C.gpll_decode(m.m, C.int(tokenID), (*C.float)(unsafe.Pointer(&logits[0])))
	if rc != 0 {
		return nil, fmt.Errorf("gpll_decode: error %d", int(rc))
	}
	return logits, nil
}

// Reset clears the KV cache and resets position to 0.
func (m *Model) Reset() { C.gpll_reset(m.m) }

// NPast returns the number of tokens decoded so far.
func (m *Model) NPast() int { return int(m.m.n_past) }

// Close frees all GGML resources.
func (m *Model) Close() {
	if m.m != nil {
		C.gpll_free(m.m)
		m.m = nil
	}
}

func boolToInt(b bool) int { if b { return 1 }; return 0 }

