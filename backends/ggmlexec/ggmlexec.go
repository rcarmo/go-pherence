// Package ggmlexec contains the GGML lowering planner for go-pherence graphs.
//
// The first lowering stage does not try to lower the whole decode graph at once:
// KV-cache, RoPE and attention need persistent graph state, so we segment a graph
// into GGML-capable islands and lower the dense islands first.
package ggmlexec

import (
	"fmt"

	gograph "github.com/rcarmo/go-pherence/runtime/graph"
)

// Capability describes whether a graph op can be lowered to GGML today.
type Capability struct {
	Supported bool
	Reason    string
}

// Capabilities returns the current GGML lowering support table.
func Capabilities() map[gograph.OpKind]Capability {
	return map[gograph.OpKind]Capability{
		gograph.OpRMSNorm:   {Supported: true},
		gograph.OpMatMul:    {Supported: true},
		gograph.OpSiLU:      {Supported: true},
		gograph.OpMul:       {Supported: true},
		gograph.OpAdd:       {Supported: true},
		gograph.OpEmbedding: {Supported: false, Reason: "embedding/gather is model-loader specific; keep in Go until weight tensors are lowered"},
		gograph.OpRoPE:      {Supported: false, Reason: "RoPE depends on position/frequency layout and needs persistent decode-state lowering"},
		gograph.OpKVWrite:   {Supported: false, Reason: "KV cache writes require persistent backend-owned KV tensors"},
		gograph.OpKVRead:    {Supported: false, Reason: "KV cache reads require persistent backend-owned KV tensors"},
		gograph.OpAttention: {Supported: false, Reason: "attention needs KV cache, masking, softmax and grouped-query layout lowering"},
		gograph.OpSoftmax:   {Supported: false, Reason: "standalone softmax supported by GGML but not useful until attention island is lowered"},
		gograph.OpSample:    {Supported: false, Reason: "sampling remains host-side"},
	}
}

// Supports reports whether op can currently be lowered into a GGML island.
func Supports(op gograph.OpKind) bool {
	cap, ok := Capabilities()[op]
	return ok && cap.Supported
}

// Island is a contiguous sequence of supported nodes in a planned graph.
type Island struct {
	Index int
	Start int // inclusive plan step index
	End   int // exclusive plan step index
	Nodes []gograph.Node
}

func (i Island) String() string {
	if len(i.Nodes) == 0 {
		return fmt.Sprintf("island[%d] empty", i.Index)
	}
	return fmt.Sprintf("island[%d] steps=%d..%d nodes=%d first=%s last=%s", i.Index, i.Start, i.End-1, len(i.Nodes), i.Nodes[0].Name, i.Nodes[len(i.Nodes)-1].Name)
}

// FindIslands returns contiguous GGML-supported node runs.
func FindIslands(p *gograph.Plan, minNodes int) []Island {
	if p == nil {
		return nil
	}
	if minNodes <= 0 {
		minNodes = 1
	}
	var out []Island
	start := -1
	var nodes []gograph.Node
	flush := func(end int) {
		if start >= 0 && len(nodes) >= minNodes {
			out = append(out, Island{Index: len(out), Start: start, End: end, Nodes: append([]gograph.Node(nil), nodes...)})
		}
		start = -1
		nodes = nil
	}
	for idx, st := range p.Steps {
		if Supports(st.Node.Op) {
			if start < 0 {
				start = idx
			}
			nodes = append(nodes, st.Node)
			continue
		}
		flush(idx)
	}
	flush(len(p.Steps))
	return out
}

// Coverage summarizes how much of a plan is currently lowerable to GGML.
type Coverage struct {
	TotalNodes     int
	SupportedNodes int
	Unsupported    map[gograph.OpKind]int
	Islands        []Island
}

// Analyze returns lowering coverage for p.
func Analyze(p *gograph.Plan, minIslandNodes int) Coverage {
	c := Coverage{Unsupported: map[gograph.OpKind]int{}}
	if p == nil {
		return c
	}
	c.TotalNodes = len(p.Steps)
	for _, st := range p.Steps {
		if Supports(st.Node.Op) {
			c.SupportedNodes++
		} else {
			c.Unsupported[st.Node.Op]++
		}
	}
	c.Islands = FindIslands(p, minIslandNodes)
	return c
}
