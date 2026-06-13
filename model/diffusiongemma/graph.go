package diffusiongemma

// ExecutionGraphPhase mirrors llama.cpp's DiffusionGemma phase switch:
// UNIFIED is a no-cache [prompt|canvas] graph, PREFILL builds prompt KV, and
// DECODE runs canvas rows against cached prompt KV plus fresh canvas KV.
type ExecutionGraphPhase string

const (
	ExecutionGraphUnified ExecutionGraphPhase = "unified"
	ExecutionGraphPrefill ExecutionGraphPhase = "prefill"
	ExecutionGraphDecode  ExecutionGraphPhase = "decode"
)

// ExecutionGraph describes the effective token region layout for one forward
// graph. P and C follow the llama.cpp comments: prompt rows are [0,P), canvas
// rows are [P,P+C). Decode uses cached prompt KV, so its input rows are only C
// canvas tokens while positions/masks still use P as the canvas offset.
type ExecutionGraph struct {
	Phase        ExecutionGraphPhase `json:"phase"`
	PromptLength int                 `json:"prompt_length"`
	CanvasLength int                 `json:"canvas_length"`
	TokenCount   int                 `json:"token_count"`
}

func BuildExecutionGraph(phase ExecutionGraphPhase, promptLength, canvasLength int) ExecutionGraph {
	if promptLength < 0 {
		promptLength = 0
	}
	if canvasLength < 0 {
		canvasLength = 0
	}
	tokens := promptLength + canvasLength
	switch phase {
	case ExecutionGraphPrefill:
		return ExecutionGraph{Phase: phase, PromptLength: promptLength, CanvasLength: 0, TokenCount: promptLength}
	case ExecutionGraphDecode:
		return ExecutionGraph{Phase: phase, PromptLength: promptLength, CanvasLength: canvasLength, TokenCount: canvasLength}
	default:
		return ExecutionGraph{Phase: ExecutionGraphUnified, PromptLength: promptLength, CanvasLength: canvasLength, TokenCount: tokens}
	}
}

func (g ExecutionGraph) CanvasPosition(row int) int {
	if row < 0 {
		return row
	}
	if g.Phase == ExecutionGraphDecode || g.Phase == ExecutionGraphUnified {
		return g.PromptLength + row
	}
	return row
}
