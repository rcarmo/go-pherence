package omnivoice

import (
	"context"
	"fmt"
	"runtime"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

// Backbone is a single-owner, batch-one OmniVoice forward runtime. It streams
// weights into a fixed arena and supports mixed text/audio token positions.
// No allocation is performed by ForwardInto after construction. It does not
// tokenize text, sample new audio tokens or run the learned waveform codec.
//
// EnableResident optionally builds a shared immutable decoder-layer cache.
// EnableWorkers optionally installs a persistent GEMM pool sized for the
// backbone hidden/intermediate projections. Worker-pool borrowing follows the
// same sequential sibling contract as resident caches: a sibling created after
// workers are enabled borrows that pool, existing siblings are unchanged, and a
// parent must not close an owned pool until borrowed siblings are done with it.
// Resident sharing follows the existing sequential sibling contract and is not
// additionally synchronized for concurrent use.
type Backbone struct {
	projectSpans                        []headSpan
	weights                             *loader.Weights
	directQ8                            []map[string][]byte
	layer                               *loader.LayerBuffer
	resident                            *residentCache
	block                               *Block
	scratch                             *Workspace
	pool                                *simd.GEMMPool
	hidden, row, head, headOutput, norm []float32
	tokens, maxTokens, chunk            int
	poolWorkers                         int
	ownsPool                            bool
}

type headSpan struct{ start, end int }

// NewBackbone borrows weights; caller must keep it open until all calls finish.
// tokens is fixed to bound memory. Audio-head projection is chunked so large
// embedding/head matrices are never materialized in full.
func NewBackbone(weights *loader.Weights, tokens int) (*Backbone, error) {
	return newBackbone(weights, tokens, nil, nil)
}

// NewBackboneSibling shares the streamed weight arena, any enabled resident
// cache, and any worker pool already attached to the parent, but owns all other
// activations. Neither sibling may execute concurrently; intended for
// sequential CFG branches.
func NewBackboneSibling(parent *Backbone, tokens int) (*Backbone, error) {
	if parent == nil {
		return nil, fmt.Errorf("omnivoice: nil parent backbone")
	}
	b, err := newBackbone(parent.weights, tokens, parent.layer, parent.resident)
	if err != nil {
		return nil, err
	}
	b.pool = parent.pool
	b.poolWorkers = parent.poolWorkers
	b.directQ8 = parent.directQ8
	return b, nil
}

func newBackbone(weights *loader.Weights, tokens int, arena *loader.LayerBuffer, resident *residentCache) (*Backbone, error) {
	if weights == nil {
		return nil, fmt.Errorf("omnivoice: nil weights")
	}
	c := weights.Config
	if tokens <= 0 || tokens > c.LLMConfig.MaxPositionEmbeddings {
		return nil, fmt.Errorf("omnivoice: invalid token capacity")
	}
	if err := weights.CheckCodebookOffsets(); err != nil {
		return nil, err
	}
	var err error
	if arena == nil {
		arena, err = weights.NewLayerBuffer()
		if err != nil {
			return nil, err
		}
	}
	block, err := NewBlock(c.LLMConfig, arena.Tensors)
	if err != nil {
		return nil, err
	}
	scratch, err := block.NewWorkspace(tokens)
	if err != nil {
		return nil, err
	}
	norm, err := weights.Float32("llm.norm.weight")
	if err != nil {
		return nil, err
	}
	const chunk = 128
	h := c.LLMConfig.HiddenSize
	b := &Backbone{projectSpans: make([]headSpan, 0, (tokens+audioHeadProjectionRowGroup()-1)/audioHeadProjectionRowGroup()), weights: weights, layer: arena, resident: resident, block: block, scratch: scratch, maxTokens: tokens, chunk: chunk, hidden: make([]float32, tokens*h), row: make([]float32, h), head: make([]float32, chunk*h), headOutput: make([]float32, tokens*chunk), norm: norm}
	if err = b.Reconfigure(tokens); err != nil {
		return nil, err
	}
	return b, nil
}

// Reconfigure narrows or restores the active token views within the original
// reservation. It never grows activations or workspace beyond construction.
func (b *Backbone) Reconfigure(tokens int) error {
	if b == nil {
		return fmt.Errorf("omnivoice: nil backbone")
	}
	if tokens <= 0 || tokens > b.maxTokens {
		return fmt.Errorf("omnivoice: token capacity %d exceeds backbone bound %d", tokens, b.maxTokens)
	}
	if err := b.scratch.Reconfigure(tokens); err != nil {
		return err
	}
	b.tokens = tokens
	h := b.weights.Config.LLMConfig.HiddenSize
	b.hidden = b.hidden[:tokens*h]
	b.headOutput = b.headOutput[:tokens*b.chunk]
	return nil
}

func (b *Backbone) tokenCapacity() int {
	if b == nil {
		return 0
	}
	return b.maxTokens
}

// ForwardInto fills logits in [codebook,time,vocabulary] order, as in upstream.
// ids are [codebook,time]; audioMask is [time]. At text positions only codebook
// zero supplies the text ID. At audio positions every codebook supplies an ID.
// positions/mask follow Block.ForwardInto. Output may not alias input or scratch.
// Cancellation is checked between layers and audio-head chunks; partial output
// after cancellation or any other error must be discarded.
func (b *Backbone) ForwardInto(ctx context.Context, logits []float32, ids []int, audioMask []bool, positions []int, mask []float32) error {
	if b == nil {
		return fmt.Errorf("omnivoice: nil context")
	}
	return b.forwardInto(ctx, logits, ids, audioMask, positions, mask, b.tokens, nil, nil)
}

// ForwardTargetInto matches ForwardInto but projects logits only for the final
// targetFrames positions, returning [codebook,target,vocabulary]. The
// transformer still runs over the full active sequence so attention/position
// semantics remain unchanged.
func (b *Backbone) ForwardTargetInto(ctx context.Context, logits []float32, ids []int, audioMask []bool, positions []int, mask []float32, targetFrames int) error {
	return b.forwardInto(ctx, logits, ids, audioMask, positions, mask, targetFrames, nil, nil)
}

func (b *Backbone) forwardInto(ctx context.Context, logits []float32, ids []int, audioMask []bool, positions []int, mask []float32, targetFrames int, prefix []float32, active []bool) error {
	if b == nil || b.weights == nil || ctx == nil {
		return fmt.Errorf("omnivoice: nil context")
	}
	if targetFrames <= 0 || targetFrames > b.tokens {
		return fmt.Errorf("omnivoice: target frame count %d exceeds active tokens %d", targetFrames, b.tokens)
	}
	c := b.weights.Config
	h := c.LLMConfig.HiddenSize
	books, vocab := c.NumAudioCodebook, c.AudioVocabSize
	inputSize, ok := product(books, b.tokens)
	if !ok || len(ids) != inputSize || len(audioMask) != b.tokens {
		return fmt.Errorf("omnivoice: input shape mismatch")
	}
	logitRows, ok := product(books, targetFrames)
	if !ok {
		return fmt.Errorf("omnivoice: logit shape mismatch")
	}
	logitSize, ok := product(logitRows, vocab)
	if !ok || len(logits) != logitSize {
		return fmt.Errorf("omnivoice: logit shape mismatch")
	}
	if err := b.block.validateInput(b.hidden, b.tokens, positions, mask); err != nil {
		return err
	}
	b.scratch.pool = b.pool
	b.scratch.executionContext = ctx
	defer func() {
		b.scratch.pool = nil
		b.scratch.executionContext = nil
	}()
	if err := ctx.Err(); err != nil {
		return err
	}
	start := 0
	if prefix != nil {
		if len(prefix) != (b.tokens-targetFrames)*h {
			return fmt.Errorf("omnivoice: cached prefix shape mismatch")
		}
		copy(b.hidden, prefix)
		start = b.tokens - targetFrames
	}
	if active != nil && len(active) != targetFrames {
		return fmt.Errorf("omnivoice: active target shape mismatch")
	}
	if err := b.embedRangeInto(b.hidden[start*h:], ids, audioMask, start, b.tokens); err != nil {
		return err
	}
	if b.resident != nil {
		for i := range b.resident.blocks {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := b.resident.blocks[i].ForwardInto(b.hidden, b.hidden, b.tokens, positions, mask, b.scratch); err != nil {
				return err
			}
		}
	} else {
		for i := 0; i < c.LLMConfig.NumHiddenLayers; i++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			var err error
			if b.directQ8 != nil {
				b.block.directQ8 = b.directQ8[i]
				err = b.layer.LoadDirectQ8(b.weights, i)
			} else {
				err = b.layer.Load(b.weights, i)
			}
			if err != nil {
				return err
			}
			if err := b.block.ForwardInto(b.hidden, b.hidden, b.tokens, positions, mask, b.scratch); err != nil {
				return err
			}
		}
	}
	normalizeInto(b.hidden, b.hidden, b.norm, b.tokens, h, float32(c.LLMConfig.RMSNormEps))
	// Head rows are contiguous [codebook*vocab,hidden]. Write transposed output
	// directly into the public [codebook,time,vocab] layout. When projecting only
	// a suffix, start from the architecture's full microkernel row-group boundary
	// so those target rows follow the same packed-GEMM row/tail path as a full
	// sequence projection. This preserves exact suffix parity without a scalar
	// fallback. The row-group size mirrors simd/runtime gebpMR until that detail
	// is exposed as API.
	projectStart := b.audioHeadProjectStart(targetFrames)
	targetStart := b.tokens - targetFrames
	b.prepareHeadSpans(projectStart, targetStart, active)
	total := books * vocab
	for first := 0; first < total; first += b.chunk {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(b.chunk, total-first)
		head := b.head[:count*h]
		if err := b.weights.MatrixRowsInto(head, "audio_heads.weight", first, count); err != nil {
			return err
		}
		for _, span := range b.projectSpans {
			rows := span.end - span.start
			out := b.headOutput[:rows*count]
			if err := b.scratch.linearIntoWithPool(out, b.hidden[span.start*h:span.end*h], head, rows, h, count); err != nil {
				return err
			}
			for j := 0; j < count; j++ {
				book, word := (first+j)/vocab, (first+j)%vocab
				for abs := max(span.start, targetStart); abs < span.end; abs++ {
					t := abs - targetStart
					if active == nil || active[t] {
						logits[(book*targetFrames+t)*vocab+word] = out[(abs-span.start)*count+j]
					}
				}
			}
		}
	}
	return nil
}

func (b *Backbone) audioHeadProjectStart(targetFrames int) int {
	start := b.tokens - targetFrames
	group := audioHeadProjectionRowGroup()
	if group <= 1 || start <= 0 {
		return start
	}
	return start - start%group
}

func audioHeadProjectionRowGroup() int {
	switch runtime.GOARCH {
	case "amd64":
		return 6
	default:
		// ARM64, RISC-V and portable gebp implementations all use four rows.
		return 4
	}
}

// embedRangeInto performs only token lookup and codebook addition, never attention.
// dst has one hidden row per position in [start,end). Callers validate shapes.
func (b *Backbone) embedRangeInto(dst []float32, ids []int, audioMask []bool, start, end int) error {
	c := b.weights.Config
	h, books, vocab := c.LLMConfig.HiddenSize, c.NumAudioCodebook, c.AudioVocabSize
	for t := start; t < end; t++ {
		isAudio := audioMask[t]
		dst := dst[(t-start)*h : (t-start+1)*h]
		if !isAudio {
			id := ids[t]
			if id < 0 || id >= c.LLMConfig.VocabSize {
				return fmt.Errorf("omnivoice: text token out of range")
			}
			if err := b.weights.MatrixRowsInto(dst, "llm.embed_tokens.weight", id, 1); err != nil {
				return err
			}
			continue
		}
		clear(dst)
		for book := 0; book < books; book++ {
			id := ids[book*b.tokens+t]
			if id < 0 || id >= vocab {
				return fmt.Errorf("omnivoice: audio token out of range")
			}
			if err := b.weights.MatrixRowsInto(b.row, "audio_embeddings.weight", book*vocab+id, 1); err != nil {
				return err
			}
			simd.VecAdd(dst, dst, b.row)
		}
	}
	return nil
}

// prepareHeadSpans preserves the full sequence microkernel row/tail grouping.
// Only whole groups with no still-masked time positions may be omitted. Merge
// adjacent groups so dense early steps retain the original GEMM invocation.
func (b *Backbone) prepareHeadSpans(start, targetStart int, active []bool) {
	b.projectSpans = b.projectSpans[:0]
	if active == nil {
		b.projectSpans = append(b.projectSpans, headSpan{start, b.tokens})
		return
	}
	group := audioHeadProjectionRowGroup()
	for first := start; first < b.tokens; first += group {
		end := min(first+group, b.tokens)
		needed := false
		for t := max(first, targetStart); t < end; t++ {
			needed = needed || active[t-targetStart]
		}
		if !needed {
			continue
		}
		n := len(b.projectSpans)
		if n > 0 && b.projectSpans[n-1].end == first {
			b.projectSpans[n-1].end = end
		} else {
			b.projectSpans = append(b.projectSpans, headSpan{first, end})
		}
	}
}
