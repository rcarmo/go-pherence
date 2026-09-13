package omnivoice

import (
	"context"
	"fmt"

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
	weights                             *loader.Weights
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
	b := &Backbone{weights: weights, layer: arena, resident: resident, block: block, scratch: scratch, maxTokens: tokens, chunk: chunk, hidden: make([]float32, tokens*h), row: make([]float32, h), head: make([]float32, chunk*h), headOutput: make([]float32, tokens*chunk), norm: norm}
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
	if b == nil || b.weights == nil || ctx == nil {
		return fmt.Errorf("omnivoice: nil context")
	}
	c := b.weights.Config
	h := c.LLMConfig.HiddenSize
	books, vocab := c.NumAudioCodebook, c.AudioVocabSize
	size, ok := product(books, b.tokens)
	if !ok || len(ids) != size || len(audioMask) != b.tokens {
		return fmt.Errorf("omnivoice: input shape mismatch")
	}
	size, ok = product(size, vocab)
	if !ok || len(logits) != size {
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
	for t, isAudio := range audioMask {
		dst := b.hidden[t*h : (t+1)*h]
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
			if err := b.layer.Load(b.weights, i); err != nil {
				return err
			}
			if err := b.block.ForwardInto(b.hidden, b.hidden, b.tokens, positions, mask, b.scratch); err != nil {
				return err
			}
		}
	}
	normalizeInto(b.hidden, b.hidden, b.norm, b.tokens, h, float32(c.LLMConfig.RMSNormEps))
	// Head rows are contiguous [codebook*vocab,hidden]. Write transposed output
	// directly into the public [codebook,time,vocab] layout.
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
		out := b.headOutput[:b.tokens*count]
		if err := b.scratch.linearIntoWithPool(out, b.hidden, head, b.tokens, h, count); err != nil {
			return err
		}
		for j := 0; j < count; j++ {
			book, word := (first+j)/vocab, (first+j)%vocab
			for t := 0; t < b.tokens; t++ {
				logits[(book*b.tokens+t)*vocab+word] = out[t*count+j]
			}
		}
	}
	return nil
}
