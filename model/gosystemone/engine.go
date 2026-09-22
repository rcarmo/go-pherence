package gosystemone

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"
)

// Branch is one suffix/trie-prefix path whose final logits are read only at
// CandidateTokens. Tokens must be non-empty.
type Branch struct {
	Tokens          []int
	CandidateTokens []int
}

// ContextScorer owns model/session execution. Implementations may score rows
// sequentially as an oracle or fuse them on a device without changing Go System One's
// schema and trie semantics.
type ContextScorer interface {
	ScoreContext(context.Context, []int, []Branch, bool) ([][]float32, error)
}

// SplitContextScorer preserves the shared-prefix boundary so a native backend
// can snapshot it once and fork context trunks, matching parallel-decision's
// sequence topology. ContextScorer remains the scalar/oracle fallback.
type SplitContextScorer interface {
	ScoreSplitContext(context.Context, []int, []int, []Branch, bool) ([][]float32, error)
}

type Engine struct {
	Tokenizer Tokenizer
	Scorer    ContextScorer
	BOSToken  int
	Now       func() time.Time
}

type fieldState struct {
	compiled      CompiledField
	winner        int
	probabilities []float64
	pathScore     float64
	scoredNodes   int
	active        []int
	chosen        []int
}

func (e *Engine) Decide(ctx context.Context, request Request) (Response, error) {
	if e == nil {
		return Response{}, fmt.Errorf("go-system-one engine is not configured")
	}
	now := e.Now
	if now == nil {
		now = time.Now
	}
	started := now()
	if err := ctx.Err(); err != nil {
		return Response{}, err
	}
	if e.Tokenizer == nil || e.Scorer == nil {
		return Response{}, fmt.Errorf("go-system-one engine is not configured")
	}
	if err := request.NormalizeAndValidate(); err != nil {
		return Response{}, err
	}
	if validator, ok := e.Tokenizer.(UserTextValidator); ok {
		if err := validateRequestUserText(validator, request); err != nil {
			return Response{}, err
		}
	}
	schema, err := CompileSchema(request.Schema, request.Instructions)
	if err != nil {
		return Response{}, err
	}
	fields := make([]CompiledField, len(schema.Inputs))
	for i, input := range schema.Inputs {
		fields[i], err = CompileField(e.Tokenizer, input, request.Mode, request.TreeMax)
		if err != nil {
			return Response{}, fmt.Errorf("field %q: %w", schema.Fields[i].Name, err)
		}
	}
	shared := e.renderShared(schema.SystemText)
	sharedTokens := e.Tokenizer.Encode(shared)
	if e.BOSToken >= 0 {
		sharedTokens = append([]int{e.BOSToken}, sharedTokens...)
	}
	if len(sharedTokens) == 0 {
		return Response{}, fmt.Errorf("decision shared prompt tokenized to empty")
	}

	response := Response{Object: "decision", Model: request.Model, Created: now().Unix(), Results: make([]Result, len(request.Contexts))}
	prefillStart := now()
	contexts := make([][]int, len(request.Contexts))
	contextTokenCounts := make([]int, len(request.Contexts))
	for i, text := range request.Contexts {
		contextTokens := e.Tokenizer.Encode(e.renderContext(text))
		if len(contextTokens) == 0 {
			return Response{}, fmt.Errorf("contexts[%d] tokenized to empty", i)
		}
		promptLen := len(sharedTokens) + len(contextTokens)
		if promptLen > MaxContextTokens {
			return Response{}, fmt.Errorf("contexts[%d] prompt tokens=%d exceeds %d", i, promptLen, MaxContextTokens)
		}
		contexts[i] = contextTokens
		contextTokenCounts[i] = len(contextTokens)
	}
	prefillPrepared := now().Sub(prefillStart)

	scoringStart := now()
	totalRounds := 0
	for i, contextTokens := range contexts {
		states := make([]fieldState, len(fields))
		for f, field := range fields {
			states[f] = newFieldState(field)
		}
		rounds, err := e.scoreFields(ctx, sharedTokens, contextTokens, states, request.AllowCache())
		if err != nil {
			return Response{}, fmt.Errorf("contexts[%d]: %w", i, err)
		}
		totalRounds += rounds
		item, err := assembleResult(schema, states, contextTokenCounts[i])
		if err != nil {
			return Response{}, fmt.Errorf("contexts[%d]: %w", i, err)
		}
		response.Results[i] = item
		response.Usage.ContextTokens += contextTokenCounts[i]
		response.Usage.ScoredRows += item.Usage.ScoredRows
	}
	scoringElapsed := now().Sub(scoringStart)
	if request.AllowCache() {
		response.Usage.CachedTokens = len(sharedTokens)
	}
	response.Usage.PromptTokens = len(sharedTokens) + response.Usage.ContextTokens
	response.Timings.PrefillMS = durationMS(prefillPrepared)
	response.Timings.ScoringMS = durationMS(scoringElapsed)
	response.Timings.TotalMS = durationMS(now().Sub(started))
	response.Timings.Rounds = totalRounds
	if len(response.Results) > 0 {
		response.Timings.PerDecisionMS = response.Timings.TotalMS / float64(len(response.Results))
	}
	return response, nil
}

func validateRequestUserText(validator UserTextValidator, request Request) error {
	if err := validator.ValidateUserText(string(request.Schema)); err != nil {
		return fmt.Errorf("schema: %w", err)
	}
	if err := validator.ValidateUserText(request.Instructions); err != nil {
		return fmt.Errorf("instructions: %w", err)
	}
	for i, text := range request.Contexts {
		if err := validator.ValidateUserText(text); err != nil {
			return fmt.Errorf("contexts[%d]: %w", i, err)
		}
	}
	return nil
}

func (e *Engine) renderShared(system string) string {
	return "<|turn>system\n" + strings.TrimSpace(system) + "<turn|>\n<|turn>user\n"
}

func (e *Engine) renderContext(context string) string {
	return strings.TrimSpace(context) + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>{\n"
}

func newFieldState(field CompiledField) fieldState {
	state := fieldState{compiled: field, winner: -1, pathScore: 1}
	state.active = make([]int, len(field.Paths))
	for i := range state.active {
		state.active[i] = i
	}
	if len(state.active) == 1 {
		state.winner = state.active[0]
	}
	return state
}

func (e *Engine) scoreFields(ctx context.Context, shared, contextTokens []int, states []fieldState, allowCache bool) (int, error) {
	rounds := 0
	first := true
	for {
		var branches []Branch
		type owner struct{ field, node int }
		var owners []owner
		for f := range states {
			state := &states[f]
			if state.compiled.Tree {
				if !first {
					continue
				}
				for n, node := range state.compiled.Nodes {
					tokens := append(append([]int(nil), state.compiled.Suffix...), node.Prefix...)
					branches = append(branches, Branch{Tokens: tokens, CandidateTokens: append([]int(nil), node.Options...)})
					owners = append(owners, owner{field: f, node: n})
				}
				continue
			}
			options := state.greedyOptions()
			if len(options) > 0 {
				tokens := append(append([]int(nil), state.compiled.Suffix...), state.chosen...)
				branches = append(branches, Branch{Tokens: tokens, CandidateTokens: options})
				owners = append(owners, owner{field: f, node: -1})
			}
		}
		if len(branches) == 0 {
			break
		}
		var scores [][]float32
		var err error
		if split, ok := e.Scorer.(SplitContextScorer); ok {
			scores, err = split.ScoreSplitContext(ctx, shared, contextTokens, branches, allowCache)
		} else {
			prompt := append(append([]int(nil), shared...), contextTokens...)
			scores, err = e.Scorer.ScoreContext(ctx, prompt, branches, allowCache)
		}
		if err != nil {
			return rounds, err
		}
		if len(scores) != len(branches) {
			return rounds, fmt.Errorf("scorer rows=%d, want %d", len(scores), len(branches))
		}
		treeScores := make([][][]float32, len(states))
		for row, own := range owners {
			if own.node >= 0 {
				treeScores[own.field] = append(treeScores[own.field], scores[row])
				continue
			}
			state := &states[own.field]
			if len(scores[row]) != len(branches[row].CandidateTokens) {
				return rounds, fmt.Errorf("field %d logits=%d, want %d", own.field, len(scores[row]), len(branches[row].CandidateTokens))
			}
			best := 0
			for j := 1; j < len(scores[row]); j++ {
				if scores[row][j] > scores[row][best] {
					best = j
				}
			}
			probability := constrainedProbability(scores[row], best)
			if err := state.selectGreedy(branches[row].CandidateTokens[best], probability); err != nil {
				return rounds, fmt.Errorf("field %d: %w", own.field, err)
			}
		}
		if first {
			for f := range states {
				state := &states[f]
				if !state.compiled.Tree {
					continue
				}
				winner, probabilities, err := FinishTree(state.compiled, treeScores[f])
				if err != nil {
					return rounds, fmt.Errorf("field %d: %w", f, err)
				}
				state.winner = winner
				state.probabilities = probabilities
				state.pathScore = probabilities[winner]
				state.scoredNodes = len(state.compiled.Nodes)
			}
		}
		rounds++
		first = false
	}
	return rounds, nil
}

func (s *fieldState) greedyOptions() []int {
	for len(s.active) > 1 {
		depth := len(s.chosen)
		seen := map[int]struct{}{}
		var options []int
		for _, candidate := range s.active {
			path := s.compiled.Paths[candidate].Tokens
			if depth >= len(path) {
				return nil
			}
			token := path[depth]
			if _, ok := seen[token]; !ok {
				seen[token] = struct{}{}
				options = append(options, token)
			}
		}
		if len(options) > 1 {
			return options
		}
		s.chosen = append(s.chosen, options[0])
	}
	if len(s.active) == 1 {
		s.winner = s.active[0]
	}
	return nil
}

func (s *fieldState) selectGreedy(token int, probability float64) error {
	depth := len(s.chosen)
	remaining := s.active[:0]
	for _, candidate := range s.active {
		path := s.compiled.Paths[candidate].Tokens
		if depth < len(path) && path[depth] == token {
			remaining = append(remaining, candidate)
		}
	}
	if len(remaining) == 0 {
		return fmt.Errorf("selected token %d is not on an active path", token)
	}
	s.active = remaining
	s.chosen = append(s.chosen, token)
	s.pathScore *= probability
	s.scoredNodes++
	return nil
}

func constrainedProbability(scores []float32, selected int) float64 {
	mx := float64(scores[selected])
	z := 0.0
	for _, score := range scores {
		z += math.Exp(float64(score) - mx)
	}
	return 1 / z
}

func assembleResult(schema CompiledSchema, states []fieldState, contextTokens int) (Result, error) {
	out := Result{Decision: make(map[string]json.RawMessage, len(states)), Fields: make(map[string]FieldResult, len(states))}
	for i, state := range states {
		if state.winner < 0 || state.winner >= len(schema.Fields[i].Candidates) {
			return Result{}, fmt.Errorf("field %q has no selected value", schema.Fields[i].Name)
		}
		field := schema.Fields[i]
		candidate := field.Candidates[state.winner]
		value := append(json.RawMessage(nil), candidate.Value...)
		out.Decision[field.Name] = value
		out.Fields[field.Name] = FieldResult{Value: append(json.RawMessage(nil), value...), Probability: state.pathScore, ScoredNodes: state.scoredNodes, Tree: state.compiled.Tree}
		out.Usage.ScoredRows += state.compiled.Rows
	}
	out.Usage.ContextTokens = contextTokens
	return out, nil
}

func durationMS(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }
