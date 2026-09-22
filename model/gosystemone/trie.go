package gosystemone

import (
	"fmt"
	"math"
	"sort"
)

// Tokenizer is the minimal deterministic tokenizer boundary required by Go System One.
type Tokenizer interface {
	Encode(string) []int
}

// UserTextValidator keeps caller-controlled text from forging the prompt's
// configured control delimiters.
type UserTextValidator interface {
	ValidateUserText(string) error
}

type CandidatePath struct {
	Candidate Candidate
	Tokens    []int
}

type TrieNode struct {
	Prefix  []int
	Options []int
}

type CompiledField struct {
	Suffix []int
	Paths  []CandidatePath
	Nodes  []TrieNode
	Tree   bool
	Rows   int
}

// CompileField tokenizes each complete suffix/value/terminator string and
// splits at the longest shared token prefix. This preserves token-boundary
// behavior: the model scores exactly the token it would write after the suffix.
func CompileField(tok Tokenizer, input FieldInput, mode Mode, treeMax int) (CompiledField, error) {
	if tok == nil {
		return CompiledField{}, fmt.Errorf("nil tokenizer")
	}
	if len(input.Candidates) < 1 || len(input.Candidates) > MaxCandidates {
		return CompiledField{}, fmt.Errorf("field needs 1-%d candidates", MaxCandidates)
	}
	if mode == "" {
		mode = ModeAuto
	}
	if treeMax == 0 {
		treeMax = DefaultTreeMax
	}
	seqs := make([][]int, len(input.Candidates))
	for i, candidate := range input.Candidates {
		seqs[i] = tok.Encode(input.Suffix + candidate.Encoded + "\n")
		if len(seqs[i]) == 0 {
			return CompiledField{}, fmt.Errorf("candidate %s tokenized to an empty path", candidate.ID)
		}
	}
	common := len(seqs[0]) - 1
	for _, seq := range seqs[1:] {
		common = min(common, len(seq)-1)
		for i := 0; i < common; i++ {
			if seq[i] != seqs[0][i] {
				common = i
				break
			}
		}
	}
	out := CompiledField{Suffix: append([]int(nil), seqs[0][:common]...), Paths: make([]CandidatePath, len(seqs))}
	if len(out.Suffix) == 0 {
		return CompiledField{}, fmt.Errorf("field suffix tokenized to an empty path")
	}
	maxPath := 0
	for i, seq := range seqs {
		path := append([]int(nil), seq[common:]...)
		if len(path) == 0 {
			return CompiledField{}, fmt.Errorf("candidate %s has no distinct path", input.Candidates[i].ID)
		}
		out.Paths[i] = CandidatePath{Candidate: input.Candidates[i], Tokens: path}
		maxPath = max(maxPath, len(path))
		for j := 0; j < i; j++ {
			if tokenPrefixCollision(path, out.Paths[j].Tokens) {
				return CompiledField{}, fmt.Errorf("candidates %s and %s tokenize to colliding paths", input.Candidates[j].ID, input.Candidates[i].ID)
			}
		}
	}
	out.Nodes = buildTrieNodes(out.Paths)
	out.Tree = mode == ModeTree || mode == ModeAuto && len(out.Paths) <= treeMax
	if out.Tree {
		for _, node := range out.Nodes {
			out.Rows += len(out.Suffix) + len(node.Prefix)
		}
	} else {
		out.Rows = len(out.Suffix) + maxPath
	}
	return out, nil
}

func tokenPrefixCollision(a, b []int) bool {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func buildTrieNodes(paths []CandidatePath) []TrieNode {
	type work struct {
		active []int
		depth  int
	}
	root := make([]int, len(paths))
	for i := range root {
		root[i] = i
	}
	stack := []work{{active: root}}
	var nodes []TrieNode
	for len(stack) > 0 {
		last := len(stack) - 1
		item := stack[last]
		stack = stack[:last]
		if len(item.active) <= 1 {
			continue
		}
		seen := make(map[int]struct{})
		var options []int
		for _, candidate := range item.active {
			if item.depth >= len(paths[candidate].Tokens) {
				continue
			}
			token := paths[candidate].Tokens[item.depth]
			if _, ok := seen[token]; !ok {
				seen[token] = struct{}{}
				options = append(options, token)
			}
		}
		sort.Ints(options)
		if len(options) > 1 {
			nodes = append(nodes, TrieNode{
				Prefix:  append([]int(nil), paths[item.active[0]].Tokens[:item.depth]...),
				Options: append([]int(nil), options...),
			})
		}
		// Reverse push keeps the deterministic ascending token traversal order.
		for i := len(options) - 1; i >= 0; i-- {
			token := options[i]
			var sub []int
			for _, candidate := range item.active {
				path := paths[candidate].Tokens
				if item.depth < len(path) && path[item.depth] == token {
					sub = append(sub, candidate)
				}
			}
			stack = append(stack, work{active: sub, depth: item.depth + 1})
		}
	}
	return nodes
}

// FinishTree converts per-node candidate logits into the exact constrained
// distribution over full candidate paths.
func FinishTree(field CompiledField, nodeLogits [][]float32) (winner int, probabilities []float64, err error) {
	if len(nodeLogits) != len(field.Nodes) {
		return -1, nil, fmt.Errorf("node logits=%d, want %d", len(nodeLogits), len(field.Nodes))
	}
	nodeLogP := make([][]float64, len(field.Nodes))
	for i, scores := range nodeLogits {
		if len(scores) != len(field.Nodes[i].Options) {
			return -1, nil, fmt.Errorf("node %d logits=%d, want %d", i, len(scores), len(field.Nodes[i].Options))
		}
		mx := float64(scores[0])
		for _, score := range scores[1:] {
			mx = math.Max(mx, float64(score))
		}
		z := 0.0
		for _, score := range scores {
			z += math.Exp(float64(score) - mx)
		}
		lz := mx + math.Log(z)
		nodeLogP[i] = make([]float64, len(scores))
		for j, score := range scores {
			nodeLogP[i][j] = float64(score) - lz
		}
	}
	pathLogP := make([]float64, len(field.Paths))
	for i, candidate := range field.Paths {
		for n, node := range field.Nodes {
			if len(node.Prefix) >= len(candidate.Tokens) || !sameTokens(node.Prefix, candidate.Tokens[:len(node.Prefix)]) {
				continue
			}
			token := candidate.Tokens[len(node.Prefix)]
			option := -1
			for j, candidateToken := range node.Options {
				if candidateToken == token {
					option = j
					break
				}
			}
			if option < 0 {
				return -1, nil, fmt.Errorf("candidate %s token missing from trie node %d", candidate.Candidate.ID, n)
			}
			pathLogP[i] += nodeLogP[n][option]
		}
	}
	winner = 0
	for i := 1; i < len(pathLogP); i++ {
		if pathLogP[i] > pathLogP[winner] {
			winner = i
		}
	}
	z := 0.0
	for _, score := range pathLogP {
		z += math.Exp(score - pathLogP[winner])
	}
	probabilities = make([]float64, len(pathLogP))
	for i, score := range pathLogP {
		probabilities[i] = math.Exp(score-pathLogP[winner]) / z
	}
	return winner, probabilities, nil
}

func sameTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
