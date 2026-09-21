package gliner2

import (
	"math"
	"sort"
	"strings"
	"testing"
)

func TestSparseBoundaryProposerOracleEquivalence(t *testing.T) {
	boundaryStates := [][]float32{
		{0.2, -0.1, 0.3, 0.5},
		{-0.4, 0.7, 0.1, -0.2},
		{0.6, 0.2, -0.5, 0.4},
		{-0.3, 0.8, 0.9, -0.7},
		{1, 1, 1, 1},
	}
	queryStates := [][]float32{{0.5, -0.2, 0.1}, {-0.3, 0.4, 0.8}}
	boundaryMask := []bool{true, true, true, true, false}
	queryMask := []bool{true, true}
	startLogits := [][]float32{{0.1, 0.9, -0.2, 0.7, -5}, {0.5, -0.3, 0.4, 0.2, -5}}
	endLogits := [][]float32{{0.0, -0.1, 0.8, 0.6, -5}, {-0.2, 0.4, 0.3, 0.9, -5}}

	for _, rotary := range []bool{false, true} {
		t.Run(map[bool]string{false: "plain", true: "rotary"}[rotary], func(t *testing.T) {
			proposer := testSparseProposer(rotary)
			got, err := proposer.Forward(boundaryStates, queryStates, boundaryMask, queryMask, startLogits, endLogits)
			if err != nil {
				t.Fatal(err)
			}
			want := oracleSparseForward(proposer, boundaryStates, queryStates, boundaryMask, queryMask, startLogits, endLogits)
			compareProposalOutputs(t, got, want, 1e-5)
			compat, err := proposer.ScoreExplicitPairs(boundaryStates, queryStates, got.Indices, got.ValidMask)
			if err != nil {
				t.Fatal(err)
			}
			for q := range compat {
				for c := range compat[q] {
					if !nearlyEqual32(compat[q][c], got.CompatLogits[q][c], 1e-5) {
						t.Fatalf("rotary=%v compat[%d][%d]=%g want %g", rotary, q, c, compat[q][c], got.CompatLogits[q][c])
					}
				}
			}
		})
	}
}

func TestSparseBoundaryProposerTieAndMasking(t *testing.T) {
	proposer := SparseBoundaryProposer{
		BoundaryDim: 2,
		QueryDim:    2,
		Settings: ProposalSettings{
			StartTopK:             2,
			EndTopK:               2,
			EndsPerStart:          2,
			StartsPerEnd:          2,
			CandidateBudget:       4,
			EndBlockSize:          1,
			Bidirectional:         true,
			ExportMode:            "streaming",
			BoundaryTopKMax:       2,
			BoundaryTopKBucket:    1,
			EnableRotaryEndpoints: false,
		},
		StartPairProjection: Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}},
		EndKeyProjection:    Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}},
		StartQueryProjection: Linear{
			InDim:  2,
			OutDim: 2,
			Weight: []float32{0, 0, 0, 0},
			Bias:   []float32{20, 20},
		},
	}
	boundaryStates := [][]float32{{0, 0}, {0, 0}, {0, 0}, {0, 0}}
	queryStates := [][]float32{{1, 1}, {2, 2}}
	boundaryMask := []bool{true, true, true, false}
	queryMask := []bool{true, false}
	startLogits := [][]float32{{1, 1, 1, 1}, {99, 99, 99, 99}}
	endLogits := [][]float32{{1, 1, 1, 1}, {99, 99, 99, 99}}

	got, err := proposer.Forward(boundaryStates, queryStates, boundaryMask, queryMask, startLogits, endLogits)
	if err != nil {
		t.Fatal(err)
	}
	wantPairs := []struct {
		start, end int
	}{{0, 1}, {0, 2}, {1, 2}}
	for i, pair := range wantPairs {
		if !got.ValidMask[0][i] {
			t.Fatalf("query0 candidate %d unexpectedly invalid", i)
		}
		if got.Indices[0][i][0] != pair.start || got.Indices[0][i][1] != pair.end {
			t.Fatalf("query0 candidate %d=%v want [%d %d]", i, got.Indices[0][i], pair.start, pair.end)
		}
		if got.CompatLogits[0][i] != 0 || got.Logits[0][i] != 2 {
			t.Fatalf("query0 scores[%d]=compat %g full %g", i, got.CompatLogits[0][i], got.Logits[0][i])
		}
	}
	if got.ValidMask[0][3] || got.Logits[0][3] != MaskLogit || got.CompatLogits[0][3] != 0 {
		t.Fatalf("query0 padding mismatch %+v", got)
	}
	for i := range got.ValidMask[1] {
		if got.ValidMask[1][i] || got.Indices[1][i][0] != 0 || got.Indices[1][i][1] != 0 || got.Logits[1][i] != MaskLogit || got.CompatLogits[1][i] != 0 {
			t.Fatalf("masked query leaked at slot %d: %+v", i, got)
		}
	}
}

func TestProposalSettingsFromConfigRejectsSharedPool(t *testing.T) {
	_, err := ProposalSettingsFromConfig(BoundaryHeadConfig{
		CandidatePool:          "shared",
		StartTopK:              1,
		EndTopK:                1,
		EndsPerStart:           1,
		StartsPerEnd:           1,
		CandidateBudget:        1,
		EndBlockSize:           1,
		ExportMode:             "streaming",
		BoundaryTopKMax:        1,
		BoundaryTopKBucket:     1,
		VectorizedPairElements: 1,
	})
	if err == nil || !strings.Contains(err.Error(), "shared") {
		t.Fatalf("shared pool accepted: %v", err)
	}
}

func testSparseProposer(rotary bool) SparseBoundaryProposer {
	settings := ProposalSettings{
		StartTopK:               2,
		EndTopK:                 2,
		EndsPerStart:            2,
		StartsPerEnd:            2,
		CandidateBudget:         5,
		EndBlockSize:            2,
		Bidirectional:           true,
		ExportMode:              "streaming",
		EnableRotaryEndpoints:   rotary,
		RotaryBase:              10000,
		BoundaryTopKAlpha:       0.75,
		BoundaryTopKMax:         4,
		BoundaryTopKBucket:      2,
		VectorizedPairElements:  1,
		TrainingCandidateBudget: 5,
		MaxGoldPerQuery:         2,
	}
	proposer := SparseBoundaryProposer{
		BoundaryDim:         4,
		QueryDim:            3,
		Settings:            settings,
		StartPairProjection: Linear{InDim: 4, OutDim: 4, Weight: []float32{1, 0.1, 0, 0.2, -0.2, 0.9, 0.1, 0, 0.3, -0.1, 0.8, 0.2, 0, 0.4, -0.3, 1}, Bias: []float32{0.05, -0.02, 0.03, 0.01}},
		EndKeyProjection:    Linear{InDim: 4, OutDim: 4, Weight: []float32{0.7, -0.2, 0.3, 0.1, 0.4, 0.6, -0.1, 0, -0.3, 0.2, 0.9, -0.4, 0.2, 0.1, 0.5, 0.8}, Bias: []float32{-0.01, 0.04, -0.02, 0.02}},
	}
	if rotary {
		proposer.StartQueryProjection = Linear{InDim: 3, OutDim: 2, Weight: []float32{0.2, -0.1, 0.3, -0.4, 0.5, 0.1}, Bias: []float32{0.1, -0.2}}
	} else {
		proposer.StartQueryProjection = Linear{InDim: 3, OutDim: 4, Weight: []float32{0.2, -0.1, 0.3, -0.4, 0.5, 0.1, 0.6, 0.2, -0.2, 0.1, 0.3, 0.4}, Bias: []float32{0.1, -0.2, 0.05, 0.3}}
	}
	return proposer
}

func oracleSparseForward(p SparseBoundaryProposer, boundaryStates, queryStates [][]float32, boundaryMask, queryMask []bool, startLogits, endLogits [][]float32) BoundaryProposals {
	startAll := oracleLinear(p.StartPairProjection, boundaryStates)
	endAll := oracleLinear(p.EndKeyProjection, boundaryStates)
	if p.Settings.EnableRotaryEndpoints {
		positions := make([]int, len(boundaryStates))
		for i := range positions {
			positions[i] = i
		}
		startAll = oracleRotary(startAll, positions, p.Settings.rotaryBase())
		endAll = oracleRotary(endAll, positions, p.Settings.rotaryBase())
	}
	gateRaw := oracleLinear(p.StartQueryProjection, queryStates)
	gates := make([][]float32, len(gateRaw))
	for q := range gateRaw {
		if p.Settings.EnableRotaryEndpoints {
			gates[q] = make([]float32, p.BoundaryDim)
			for i, v := range gateRaw[q] {
				s := oracleSigmoid(v)
				gates[q][2*i] = s
				gates[q][2*i+1] = s
			}
			continue
		}
		gates[q] = make([]float32, len(gateRaw[q]))
		for i, v := range gateRaw[q] {
			gates[q][i] = oracleSigmoid(v)
		}
	}
	startK := oracleResolveBudget(len(boundaryStates), p.Settings.StartTopK, p.Settings.BoundaryTopKAlpha, p.Settings.boundaryTopKMaxValue(), p.Settings.boundaryTopKBucketValue())
	endK := oracleResolveBudget(len(boundaryStates), p.Settings.EndTopK, p.Settings.BoundaryTopKAlpha, p.Settings.boundaryTopKMaxValue(), p.Settings.boundaryTopKBucketValue())
	scale := float32(1 / math.Sqrt(float64(p.BoundaryDim)))
	out := newProposalOutput(len(queryStates), p.Settings.CandidateBudget)
	for q := range queryStates {
		validBoundaries := make([]bool, len(boundaryStates))
		for i := range boundaryStates {
			validBoundaries[i] = boundaryMask[i] && queryMask[q]
		}
		stScores, stIdx, stValid := oracleTopBoundaries(startLogits[q], validBoundaries, startK)
		all := make([]oracleCandidate, 0, len(stIdx)*p.Settings.EndsPerStart+len(stIdx)*p.Settings.StartsPerEnd)
		for slot := range stIdx {
			if !stValid[slot] {
				continue
			}
			ends := make([]oracleCandidate, 0, len(boundaryStates))
			for end := range boundaryStates {
				if !queryMask[q] || !boundaryMask[end] || end <= stIdx[slot] {
					continue
				}
				score := oracleDot(oracleGated(startAll[stIdx[slot]], gates[q]), endAll[end])*scale + stScores[slot] + endLogits[q][end]
				ends = append(ends, oracleCandidate{start: stIdx[slot], end: end, score: score})
			}
			sort.SliceStable(ends, func(i, j int) bool { return ends[i].score > ends[j].score })
			all = append(all, ends[:min(p.Settings.EndsPerStart, len(ends))]...)
		}
		if p.Settings.Bidirectional {
			enScores, enIdx, enValid := oracleTopBoundaries(endLogits[q], validBoundaries, endK)
			for slot := range enIdx {
				if !enValid[slot] {
					continue
				}
				starts := make([]oracleCandidate, 0, len(boundaryStates))
				for start := range boundaryStates {
					if !queryMask[q] || !boundaryMask[start] || start >= enIdx[slot] {
						continue
					}
					score := oracleDot(oracleGated(endAll[enIdx[slot]], gates[q]), startAll[start])*scale + enScores[slot] + startLogits[q][start]
					starts = append(starts, oracleCandidate{start: start, end: enIdx[slot], score: score})
				}
				sort.SliceStable(starts, func(i, j int) bool { return starts[i].score > starts[j].score })
				all = append(all, starts[:min(p.Settings.StartsPerEnd, len(starts))]...)
			}
		}
		unique := make(map[int]float32)
		for _, candidate := range all {
			key := candidate.start*len(boundaryStates) + candidate.end
			if score, ok := unique[key]; !ok || candidate.score > score {
				unique[key] = candidate.score
			}
		}
		pairs := make([]oracleCandidate, 0, len(unique))
		for key, score := range unique {
			start := key / len(boundaryStates)
			pairs = append(pairs, oracleCandidate{start: start, end: key - start*len(boundaryStates), score: score})
		}
		sort.Slice(pairs, func(i, j int) bool {
			if pairs[i].score != pairs[j].score {
				return pairs[i].score > pairs[j].score
			}
			if pairs[i].start != pairs[j].start {
				return pairs[i].start < pairs[j].start
			}
			return pairs[i].end < pairs[j].end
		})
		for c := 0; c < min(p.Settings.CandidateBudget, len(pairs)); c++ {
			out.ValidMask[q][c] = true
			out.Indices[q][c][0] = pairs[c].start
			out.Indices[q][c][1] = pairs[c].end
			compat := oracleDot(oracleGated(startAll[pairs[c].start], gates[q]), endAll[pairs[c].end]) * scale
			out.CompatLogits[q][c] = compat
			out.Logits[q][c] = compat + startLogits[q][pairs[c].start] + endLogits[q][pairs[c].end]
		}
	}
	return out
}

type oracleCandidate struct {
	start, end int
	score      float32
}

func oracleTopBoundaries(logits []float32, valid []bool, k int) ([]float32, []int, []bool) {
	order := make([]int, len(logits))
	for i := range order {
		order[i] = i
	}
	masked := func(i int) float32 {
		if !valid[i] {
			return MaskLogit
		}
		return logits[i]
	}
	sort.SliceStable(order, func(i, j int) bool { return masked(order[i]) > masked(order[j]) })
	k = min(k, len(logits))
	scores := make([]float32, k)
	indices := make([]int, k)
	mask := make([]bool, k)
	for i := 0; i < k; i++ {
		idx := order[i]
		mask[i] = valid[idx]
		if mask[i] {
			scores[i] = logits[idx]
			indices[i] = idx
		}
	}
	return scores, indices, mask
}

func oracleResolveBudget(nBoundaries, baseK int, alpha float64, kMax, bucket int) int {
	if alpha <= 0 {
		return baseK
	}
	requested := int(math.Ceil(alpha * float64(max(nBoundaries-1, 0))))
	if requested < baseK {
		requested = baseK
	}
	if requested > kMax {
		requested = kMax
	}
	return min(kMax, int(math.Ceil(float64(requested)/float64(bucket)))*bucket)
}

func oracleLinear(l Linear, rows [][]float32) [][]float32 {
	out := make([][]float32, len(rows))
	for r := range rows {
		out[r] = make([]float32, l.OutDim)
		for o := 0; o < l.OutDim; o++ {
			base := o * l.InDim
			var sum float32
			for i := 0; i < l.InDim; i++ {
				sum += l.Weight[base+i] * rows[r][i]
			}
			if len(l.Bias) != 0 {
				sum += l.Bias[o]
			}
			out[r][o] = sum
		}
	}
	return out
}

func oracleRotary(states [][]float32, positions []int, base float32) [][]float32 {
	out := make([][]float32, len(states))
	dim := len(states[0])
	for r := range states {
		out[r] = make([]float32, dim)
		for i := 0; i < dim/2; i++ {
			freq := float32(1 / math.Pow(float64(base), float64(float32(2*i)/float32(dim))))
			angle := float32(positions[r]) * freq
			c := float32(math.Cos(float64(angle)))
			s := float32(math.Sin(float64(angle)))
			even, odd := states[r][2*i], states[r][2*i+1]
			out[r][2*i] = even*c - odd*s
			out[r][2*i+1] = even*s + odd*c
		}
	}
	return out
}

func oracleSigmoid(x float32) float32 {
	if x >= 0 {
		return 1 / (1 + float32(math.Exp(-float64(x))))
	}
	e := float32(math.Exp(float64(x)))
	return e / (1 + e)
}

func oracleGated(row, gate []float32) []float32 {
	out := make([]float32, len(row))
	for i := range row {
		out[i] = row[i] * gate[i]
	}
	return out
}

func oracleDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func compareProposalOutputs(t *testing.T, got, want BoundaryProposals, tol float32) {
	t.Helper()
	if len(got.Indices) != len(want.Indices) {
		t.Fatalf("query count=%d want=%d", len(got.Indices), len(want.Indices))
	}
	for q := range want.Indices {
		if len(got.Indices[q]) != len(want.Indices[q]) {
			t.Fatalf("query %d candidate count=%d want=%d", q, len(got.Indices[q]), len(want.Indices[q]))
		}
		for c := range want.Indices[q] {
			if got.ValidMask[q][c] != want.ValidMask[q][c] {
				t.Fatalf("valid[%d][%d]=%v want %v", q, c, got.ValidMask[q][c], want.ValidMask[q][c])
			}
			if got.Indices[q][c][0] != want.Indices[q][c][0] || got.Indices[q][c][1] != want.Indices[q][c][1] {
				t.Fatalf("indices[%d][%d]=%v want %v", q, c, got.Indices[q][c], want.Indices[q][c])
			}
			if !nearlyEqual32(got.CompatLogits[q][c], want.CompatLogits[q][c], tol) {
				t.Fatalf("compat[%d][%d]=%g want %g", q, c, got.CompatLogits[q][c], want.CompatLogits[q][c])
			}
			if !nearlyEqual32(got.Logits[q][c], want.Logits[q][c], tol) {
				t.Fatalf("logits[%d][%d]=%g want %g", q, c, got.Logits[q][c], want.Logits[q][c])
			}
		}
	}
}

func nearlyEqual32(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}
