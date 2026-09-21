package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"testing"
)

func TestDocumentCandidatePoolOracleEquivalence(t *testing.T) {
	pool := testDocumentCandidatePool()
	boundaryStates := [][]float32{{0.2, -0.1}, {0.7, 0.4}, {-0.3, 0.9}, {0.5, -0.6}, {0, 0}}
	boundaryMask := []bool{true, true, true, true, false}
	queryMask := []bool{true, true, false}
	startLogits := [][]float32{
		{0.1, 1.2, 0.6, 0.3, -9},
		{0.9, 0.2, 1.1, 0.8, -9},
		{50, 50, 50, 50, 50},
	}
	endLogits := [][]float32{
		{0.0, 0.7, 1.0, 0.4, -9},
		{0.2, 0.8, 0.5, 1.3, -9},
		{50, 50, 50, 50, 50},
	}

	got, err := pool.Forward(boundaryStates, boundaryMask, queryMask, startLogits, endLogits)
	if err != nil {
		t.Fatal(err)
	}
	want := oracleDocumentPoolForward(pool, boundaryStates, boundaryMask, queryMask, startLogits, endLogits)
	comparePooledCandidates(t, got, want, 1e-5)
	for i := range got.ValidMask {
		if got.ValidMask[i] && got.Indices[i][1] <= got.Indices[i][0] {
			t.Fatalf("invalid half-open span at slot %d: %v", i, got.Indices[i])
		}
	}
}

func TestDocumentCandidatePoolMinPoolPerQueryAndMaskedQueries(t *testing.T) {
	pool := DocumentCandidatePool{
		BoundaryDim:      2,
		PoolBoundaryTopK: 3,
		PoolSize:         2,
		MinPoolPerQuery:  1,
		StartProjection:  Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}},
		EndProjection:    Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}},
	}
	boundaryStates := [][]float32{{0, 0}, {0, 0}, {0, 0}, {0, 0}}
	boundaryMask := []bool{true, true, true, true}
	queryMask := []bool{true, true, false}
	startLogits := [][]float32{
		{10, 9, 0, 0},
		{0, 0, 8, 0},
		{100, 100, 100, 100},
	}
	endLogits := [][]float32{
		{0, 0, 0, 10},
		{0, 0, 0, 8},
		{100, 100, 100, 100},
	}

	got, err := pool.Forward(boundaryStates, boundaryMask, queryMask, startLogits, endLogits)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		start, end int
		score      float32
	}{{0, 3, 20}, {2, 3, 18}}
	for i, candidate := range want {
		if !got.ValidMask[i] {
			t.Fatalf("slot %d unexpectedly invalid: %+v", i, got)
		}
		if got.Indices[i][0] != candidate.start || got.Indices[i][1] != candidate.end {
			t.Fatalf("slot %d indices=%v want [%d %d]", i, got.Indices[i], candidate.start, candidate.end)
		}
		if got.CompatLogits[i] != 0 {
			t.Fatalf("slot %d compat=%g want 0", i, got.CompatLogits[i])
		}
		if !nearlyEqual32(got.ProposalLogits[i], candidate.score, 1e-6) {
			t.Fatalf("slot %d proposal=%g want %g", i, got.ProposalLogits[i], candidate.score)
		}
	}
	if got.Indices[1][0] == 1 && got.Indices[1][1] == 3 {
		t.Fatalf("global fill displaced per-query floor: %+v", got)
	}
}

func TestDeduplicatePoolKeysStablePriorityAndKeyTies(t *testing.T) {
	keys := []int{7, 2, 7, 1, 99}
	scores := []float32{0.5, 0.5, 1.5, 0.5, 9.0}
	valid := []bool{true, true, true, true, false}
	gotKeys, gotValid, err := deduplicatePoolKeys(keys, scores, valid, 4, 10)
	if err != nil {
		t.Fatal(err)
	}
	wantKeys := []int{7, 1, 2}
	wantValid := []bool{true, true, true}
	for i := range wantKeys {
		if gotKeys[i] != wantKeys[i] || gotValid[i] != wantValid[i] {
			t.Fatalf("slot %d = (%d,%v) want (%d,%v)", i, gotKeys[i], gotValid[i], wantKeys[i], wantValid[i])
		}
	}
	if gotValid[3] {
		t.Fatalf("padding slot unexpectedly valid: keys=%v valid=%v", gotKeys, gotValid)
	}
}

func TestDocumentCandidatePoolAllQueriesMasked(t *testing.T) {
	pool := testDocumentCandidatePool()
	boundaryStates := [][]float32{{1, 0}, {0, 1}, {1, 1}}
	boundaryMask := []bool{true, true, true}
	queryMask := []bool{false, false}
	startLogits := [][]float32{{99, 99, 99}, {88, 88, 88}}
	endLogits := [][]float32{{77, 77, 77}, {66, 66, 66}}

	got, err := pool.Forward(boundaryStates, boundaryMask, queryMask, startLogits, endLogits)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.ValidMask {
		if got.ValidMask[i] || got.Indices[i][0] != 0 || got.Indices[i][1] != 0 || got.ProposalLogits[i] != MaskLogit || got.CompatLogits[i] != 0 {
			t.Fatalf("masked queries leaked at slot %d: %+v", i, got)
		}
	}
}

func TestLoadDocumentCandidatePoolPublishedBindings(t *testing.T) {
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/shared_pool_tensors.json")
	if err != nil {
		t.Fatal(err)
	}
	var src map[string]struct {
		Shape []int `json:"shape"`
	}
	if err := json.Unmarshal(raw, &src); err != nil {
		t.Fatal(err)
	}
	shapes := make(shapeSource)
	for name, tensor := range src {
		shapes[name] = tensor.Shape
	}
	pool, err := LoadDocumentCandidatePool(shapes, c.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	if pool.BoundaryDim != c.BoundaryHead.BoundaryDim || pool.PoolSize != c.BoundaryHead.PoolSize || pool.StartProjection.InDim != c.BoundaryHead.BoundaryDim {
		t.Fatalf("binding mismatch: %+v", pool)
	}
	delete(shapes, "boundary_head.shared_pool_builder.start_projection.weight")
	if _, err := LoadDocumentCandidatePool(shapes, c.BoundaryHead); err == nil {
		t.Fatal("missing shared pool tensor accepted")
	}
	shapes["boundary_head.shared_pool_builder.start_projection.weight"] = []int{1}
	if _, err := LoadDocumentCandidatePool(shapes, c.BoundaryHead); err == nil {
		t.Fatal("wrong shared pool tensor shape accepted")
	}
}

func testDocumentCandidatePool() DocumentCandidatePool {
	return DocumentCandidatePool{
		BoundaryDim:      2,
		PoolBoundaryTopK: 3,
		PoolSize:         5,
		MinPoolPerQuery:  2,
		StartProjection:  Linear{InDim: 2, OutDim: 2, Weight: []float32{1.0, 0.2, -0.3, 0.8}, Bias: []float32{0.1, -0.2}},
		EndProjection:    Linear{InDim: 2, OutDim: 2, Weight: []float32{0.7, -0.1, 0.4, 0.9}, Bias: []float32{-0.05, 0.15}},
	}
}

func oracleDocumentPoolForward(p DocumentCandidatePool, boundaryStates [][]float32, boundaryMask, queryMask []bool, startLogits, endLogits [][]float32) PooledCandidates {
	n := len(boundaryStates)
	startAll := oracleLinear(p.StartProjection, boundaryStates)
	endAll := oracleLinear(p.EndProjection, boundaryStates)
	active := false
	for _, ok := range queryMask {
		if ok {
			active = true
			break
		}
	}
	unionStart := make([]float32, n)
	unionEnd := make([]float32, n)
	unionValid := make([]bool, n)
	for i := 0; i < n; i++ {
		unionStart[i] = MaskLogit
		unionEnd[i] = MaskLogit
		unionValid[i] = boundaryMask[i] && active
		if !boundaryMask[i] {
			continue
		}
		for q := range queryMask {
			if !queryMask[q] {
				continue
			}
			if startLogits[q][i] > unionStart[i] {
				unionStart[i] = startLogits[q][i]
			}
			if endLogits[q][i] > unionEnd[i] {
				unionEnd[i] = endLogits[q][i]
			}
		}
	}
	_, starts, startsValid := oracleTopBoundaries(unionStart, unionValid, p.PoolBoundaryTopK)
	_, ends, endsValid := oracleTopBoundaries(unionEnd, unionValid, p.PoolBoundaryTopK)
	type oraclePoolPair struct {
		start  int
		end    int
		valid  bool
		compat float32
		score  float32
	}
	scale := float32(1 / math.Sqrt(float64(p.BoundaryDim)))
	pairs := make([]oraclePoolPair, 0, len(starts)*len(ends))
	for i := range starts {
		for j := range ends {
			start, end := starts[i], ends[j]
			compat := oracleDot(startAll[start], endAll[end]) * scale
			pairs = append(pairs, oraclePoolPair{
				start:  start,
				end:    end,
				valid:  startsValid[i] && endsValid[j] && end > start,
				compat: compat,
				score:  compat + unionStart[start] + unionEnd[end],
			})
		}
	}
	quota := min(p.MinPoolPerQuery, len(pairs))
	allKeys := make([]int, 0, len(queryMask)*quota+len(pairs))
	allScores := make([]float32, 0, cap(allKeys))
	allValid := make([]bool, 0, cap(allKeys))
	if quota > 0 {
		priorityBase := -MaskLogit * 0.5
		for q := range queryMask {
			order := make([]int, len(pairs))
			masked := make([]float32, len(pairs))
			valids := make([]bool, len(pairs))
			for i := range pairs {
				order[i] = i
				valids[i] = pairs[i].valid && queryMask[q]
				if valids[i] {
					masked[i] = pairs[i].compat + startLogits[q][pairs[i].start] + endLogits[q][pairs[i].end]
				} else {
					masked[i] = MaskLogit
				}
			}
			sort.SliceStable(order, func(i, j int) bool { return masked[order[i]] > masked[order[j]] })
			for slot := 0; slot < quota; slot++ {
				pick := pairs[order[slot]]
				allKeys = append(allKeys, pick.start*n+pick.end)
				allScores = append(allScores, priorityBase+float32(quota-slot))
				allValid = append(allValid, valids[order[slot]])
			}
		}
	}
	for _, pair := range pairs {
		allKeys = append(allKeys, pair.start*n+pair.end)
		allScores = append(allScores, pair.score)
		allValid = append(allValid, pair.valid)
	}
	selectedKeys, selectedValid := oracleDeduplicatePoolKeys(allKeys, allScores, allValid, p.PoolSize, n)
	out := newPooledCandidates(p.PoolSize)
	for i := 0; i < p.PoolSize; i++ {
		out.ValidMask[i] = selectedValid[i]
		if !selectedValid[i] {
			continue
		}
		start := selectedKeys[i] / n
		end := selectedKeys[i] - start*n
		out.Indices[i][0] = start
		out.Indices[i][1] = end
		compat := oracleDot(startAll[start], endAll[end]) * scale
		out.CompatLogits[i] = compat
		out.ProposalLogits[i] = compat + unionStart[start] + unionEnd[end]
	}
	return out
}

func oracleDeduplicatePoolKeys(keys []int, scores []float32, valid []bool, capacity, nBoundaries int) ([]int, []bool) {
	type item struct {
		key   int
		score float32
		valid bool
	}
	invalidKey := nBoundaries * nBoundaries
	items := make([]item, len(keys))
	for i := range keys {
		items[i] = item{key: invalidKey, score: MaskLogit, valid: valid[i]}
		if valid[i] {
			items[i].key = keys[i]
			items[i].score = scores[i]
		}
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].score > items[j].score })
	sort.SliceStable(items, func(i, j int) bool { return items[i].key < items[j].key })
	keep := make([]bool, len(items))
	for i := range items {
		if items[i].valid && (i == 0 || items[i].key != items[i-1].key) {
			keep[i] = true
		}
	}
	order := make([]int, len(items))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		si, sj := MaskLogit, MaskLogit
		if keep[order[i]] {
			si = items[order[i]].score
		}
		if keep[order[j]] {
			sj = items[order[j]].score
		}
		return si > sj
	})
	selectedKeys := make([]int, capacity)
	selectedValid := make([]bool, capacity)
	for i := 0; i < min(capacity, len(order)); i++ {
		selectedKeys[i] = items[order[i]].key
		selectedValid[i] = keep[order[i]]
	}
	return selectedKeys, selectedValid
}

func comparePooledCandidates(t *testing.T, got, want PooledCandidates, tol float32) {
	t.Helper()
	if len(got.Indices) != len(want.Indices) {
		t.Fatalf("candidate count=%d want=%d", len(got.Indices), len(want.Indices))
	}
	for i := range want.Indices {
		if got.ValidMask[i] != want.ValidMask[i] {
			t.Fatalf("valid[%d]=%v want %v", i, got.ValidMask[i], want.ValidMask[i])
		}
		if got.Indices[i][0] != want.Indices[i][0] || got.Indices[i][1] != want.Indices[i][1] {
			t.Fatalf("indices[%d]=%v want %v", i, got.Indices[i], want.Indices[i])
		}
		if !nearlyEqual32(got.CompatLogits[i], want.CompatLogits[i], tol) {
			t.Fatalf("compat[%d]=%g want %g", i, got.CompatLogits[i], want.CompatLogits[i])
		}
		if !nearlyEqual32(got.ProposalLogits[i], want.ProposalLogits[i], tol) {
			t.Fatalf("proposal[%d]=%g want %g", i, got.ProposalLogits[i], want.ProposalLogits[i])
		}
	}
}
