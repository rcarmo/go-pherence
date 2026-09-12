package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"sync"
	"testing"
)

type centroidOracle struct {
	Name                            string
	Config                          VBxCentroidConfig
	Embeddings, Q, Priors, Expected []float64
	SpeakerIndices                  []int     `json:"speaker_indices"`
	WeightSums                      []float64 `json:"weight_sums"`
}
type matchingOracle struct {
	Name   string
	Config AssignmentConfig
	Scores []*float64
	Labels []int
}
type cosineOracle struct {
	Name                          string
	Config                        CosineAssignmentConfig
	Embeddings, Centroids, Scores []float64
	Segmentations                 []*float32
	Labels                        []int
}

func loadAssignmentOracles(t *testing.T) ([]centroidOracle, []matchingOracle, []cosineOracle) {
	t.Helper()
	data, err := os.ReadFile("testdata/assignment-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema    int
		Abs       float64 `json:"absolute_tolerance"`
		Rel       float64 `json:"relative_tolerance"`
		Reference struct {
			Clustering string `json:"clustering_sha256"`
			Solver     string `json:"solver_sha256"`
		}
		Centroids   []centroidOracle
		Matching    []matchingOracle
		Assignments []cosineOracle
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-10 || f.Rel != 2e-10 || f.Reference.Clustering != "6031fb7c21277a7e9901ef2cdaed7d5cd69f7ef45508dc4b45e82ce0da3c8fba" || f.Reference.Solver != "73d78155990732311a116bc04f7f64d874c0f3793172e9aab12a476cb3d15c2c" || len(f.Centroids) != 13 || len(f.Matching) != 60 || len(f.Assignments) != 21 {
		t.Fatal("assignment fixture contract")
	}
	return f.Centroids, f.Matching, f.Assignments
}
func matchingValues(values []*float64) []float64 {
	out := make([]float64, len(values))
	for i, v := range values {
		if v == nil {
			out[i] = math.NaN()
		} else {
			out[i] = *v
		}
	}
	return out
}
func sameMatchingBits(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if math.Float64bits(v) != math.Float64bits(b[i]) {
			return false
		}
	}
	return true
}
func TestVBxCentroidsPinnedOracle(t *testing.T) {
	cases, _, _ := loadAssignmentOracles(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			before := append([]float64(nil), c.Embeddings...)
			out, err := ComputeVBxCentroids(context.Background(), c.Embeddings, c.Q, c.Priors, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			closeClustering64(t, out.Centroids, c.Expected)
			closeClustering64(t, out.WeightSums, c.WeightSums)
			if !reflect.DeepEqual(out.SpeakerIndices, c.SpeakerIndices) || !reflect.DeepEqual(before, c.Embeddings) {
				t.Fatal("centroid order/source mutation")
			}
			for i := range c.Embeddings {
				c.Embeddings[i] = 99
			}
			closeClustering64(t, out.Centroids, c.Expected)
		})
	}
}
func TestConstrainedAssignmentPinnedOracle(t *testing.T) {
	_, cases, _ := loadAssignmentOracles(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			scores := matchingValues(c.Scores)
			before := append([]float64(nil), scores...)
			labels, err := ConstrainedSpeakerAssignment(context.Background(), scores, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(labels, c.Labels) || !sameMatchingBits(before, scores) {
				t.Fatal("matching differs", labels, c.Labels)
			}
			// Independent exhaustive optimum check for smaller rectangular cases.
			if c.Config.Speakers <= 5 && c.Config.Clusters <= 5 {
				minValue := math.Inf(1)
				for _, v := range scores {
					if !math.IsNaN(v) {
						minValue = math.Min(minValue, v)
					}
				}
				for i, v := range scores {
					if math.IsNaN(v) {
						scores[i] = minValue
					}
				}
				for chunk := 0; chunk < c.Config.Chunks; chunk++ {
					n, k := c.Config.Speakers, c.Config.Clusters
					matrix := scores[chunk*n*k : (chunk+1)*n*k]
					want := bruteAssignment(matrix, n, k)
					got := float64(0)
					assigned := 0
					used := make(map[int]bool)
					for row, label := range labels[chunk*n : (chunk+1)*n] {
						if label == -2 {
							continue
						}
						if label < 0 || label >= k || used[label] {
							t.Fatal("non-bijective assignment")
						}
						used[label] = true
						assigned++
						got += matrix[row*k+label]
					}
					if assigned != min(n, k) || math.Abs(got-want) > 1e-10 {
						t.Fatal("not global optimum", got, want, assigned)
					}
				}
			}
		})
	}
}
func bruteAssignment(cost []float64, rows, columns int) float64 {
	best := math.Inf(-1)
	var search func(row, used, assigned int, sum float64)
	search = func(row, used, assigned int, sum float64) {
		if row == rows {
			if assigned == min(rows, columns) {
				best = math.Max(best, sum)
			}
			return
		}
		if rows > columns {
			search(row+1, used, assigned, sum)
		}
		for c := 0; c < columns; c++ {
			if used&(1<<c) == 0 {
				search(row+1, used|(1<<c), assigned+1, sum+cost[row*columns+c])
			}
		}
	}
	search(0, 0, 0, 0)
	return best
}
func TestCosineAssignmentPinnedOracle(t *testing.T) {
	_, _, cases := loadAssignmentOracles(t)
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			embeddings := append([]float64(nil), c.Embeddings...)
			centroids := append([]float64(nil), c.Centroids...)
			seg := maskValues(c.Segmentations)
			segBefore := append([]float32(nil), seg...)
			out, err := AssignCosineSpeakers(context.Background(), embeddings, centroids, seg, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			closeClustering64(t, out.Scores, c.Scores)
			if !reflect.DeepEqual(out.Labels, c.Labels) || !reflect.DeepEqual(embeddings, c.Embeddings) || !reflect.DeepEqual(centroids, c.Centroids) || !sameMaskBits(seg, segBefore) {
				t.Fatal("cosine labels/source mutation", out.Labels, c.Labels)
			}
			for i := range embeddings {
				embeddings[i] = 0
			}
			for i := range centroids {
				centroids[i] = 0
			}
			closeClustering64(t, out.Scores, c.Scores)
		})
	}
}
func TestPLDAVBxCentroidAssignmentBridge(t *testing.T) {
	ps, vs := loadPLDAVBxOracles(t)
	cs, _, as := loadAssignmentOracles(t)
	p, v, c, a := ps[1], vs[len(vs)-1], cs[len(cs)-1], as[len(as)-1]
	model, err := NewPreparedPLDA(context.Background(), p.Config, p.Weights)
	if err != nil {
		t.Fatal(err)
	}
	x, err := model.Transform(context.Background(), p.Input, p.Rows)
	if err != nil {
		t.Fatal(err)
	}
	vb, err := ClusterVBx(context.Background(), x, model.Phi(), v.Labels, v.Config)
	if err != nil {
		t.Fatal(err)
	}
	centroids, err := ComputeVBxCentroids(context.Background(), p.Input, vb.Responsibilities, vb.Priors, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	closeClustering64(t, centroids.Centroids, c.Expected)
	out, err := AssignCosineSpeakers(context.Background(), p.Input, centroids.Centroids, maskValues(a.Segmentations), a.Config)
	if err != nil {
		t.Fatal(err)
	}
	closeClustering64(t, out.Scores, a.Scores)
	if !reflect.DeepEqual(out.Labels, a.Labels) {
		t.Fatal("bridge assignment", out.Labels, a.Labels)
	}
}
func TestCentroidAssignmentValidation(t *testing.T) {
	// Separate exact threshold values: equality pruned; nextafter above retained.
	priors := []float64{1e-7, math.Nextafter(1e-7, 1), 1 - 1e-7 - math.Nextafter(1e-7, 1)}
	out, err := ComputeVBxCentroids(context.Background(), []float64{2}, []float64{.2, .3, .5}, priors, VBxCentroidConfig{1, 1, 3})
	if err != nil || !reflect.DeepEqual(out.SpeakerIndices, []int{1, 2}) {
		t.Fatal("pruning threshold", out, err)
	}
	for _, kind := range []string{"rows", "dim", "speakers", "emb_len", "q_len", "p_len", "nan", "q_negative", "q_sum", "p_inf", "p_sum", "zero_mass", "overflow"} {
		cfg := VBxCentroidConfig{2, 1, 2}
		e, q, p := []float64{1, 2}, []float64{.3, .7, .2, .8}, []float64{.4, .6}
		switch kind {
		case "rows":
			cfg.Rows = int(^uint(0) >> 1)
		case "dim":
			cfg.Dimension = 0
		case "speakers":
			cfg.Speakers = 65
		case "emb_len":
			e = e[:1]
		case "q_len":
			q = q[:3]
		case "p_len":
			p = p[:1]
		case "nan":
			e[0] = math.NaN()
		case "q_negative":
			q[0] = -.3
		case "q_sum":
			q[0] = .4
		case "p_inf":
			p[0] = math.Inf(1)
		case "p_sum":
			p[0] = .3
		case "zero_mass":
			q = []float64{0, 1, 0, 1}
		case "overflow":
			e = []float64{math.MaxFloat64, math.MaxFloat64}
			q = []float64{1, 0, 1, 0}
			p = []float64{1, 0}
		}
		if out, err := ComputeVBxCentroids(context.Background(), e, q, p, cfg); err == nil || out != nil {
			t.Fatal("accepted bad centroids", kind)
		}
	}
	for _, kind := range []string{"chunks", "rows", "cols", "short", "nan", "inf", "huge"} {
		cfg := AssignmentConfig{1, 2, 2}
		scores := []float64{1, 2, 3, 4}
		switch kind {
		case "chunks":
			cfg.Chunks = 0
		case "rows":
			cfg.Speakers = 9
		case "cols":
			cfg.Clusters = int(^uint(0) >> 1)
		case "short":
			scores = scores[:3]
		case "nan":
			for i := range scores {
				scores[i] = math.NaN()
			}
		case "inf":
			scores[0] = math.Inf(-1)
		case "huge":
			scores[0] = 1e101
		}
		if out, err := ConstrainedSpeakerAssignment(context.Background(), scores, cfg); err == nil || out != nil {
			t.Fatal("accepted bad matching", kind)
		}
	}
	for _, kind := range []string{"dimension", "frames", "short_e", "short_c", "short_s", "soft", "inf_s", "nan_e", "inf_c", "zero_e", "zero_c", "overflow", "underflow"} {
		cfg := CosineAssignmentConfig{1, 2, 2, 2, 1, true}
		e, c, s := []float64{1, 0, 0, 1}, []float64{1, 1, 1, -1}, []float32{1, 0}
		switch kind {
		case "dimension":
			cfg.Dimension = 513
		case "frames":
			cfg.Frames = 0
		case "short_e":
			e = e[:3]
		case "short_c":
			c = c[:3]
		case "short_s":
			s = s[:1]
		case "soft":
			s[0] = .5
		case "inf_s":
			s[0] = float32(math.Inf(1))
		case "nan_e":
			e[0] = math.NaN()
		case "inf_c":
			c[0] = math.Inf(1)
		case "zero_e":
			e[0] = 0
		case "zero_c":
			c[0] = 0
			c[1] = 0
		case "overflow":
			e[0] = math.MaxFloat64
		case "underflow":
			e[0] = math.SmallestNonzeroFloat64
		}
		if out, err := AssignCosineSpeakers(context.Background(), e, c, s, cfg); err == nil || out != nil {
			t.Fatal("accepted bad cosine", kind)
		}
	}
}
func TestCentroidAssignmentCancellationConcurrency(t *testing.T) {
	cs, ms, as := loadAssignmentOracles(t)
	c, m, a := cs[7], ms[47], as[15]
	scores := matchingValues(m.Scores)
	seg := maskValues(a.Segmentations)
	runs := []func(context.Context) error{
		func(ctx context.Context) error {
			out, err := ComputeVBxCentroids(ctx, c.Embeddings, c.Q, c.Priors, c.Config)
			if err != nil && out != nil {
				t.Fatal("partial centroids")
			}
			return err
		},
		func(ctx context.Context) error {
			out, err := ConstrainedSpeakerAssignment(ctx, scores, m.Config)
			if err != nil && out != nil {
				t.Fatal("partial matching")
			}
			return err
		},
		func(ctx context.Context) error {
			out, err := AssignCosineSpeakers(ctx, a.Embeddings, a.Centroids, seg, a.Config)
			if err != nil && out != nil {
				t.Fatal("partial cosine")
			}
			return err
		},
	}
	for stage, run := range runs {
		counter := newPowersetContext(0)
		err := run(counter)
		counter.cancel()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("centroid/assignment stage%d checkpoints%d", stage, counter.calls)
		for at := 1; at <= counter.calls; at++ {
			ctx := newPowersetContext(at)
			err := run(ctx)
			ctx.cancel()
			if !errors.Is(err, context.Canceled) {
				t.Fatal("missed cancellation", stage, at, err)
			}
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := ComputeVBxCentroids(context.Background(), c.Embeddings, c.Q, c.Priors, c.Config)
			if err != nil {
				t.Error(err)
				return
			}
			closeClustering64(t, out.Centroids, c.Expected)
			labels, err := ConstrainedSpeakerAssignment(context.Background(), scores, m.Config)
			if err != nil || !reflect.DeepEqual(labels, m.Labels) {
				t.Error("concurrent matching", err)
			}
			result, err := AssignCosineSpeakers(context.Background(), a.Embeddings, a.Centroids, seg, a.Config)
			if err != nil {
				t.Error(err)
				return
			}
			closeClustering64(t, result.Scores, a.Scores)
		}()
	}
	wg.Wait()
}
