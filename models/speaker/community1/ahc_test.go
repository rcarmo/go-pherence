package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"os"
	"reflect"
	"sync"
	"testing"
)

type ahcOracle struct {
	Name     string
	Config   CentroidAHCConfig
	Input    []float64
	Linkage  []AHCLink
	Labels   []int
	Clusters int
}
type ahcBridge struct {
	Gamma, Priors, Centroids, Scores []float64
	Labels                           []int
	Speakers, Clusters               int
}

func loadAHCOracles(t *testing.T) ([]ahcOracle, ahcBridge) {
	t.Helper()
	data, err := os.ReadFile("testdata/ahc-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Schema    int
		Abs       float64 `json:"absolute_tolerance"`
		Rel       float64 `json:"relative_tolerance"`
		Reference struct {
			Revision string            `json:"scipy_revision"`
			Hashes   map[string]string `json:"source_hashes"`
		}
		Cases  []ahcOracle
		Bridge ahcBridge
	}
	if err := json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	if f.Schema != 1 || f.Abs != 2e-10 || f.Rel != 2e-10 || len(f.Cases) != 42 || f.Reference.Revision != "e4e854eaa8f18d807cd3496028e257e36caa93cc" || f.Reference.Hashes["hierarchy"] != "50141ab68a04ee82d51bbba906afdf29a7a79877223f9e89699500e926dabf9f" || f.Reference.Hashes["structures"] != "1bf594add2b5f92d07a0649cf8fd54ae5b2352a75ce07dcc462d82a821493e3d" || f.Reference.Hashes["updates"] != "64d6c695524fc08fc1d4c9679bd4424b28c15232b9776df4224ce432b8ee725f" {
		t.Fatal("AHC fixture contract")
	}
	return f.Cases, f.Bridge
}
func checkAHC(t *testing.T, got *CentroidAHCResult, want ahcOracle) {
	t.Helper()
	if len(got.Linkage) != len(want.Linkage) || got.Clusters != want.Clusters || !reflect.DeepEqual(got.Labels, want.Labels) {
		t.Fatalf("AHC label mismatch %s got%v want%v", want.Name, got.Labels, want.Labels)
	}
	for i, link := range got.Linkage {
		w := want.Linkage[i]
		if link.Left != w.Left || link.Right != w.Right || link.Size != w.Size {
			t.Fatalf("AHC merge %s/%d got%v want%v", want.Name, i, link, w)
		}
		closeClustering64(t, []float64{link.Distance}, []float64{w.Distance})
	}
}
func TestCentroidAHCPinnedOracle(t *testing.T) {
	cases, _ := loadAHCOracles(t)
	merges := 0
	inversions := 0
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			input := append([]float64(nil), c.Input...)
			out, err := CentroidAHC(context.Background(), input, c.Config)
			if err != nil {
				t.Fatal(err)
			}
			checkAHC(t, out, c)
			if !reflect.DeepEqual(input, c.Input) {
				t.Fatal("input mutation")
			}
			for i := range input {
				input[i] = 99
			}
			checkAHC(t, out, c)
			merges += len(out.Linkage)
			inverted := false
			for i := 1; i < len(out.Linkage); i++ {
				if out.Linkage[i].Distance < out.Linkage[i-1].Distance {
					inverted = true
				}
			}
			if inverted {
				inversions++
			}
		})
	}
	if merges != 403 || inversions != 21 {
		t.Fatal("fixture coverage changed", merges, inversions)
	}
}
func TestCentroidAHCInversionCutAndTieOrder(t *testing.T) {
	// Source fcluster must not accept the root at1.5 when a child height is2.
	links := []AHCLink{{0, 1, 2, 2}, {2, 3, 1, 3}}
	got, count, err := cutCentroidLinkage(context.Background(), links, 3, 1.5)
	if err != nil || count != 3 || !reflect.DeepEqual(got, []int{0, 1, 2}) {
		t.Fatal("inversion cut / internal-before-leaf order", got, count, err)
	}
	got, count, err = cutCentroidLinkage(context.Background(), links, 3, 2)
	if err != nil || count != 1 || !reflect.DeepEqual(got, []int{0, 0, 0}) {
		t.Fatal("inclusive subtree cut", got, count, err)
	}
	// Maximum supported row count, small dimension, duplicates to exercise equal
	// distance heap keys and zero-distance updates without a heavy workload.
	x := make([]float64, 512)
	for i := range x {
		x[i] = 1
	}
	result, err := CentroidAHC(context.Background(), x, CentroidAHCConfig{512, 1, 0})
	if err != nil || len(result.Linkage) != 511 || result.Clusters != 1 {
		t.Fatal("bounded512", err)
	}
}
func TestAHCPreparedClusteringBridge(t *testing.T) {
	cases, want := loadAHCOracles(t)
	a := cases[len(cases)-1]
	ps, _ := loadPLDAVBxOracles(t)
	p := ps[1]
	initial, err := CentroidAHC(context.Background(), a.Input, a.Config)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewPreparedPLDA(context.Background(), p.Config, p.Weights)
	if err != nil {
		t.Fatal(err)
	}
	features, err := model.Transform(context.Background(), a.Input, a.Config.Rows)
	if err != nil {
		t.Fatal(err)
	}
	vb, err := ClusterVBx(context.Background(), features, model.Phi(), initial.Labels, Community1VBxConfig(a.Config.Rows, p.Config.OutputDim))
	if err != nil {
		t.Fatal(err)
	}
	closeClustering64(t, vb.Responsibilities, want.Gamma)
	closeClustering64(t, vb.Priors, want.Priors)
	centers, err := ComputeVBxCentroids(context.Background(), a.Input, vb.Responsibilities, vb.Priors, VBxCentroidConfig{a.Config.Rows, a.Config.Dimension, vb.Speakers})
	if err != nil {
		t.Fatal(err)
	}
	closeClustering64(t, centers.Centroids, want.Centroids)
	seg := make([]float32, 5*a.Config.Rows)
	for i := range seg {
		seg[i] = 1
	}
	assignment, err := AssignCosineSpeakers(context.Background(), a.Input, centers.Centroids, seg, CosineAssignmentConfig{1, a.Config.Rows, want.Clusters, a.Config.Dimension, 5, true})
	if err != nil {
		t.Fatal(err)
	}
	closeClustering64(t, assignment.Scores, want.Scores)
	if !reflect.DeepEqual(assignment.Labels, want.Labels) {
		t.Fatal("AHC bridge labels", assignment.Labels, want.Labels)
	}
}
func TestCentroidAHCHeapInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(9182))
	for n := 1; n <= 64; n++ {
		values := make([]float64, n)
		active := make([]bool, n)
		for i := range values {
			values[i] = float64(rng.Intn(7))
			active[i] = true
		}
		h := newAHCHeap(values)
		for h.size > 0 {
			// Repeated ties, increasing and decreasing keys exercise both sifts.
			for attempt := 0; attempt < 3; attempt++ {
				key := h.keyByIndex[rng.Intn(h.size)]
				value := float64(rng.Intn(9) - 4)
				values[key] = value
				h.change(key, value)
			}
			minimum := math.Inf(1)
			for key, value := range values {
				if active[key] {
					minimum = math.Min(minimum, value)
				}
			}
			if h.values[0] != minimum {
				t.Fatal("heap minimum", n, h.size)
			}
			for index := 0; index < h.size; index++ {
				key := h.keyByIndex[index]
				if !active[key] || h.indexByKey[key] != index || h.values[index] != values[key] {
					t.Fatal("heap key mapping")
				}
				if index > 0 && h.values[(index-1)/2] > h.values[index] {
					t.Fatal("heap order")
				}
			}
			active[h.keyByIndex[0]] = false
			h.removeMin()
		}
	}
}

func TestCentroidAHCValidation(t *testing.T) {
	for _, kind := range []string{"rows_zero", "rows_one", "rows_huge", "dim_zero", "dim_huge", "threshold_negative", "threshold_nan", "threshold_inf", "short", "long", "zero", "nan", "inf", "overflow", "underflow"} {
		cfg := CentroidAHCConfig{3, 2, .6}
		x := []float64{1, 0, 0, 1, 1, 1}
		switch kind {
		case "rows_zero":
			cfg.Rows = 0
		case "rows_one":
			cfg.Rows = 1
		case "rows_huge":
			cfg.Rows = int(^uint(0) >> 1)
		case "dim_zero":
			cfg.Dimension = 0
		case "dim_huge":
			cfg.Dimension = 513
		case "threshold_negative":
			cfg.Threshold = -1
		case "threshold_nan":
			cfg.Threshold = math.NaN()
		case "threshold_inf":
			cfg.Threshold = math.Inf(1)
		case "short":
			x = x[:5]
		case "long":
			x = append(x, 0)
		case "zero":
			x[0] = 0
		case "nan":
			x[0] = math.NaN()
		case "inf":
			x[0] = math.Inf(-1)
		case "overflow":
			x[0] = math.MaxFloat64
		case "underflow":
			x[0] = math.SmallestNonzeroFloat64
		}
		if out, err := CentroidAHC(context.Background(), x, cfg); err == nil || out != nil {
			t.Fatal("accepted AHC", kind)
		}
	}
	// Private update guard: impossible distances must not silently clamp to0.
	if out, err := centroidLinkage(context.Background(), []float64{1, math.Inf(1), math.NaN()}, 3); err == nil || out != nil {
		t.Fatal("accepted malformed condensed distances")
	}
}
func TestCentroidAHCCancellationAndConcurrency(t *testing.T) {
	cases, _ := loadAHCOracles(t)
	c := cases[len(cases)-2]
	count := newPowersetContext(0)
	_, err := CentroidAHC(count, c.Input, c.Config)
	count.cancel()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("AHC cancellation checkpoints %d", count.calls)
	for at := 1; at <= count.calls; at++ {
		ctx := newPowersetContext(at)
		out, err := CentroidAHC(ctx, c.Input, c.Config)
		ctx.cancel()
		if out != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("AHC cancellation", at, err)
		}
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out, err := CentroidAHC(context.Background(), c.Input, c.Config)
			if err != nil {
				t.Error(err)
				return
			}
			checkAHC(t, out, c)
		}()
	}
	wg.Wait()
	out, err := CentroidAHC(context.Background(), c.Input, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	for i := range out.Labels {
		out.Labels[i] = -100
	}
	for i := range out.Linkage {
		out.Linkage[i].Distance = 99
	}
	fresh, err := CentroidAHC(context.Background(), c.Input, c.Config)
	if err != nil {
		t.Fatal(err)
	}
	checkAHC(t, fresh, c)
}
