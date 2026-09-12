package community1

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

type communityKMeansOracle struct {
	Name                      string
	Rows, Dimension, Clusters int
	Input, Normalized         []float32
	Labels                    []int
	Centroids                 []float32
}

func loadCommunityKMeansOracles(t *testing.T) ([]float64, []communityKMeansOracle) {
	t.Helper()
	data, err := os.ReadFile("testdata/kmeans-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema    int
		Reference struct {
			NumPy              string `json:"numpy"`
			Sklearn            string `json:"sklearn"`
			NumPyMTRandSHA256  string `json:"numpy_mtrand_sha256"`
			KMeansPySHA256     string `json:"kmeans_py_sha256"`
			KMeansCommonSHA256 string `json:"kmeans_common_sha256"`
			KMeansLloydSHA256  string `json:"kmeans_lloyd_sha256"`
			RandomState        int    `json:"random_state"`
			NInit              int    `json:"n_init"`
		}
		RandomUniform []float64 `json:"random_uniform"`
		Cases         []communityKMeansOracle
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Reference.NumPy != "2.5.3" || fixture.Reference.Sklearn != "1.9.0" || fixture.Reference.NumPyMTRandSHA256 != "53a0dea25039fd6e55ce2a6c94789c9273fdfa1cc4bda60dde6e813f455c9bc5" || fixture.Reference.KMeansPySHA256 != "7d9cd3c75f1c40616223fbceb23bc1e115de043b355ab90716898301756d746c" || fixture.Reference.KMeansCommonSHA256 != "28273c46e0c190277ad7739aadc086c7dacf6a9ad9e0232a26f223b67274b25e" || fixture.Reference.KMeansLloydSHA256 != "888274137c2f633b5814926ebdea181b7c508222549d0ad7e0c5c4554e831214" || fixture.Reference.RandomState != 42 || fixture.Reference.NInit != 3 || len(fixture.Cases) != 27 || len(fixture.RandomUniform) != 24 {
		t.Fatal("KMeans fixture contract")
	}
	return fixture.RandomUniform, fixture.Cases
}

func TestCommunity1KMeansPinnedOracle(t *testing.T) {
	uniform, cases := loadCommunityKMeansOracles(t)
	rng := newNumpyMT19937(42)
	for i, want := range uniform {
		if got := rng.randomSample(); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("RandomState sample%d got%.17g want%.17g", i, got, want)
		}
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			labels, err := forcedKMeansLabels(context.Background(), tc.Input, tc.Rows, tc.Dimension, tc.Clusters)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(labels, tc.Labels) {
				t.Fatalf("labels got%v want%v", labels, tc.Labels)
			}
			centroids := make([]float32, tc.Clusters*tc.Dimension)
			counts := make([]int, tc.Clusters)
			for row, label := range labels {
				counts[label]++
				for d := 0; d < tc.Dimension; d++ {
					centroids[label*tc.Dimension+d] += tc.Input[row*tc.Dimension+d]
				}
			}
			for cluster, count := range counts {
				if count == 0 {
					t.Fatal("empty result cluster")
				}
				for d := 0; d < tc.Dimension; d++ {
					centroids[cluster*tc.Dimension+d] /= float32(count)
				}
			}
			for i, got := range centroids {
				if diff := math.Abs(float64(got - tc.Centroids[i])); diff > 1e-7 {
					t.Fatalf("centroid%d got%.9g want%.9g diff%g", i, got, tc.Centroids[i], diff)
				}
			}
		})
	}
}

func TestCommunity1KMeansEmptyClusterRelocation(t *testing.T) {
	for _, tc := range []struct {
		name       string
		x, centers []float32
		labels     []int
		counts     []int
		want       []float32
		wantCounts []int
		wantErr    error
	}{
		{name: "one", x: []float32{0, 0, 1, 0, 2, 0, 10, 0}, centers: []float32{0, 0, 0, 0, 10, 0}, labels: []int{0, 0, 0, 2}, counts: []int{3, 0, 1}, want: []float32{1, 0, 2, 0, 10, 0}, wantCounts: []int{2, 1, 1}},
		{name: "multiple-ambiguous", x: []float32{0, 0, 1, 0, 2, 0, 3, 0, 10, 0}, centers: []float32{0, 0, 0, 0, 0, 0, 10, 0}, labels: []int{0, 0, 0, 0, 3}, counts: []int{4, 0, 0, 1}, wantErr: ErrAmbiguousKMeansRelocation},
		{name: "duplicate-only", x: []float32{1, 0, 1, 0, 1, 0, 1, 0}, centers: []float32{1, 0, 1, 0, 1, 0}, labels: []int{0, 0, 0, 0}, counts: []int{4, 0, 0}, want: []float32{4, 0, 0, 0, 0, 0}, wantCounts: []int{4, 0, 0}},
		{name: "ambiguous", x: []float32{-2, 0, -1, 0, 1, 0, 2, 0, 10, 0}, centers: []float32{0, 0, 0, 0, 0, 0, 10, 0}, labels: []int{0, 0, 0, 0, 3}, counts: []int{4, 0, 0, 1}, wantErr: ErrAmbiguousKMeansRelocation},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := make([]float32, len(tc.centers))
			for row, label := range tc.labels {
				for d := 0; d < 2; d++ {
					got[label*2+d] += tc.x[row*2+d]
				}
			}
			counts := append([]int(nil), tc.counts...)
			labelsBefore := append([]int(nil), tc.labels...)
			err := relocateKMeansEmptyClusters(tc.x, tc.centers, got, tc.labels, counts, len(tc.labels), 2)
			if !errors.Is(err, tc.wantErr) || tc.wantErr == nil && (!reflect.DeepEqual(got, tc.want) || !reflect.DeepEqual(counts, tc.wantCounts)) {
				t.Fatalf("got centers%v counts%v err%v", got, counts, err)
			}
			if !reflect.DeepEqual(tc.labels, labelsBefore) {
				t.Fatal("relocation mutated E-step labels")
			}
		})
	}
}

func TestCommunity1KMeansLloydEmptyClusterReference(t *testing.T) {
	for _, tc := range []struct {
		name                string
		x, centers          []float32
		rows, dim, clusters int
		wantLabels          []int
		wantInertia         float32
		wantErr             error
	}{
		{name: "unique-relocation", x: []float32{0, 0, 1, 0, 2, 0, 10, 0}, centers: []float32{0, 0, 0, 0, 10, 0}, rows: 4, dim: 2, clusters: 3, wantLabels: []int{0, 0, 1, 2}, wantInertia: .5},
		{name: "duplicate-only", x: []float32{1, 0, 1, 0, 1, 0, 1, 0}, centers: []float32{1, 0, 1, 0, 1, 0}, rows: 4, dim: 2, clusters: 3, wantLabels: []int{0, 0, 0, 0}},
		{name: "ambiguous", x: []float32{-2, 0, -1, 0, 1, 0, 2, 0, 10, 0}, centers: []float32{0, 0, 0, 0, 0, 0, 10, 0}, rows: 5, dim: 2, clusters: 4, wantErr: ErrAmbiguousKMeansRelocation},
	} {
		t.Run(tc.name, func(t *testing.T) {
			labels, inertia, err := kmeansLloyd(context.Background(), tc.x, tc.rows, tc.dim, tc.clusters, append([]float32(nil), tc.centers...), 0)
			if !errors.Is(err, tc.wantErr) {
				t.Fatal(err)
			}
			if tc.wantErr == nil && (!reflect.DeepEqual(labels, tc.wantLabels) || inertia != tc.wantInertia) {
				t.Fatalf("labels%v inertia%g", labels, inertia)
			}
			if tc.wantErr != nil && labels != nil {
				t.Fatal("partial labels on relocation failure")
			}
		})
	}
}

func TestCommunity1KMeansValidationCancellationOwnership(t *testing.T) {
	_, cases := loadCommunityKMeansOracles(t)
	tc := cases[1]
	for _, test := range []struct {
		rows, dim, clusters int
		input               []float32
	}{{0, tc.Dimension, tc.Clusters, tc.Input}, {tc.Rows, 0, tc.Clusters, tc.Input}, {tc.Rows, tc.Dimension, 0, tc.Input}, {tc.Rows, tc.Dimension, tc.Rows + 1, tc.Input}, {tc.Rows, tc.Dimension, tc.Clusters, tc.Input[:len(tc.Input)-1]}, {512, 512, 64, make([]float32, 512*512)}} {
		if labels, err := forcedKMeansLabels(context.Background(), test.input, test.rows, test.dim, test.clusters); err == nil || labels != nil {
			t.Fatal("invalid geometry accepted", test)
		}
	}
	if !communityKMeansWorkAllowed(64, 256, 8) || communityKMeansWorkAllowed(512, 512, 64) {
		t.Fatal("KMeans work admission")
	}
	bad := append([]float32(nil), tc.Input...)
	bad[0] = float32(math.NaN())
	if labels, err := forcedKMeansLabels(context.Background(), bad, tc.Rows, tc.Dimension, tc.Clusters); err == nil || labels != nil {
		t.Fatal("nonfinite input accepted")
	}
	clear(bad)
	if labels, err := forcedKMeansLabels(context.Background(), bad, tc.Rows, tc.Dimension, tc.Clusters); err == nil || labels != nil {
		t.Fatal("zero rows accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if labels, err := forcedKMeansLabels(ctx, tc.Input, tc.Rows, tc.Dimension, tc.Clusters); !errors.Is(err, context.Canceled) || labels != nil {
		t.Fatal("pre-cancel", err)
	}
	counter := newPowersetContext(0)
	labels, err := forcedKMeansLabels(counter, tc.Input, tc.Rows, tc.Dimension, tc.Clusters)
	counter.cancel()
	if err != nil || labels == nil {
		t.Fatal(err)
	}
	for at := 1; at <= counter.calls; at++ {
		ctx := newPowersetContext(at)
		labels, err := forcedKMeansLabels(ctx, tc.Input, tc.Rows, tc.Dimension, tc.Clusters)
		ctx.cancel()
		if labels != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("cancellation", at, err)
		}
	}
	before := append([]float32(nil), tc.Input...)
	labels, err = forcedKMeansLabels(context.Background(), tc.Input, tc.Rows, tc.Dimension, tc.Clusters)
	if err != nil || !reflect.DeepEqual(tc.Input, before) {
		t.Fatal("source mutation", err)
	}
	labels[0] = 99
	again, err := forcedKMeansLabels(context.Background(), tc.Input, tc.Rows, tc.Dimension, tc.Clusters)
	if err != nil || again[0] == 99 {
		t.Fatal("result storage retained")
	}
	centroidCounter := newPowersetContext(0)
	centroids, err := originalKMeansCentroids(centroidCounter, tc.Input, again, tc.Rows, tc.Dimension, tc.Clusters)
	centroidCounter.cancel()
	if err != nil || centroids == nil {
		t.Fatal("centroid baseline", err)
	}
	for at := 1; at <= centroidCounter.calls; at++ {
		ctx := newPowersetContext(at)
		centroids, err := originalKMeansCentroids(ctx, tc.Input, again, tc.Rows, tc.Dimension, tc.Clusters)
		ctx.cancel()
		if centroids != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("centroid cancellation", at, err)
		}
	}
	badLabels := append([]int(nil), again...)
	badLabels[0] = tc.Clusters
	if centroids, err := originalKMeansCentroids(context.Background(), tc.Input, badLabels, tc.Rows, tc.Dimension, tc.Clusters); err == nil || centroids != nil {
		t.Fatal("invalid centroid label accepted")
	}
}
