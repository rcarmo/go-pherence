// Copyright (c) 2007-2026 The scikit-learn developers.
// Copyright (c) 2026 Rui Carmo.
// SPDX-License-Identifier: BSD-3-Clause
package community1

import (
	"context"
	"errors"
	"fmt"
	"math"
)

const (
	communityKMeansSeed       = uint32(42)
	communityKMeansInits      = 3
	communityKMeansMaxIter    = 300
	communityKMeansMaxWorkOps = uint64(1 << 28)
)

// ErrAmbiguousKMeansRelocation means an empty Lloyd cluster would depend on
// NumPy's ISA-selected argpartition identity/order for equal-distance donors.
// The checked Go path fails instead of assigning a non-portable speaker ID.
var ErrAmbiguousKMeansRelocation = errors.New("ambiguous KMeans empty-cluster relocation")

// numpyMT19937 implements the legacy NumPy RandomState MT19937 stream used by
// the pinned scikit-learn KMeans(random_state=42) fallback. randomSample follows
// RandomState.random_sample's 53-bit double construction.
type numpyMT19937 struct {
	state [624]uint32
	index int
}

func newNumpyMT19937(seed uint32) *numpyMT19937 {
	r := &numpyMT19937{}
	r.state[0] = seed
	for i := 1; i < len(r.state); i++ {
		r.state[i] = 1812433253*(r.state[i-1]^(r.state[i-1]>>30)) + uint32(i)
	}
	r.index = len(r.state)
	return r
}
func (r *numpyMT19937) uint32() uint32 {
	if r.index >= len(r.state) {
		for i := range r.state {
			y := (r.state[i] & 0x80000000) | (r.state[(i+1)%624] & 0x7fffffff)
			r.state[i] = r.state[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				r.state[i] ^= 0x9908b0df
			}
		}
		r.index = 0
	}
	y := r.state[r.index]
	r.index++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}
func (r *numpyMT19937) randomSample() float64 {
	a, b := uint64(r.uint32()>>5), uint64(r.uint32()>>6)
	return float64(a*67108864+b) / 9007199254740992
}

// forcedKMeansLabels reproduces the bounded float32 path used by pinned
// scikit-learn 1.9 KMeans(n_init=3,random_state=42,copy_x=False). Input is
// row-major original-space float32 embeddings; cosine-normalised rows are
// clustered with Euclidean Lloyd iterations. Returned labels own their storage.
// Empty-cluster relocation fails when NumPy's ISA-selected argpartition would
// control donor identity/order; see ErrAmbiguousKMeansRelocation.
func forcedKMeansLabels(ctx context.Context, input []float32, rows, dimension, clusters int) ([]int, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 KMeans: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if rows < 2 || rows > 512 || dimension < 1 || dimension > 512 || clusters < 1 || clusters > 64 || clusters > rows || len(input) != rows*dimension || rows*dimension > 1<<24 || !communityKMeansWorkAllowed(rows, dimension, clusters) {
		return nil, fmt.Errorf("Community-1 KMeans: invalid geometry/work bound")
	}
	if clusters == 1 {
		for row := 0; row < rows; row++ {
			if row%32 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			var norm float32
			for d := 0; d < dimension; d++ {
				value := input[row*dimension+d]
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return nil, fmt.Errorf("Community-1 KMeans: nonfinite input")
				}
				norm += value * value
			}
			if norm <= 0 || math.IsInf(float64(norm), 0) {
				return nil, fmt.Errorf("Community-1 KMeans: zero/nonfinite norm")
			}
		}
		return make([]int, rows), nil
	}
	x := make([]float32, len(input))
	mean := make([]float32, dimension)
	for row := 0; row < rows; row++ {
		if row%32 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		var norm float32
		for d := 0; d < dimension; d++ {
			v := input[row*dimension+d]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("Community-1 KMeans: nonfinite input")
			}
			norm += v * v
		}
		if norm <= 0 || math.IsInf(float64(norm), 0) {
			return nil, fmt.Errorf("Community-1 KMeans: zero/nonfinite norm")
		}
		inv := float32(1 / math.Sqrt(float64(norm)))
		for d := 0; d < dimension; d++ {
			v := input[row*dimension+d] * inv
			x[row*dimension+d] = v
			mean[d] += v
		}
	}
	for d := range mean {
		mean[d] /= float32(rows)
	}
	for row := 0; row < rows; row++ {
		for d := 0; d < dimension; d++ {
			x[row*dimension+d] -= mean[d]
		}
	}
	// scikit-learn scales tol by the mean per-feature population variance.
	var tolerance float32
	for d := 0; d < dimension; d++ {
		var variance float32
		for row := 0; row < rows; row++ {
			v := x[row*dimension+d]
			variance += v * v
		}
		tolerance += variance / float32(rows)
	}
	tolerance = tolerance / float32(dimension) * 1e-4

	rng := newNumpyMT19937(communityKMeansSeed)
	var bestLabels []int
	bestInertia := float32(math.Inf(1))
	for init := 0; init < communityKMeansInits; init++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		centers, err := kmeansPlusPlus(ctx, x, rows, dimension, clusters, rng)
		if err != nil {
			return nil, err
		}
		labels, inertia, err := kmeansLloyd(ctx, x, rows, dimension, clusters, centers, tolerance)
		if err != nil {
			return nil, err
		}
		if bestLabels == nil || inertia < bestInertia && !sameClustering(labels, bestLabels, clusters) {
			bestLabels = append(bestLabels[:0], labels...)
			bestInertia = inertia
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return bestLabels, nil
}

func communityKMeansWorkAllowed(rows, dimension, clusters int) bool {
	if rows < 1 || dimension < 1 || clusters < 1 {
		return false
	}
	work := uint64(rows) * uint64(dimension) * uint64(clusters) * communityKMeansMaxIter * communityKMeansInits
	return work <= communityKMeansMaxWorkOps
}

func squaredDistanceF32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

func kmeansPlusPlus(ctx context.Context, x []float32, rows, dim, clusters int, rng *numpyMT19937) ([]float32, error) {
	centers := make([]float32, clusters*dim)
	closest := make([]float32, rows)
	first := min(int(rng.randomSample()*float64(rows)), rows-1)
	copy(centers[:dim], x[first*dim:(first+1)*dim])
	var potential float64
	for row := 0; row < rows; row++ {
		closest[row] = squaredDistanceF32(centers[:dim], x[row*dim:(row+1)*dim])
		potential += float64(closest[row])
	}
	trials := 2 + int(math.Log(float64(clusters)))
	candidateDistances := make([]float32, trials*rows)
	for cluster := 1; cluster < clusters; cluster++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		bestTrial, bestPotential := 0, math.Inf(1)
		candidateIDs := make([]int, trials)
		for trial := 0; trial < trials; trial++ {
			target, cumulative := rng.randomSample()*potential, float64(0)
			candidateIDs[trial] = rows - 1
			for row, distance := range closest {
				cumulative += float64(distance)
				if cumulative >= target {
					candidateIDs[trial] = row
					break
				}
			}
			candidate := x[candidateIDs[trial]*dim : (candidateIDs[trial]+1)*dim]
			var candidatePotential float64
			for row := 0; row < rows; row++ {
				distance := squaredDistanceF32(candidate, x[row*dim:(row+1)*dim])
				distance = min(distance, closest[row])
				candidateDistances[trial*rows+row] = distance
				candidatePotential += float64(distance)
			}
			if candidatePotential < bestPotential {
				bestTrial, bestPotential = trial, candidatePotential
			}
		}
		copy(closest, candidateDistances[bestTrial*rows:(bestTrial+1)*rows])
		potential = bestPotential
		id := candidateIDs[bestTrial]
		copy(centers[cluster*dim:(cluster+1)*dim], x[id*dim:(id+1)*dim])
	}
	return centers, nil
}

func kmeansLloyd(ctx context.Context, x []float32, rows, dim, clusters int, centers []float32, tolerance float32) ([]int, float32, error) {
	labels, old := make([]int, rows), make([]int, rows)
	for i := range old {
		old[i] = -1
	}
	newCenters := make([]float32, len(centers))
	counts := make([]int, clusters)
	strict := false
	for iteration := 0; iteration < communityKMeansMaxIter; iteration++ {
		clear(newCenters)
		clear(counts)
		for row := 0; row < rows; row++ {
			if row%32 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, 0, err
				}
			}
			point := x[row*dim : (row+1)*dim]
			best, bestDistance := 0, squaredDistanceF32(point, centers[:dim])
			for cluster := 1; cluster < clusters; cluster++ {
				distance := squaredDistanceF32(point, centers[cluster*dim:(cluster+1)*dim])
				if distance < bestDistance {
					best, bestDistance = cluster, distance
				}
			}
			labels[row] = best
			counts[best]++
			for d, value := range point {
				newCenters[best*dim+d] += value
			}
		}
		if err := relocateKMeansEmptyClusters(x, centers, newCenters, labels, counts, rows, dim); err != nil {
			return nil, 0, err
		}
		// scikit-learn averages nonempty sums and copies the largest cluster's
		// center into any remaining duplicate-only empty slots.
		largest := 0
		for cluster := 1; cluster < clusters; cluster++ {
			if counts[cluster] > counts[largest] {
				largest = cluster
			}
		}
		var shift float32
		for cluster := 0; cluster < clusters; cluster++ {
			if counts[cluster] > 0 {
				for d := 0; d < dim; d++ {
					newCenters[cluster*dim+d] /= float32(counts[cluster])
				}
			} else {
				copy(newCenters[cluster*dim:(cluster+1)*dim], newCenters[largest*dim:(largest+1)*dim])
			}
			for d := 0; d < dim; d++ {
				index := cluster*dim + d
				delta := newCenters[index] - centers[index]
				shift += delta * delta
			}
		}
		centers, newCenters = newCenters, centers
		strict = true
		for i := range labels {
			if labels[i] != old[i] {
				strict = false
				break
			}
		}
		if strict || shift <= tolerance {
			break
		}
		copy(old, labels)
	}
	if !strict {
		for row := 0; row < rows; row++ {
			point := x[row*dim : (row+1)*dim]
			best, bestDistance := 0, squaredDistanceF32(point, centers[:dim])
			for cluster := 1; cluster < clusters; cluster++ {
				distance := squaredDistanceF32(point, centers[cluster*dim:(cluster+1)*dim])
				if distance < bestDistance {
					best, bestDistance = cluster, distance
				}
			}
			labels[row] = best
		}
	}
	var inertia float32
	for row, label := range labels {
		inertia += squaredDistanceF32(x[row*dim:(row+1)*dim], centers[label*dim:(label+1)*dim])
	}
	return labels, inertia, nil
}

type kmeansRelocationCandidate struct {
	row      int
	distance float32
}

func relocateKMeansEmptyClusters(x, oldCenters, newCenters []float32, labels, counts []int, rows, dim int) error {
	empty := make([]int, 0, len(counts))
	for cluster, count := range counts {
		if count == 0 {
			empty = append(empty, cluster)
		}
	}
	if len(empty) == 0 {
		return nil
	}
	maximum := kmeansRelocationCandidate{row: -1, distance: -1}
	maximumCount := 0
	for row, label := range labels {
		distance := squaredDistanceF32(x[row*dim:(row+1)*dim], oldCenters[label*dim:(label+1)*dim])
		if distance > maximum.distance {
			maximum = kmeansRelocationCandidate{row: row, distance: distance}
			maximumCount = 1
		} else if distance == maximum.distance {
			maximumCount++
		}
	}
	if maximum.distance == 0 {
		return nil
	}
	// NumPy argpartition is ISA-dispatched. A unique maximum selects the same
	// donor for one empty cluster on every checked implementation. Multiple
	// empties have a non-portable donor-to-cluster order even without ties.
	if len(empty) != 1 || maximumCount != 1 {
		return ErrAmbiguousKMeansRelocation
	}
	cluster, row := empty[0], maximum.row
	oldCluster := labels[row]
	for d := 0; d < dim; d++ {
		value := x[row*dim+d]
		newCenters[oldCluster*dim+d] -= value
		newCenters[cluster*dim+d] = value
	}
	counts[oldCluster]--
	counts[cluster] = 1
	return nil
}

func originalKMeansCentroids(ctx context.Context, input []float32, labels []int, rows, dimension, clusters int) ([]float64, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 KMeans centroids: nil context")
	}
	if rows < 2 || dimension < 1 || clusters < 1 || clusters > rows || len(input) != rows*dimension || len(labels) != rows {
		return nil, fmt.Errorf("Community-1 KMeans centroids: invalid geometry")
	}
	out := make([]float64, clusters*dimension)
	counts := make([]int, clusters)
	for row, label := range labels {
		if row%32 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		if label < 0 || label >= clusters {
			return nil, fmt.Errorf("Community-1 KMeans centroids: invalid label")
		}
		counts[label]++
		for d := 0; d < dimension; d++ {
			out[label*dimension+d] += float64(input[row*dimension+d])
		}
	}
	for cluster, count := range counts {
		if count == 0 {
			return nil, fmt.Errorf("Community-1 KMeans centroids: empty cluster")
		}
		for d := 0; d < dimension; d++ {
			out[cluster*dimension+d] /= float64(count)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func sameClustering(a, b []int, clusters int) bool {
	if len(a) != len(b) {
		return false
	}
	mapping := make([]int, clusters)
	for i := range mapping {
		mapping[i] = -1
	}
	for i := range a {
		if a[i] < 0 || a[i] >= clusters || b[i] < 0 || b[i] >= clusters {
			return false
		}
		if mapping[a[i]] < 0 {
			mapping[a[i]] = b[i]
		} else if mapping[a[i]] != b[i] {
			return false
		}
	}
	return true
}
