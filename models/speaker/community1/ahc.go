// Copyright (c) 2026 Rui Carmo (Go adaptation)
// SPDX-License-Identifier: BSD-3-Clause
// SciPy hierarchy/centroid algorithms adapted as attributed in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
)

// CentroidAHCConfig describes row-major float64 embeddings. Rows2..512,
// dimension1..512, rows*rows*dimension<=2^27, finite threshold>=0. Rows are
// intentionally capped: the pinned generic linkage has O(n^3) worst-case time
// and O(n^2) distances. This development bound is not long-recording admission.
// Inputs must have finite nonzero norms. Single/empty training sets belong to
// the caller's explicit fallback policy, not an invented dendrogram.
type CentroidAHCConfig struct {
	Rows, Dimension int
	Threshold       float64
}

// AHCLink identifies leaves 0..n-1 and prior merges n..n+i-1. Size is leaf
// membership count. Heights can DECREASE under centroid linkage. Output rows
// preserve original merge order; they must not be sorted by height.
type AHCLink struct {
	Left, Right int
	Distance    float64
	Size        int
}

// CentroidAHCResult owns linkage and zero-based labels in SciPy fcluster's
// traversal order (not first occurrence order). Labels are contiguous and
// suitable as VBx initial labels if their count fits its separate slot bound.
type CentroidAHCResult struct {
	Linkage  []AHCLink
	Labels   []int
	Clusters int
}

// CentroidAHC L2-normalises each embedding, computes Euclidean distances, then
// applies SciPy's fast_linkage centroid update and distance-criterion cut. It
// preserves heap/nearest-neighbour ties and uses subtree maximum distance for
// inversion-aware INCLUSIVE thresholding. Float64 synthetic fixtures qualify
// this component only; real float32 normalisation can have different ties.
// No raw checkpoint, AHC defaults mutation, native runtime, SIMD or GPU work.
// Sources immutable during call; all output owned; no partial cancellation.
func CentroidAHC(ctx context.Context, embeddings []float64, cfg CentroidAHCConfig) (*CentroidAHCResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := cfg
	if c.Rows < 2 || c.Rows > 512 || c.Dimension < 1 || c.Dimension > 512 || len(embeddings) != c.Rows*c.Dimension || math.IsNaN(c.Threshold) || math.IsInf(c.Threshold, 0) || c.Threshold < 0 || int64(c.Rows)*int64(c.Rows)*int64(c.Dimension) > 1<<27 {
		return nil, fmt.Errorf("invalid centroid AHC geometry/policy")
	}
	if err := finiteClustering64(ctx, embeddings); err != nil {
		return nil, err
	}
	normalized := make([]float64, len(embeddings))
	for row := 0; row < c.Rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		x := embeddings[row*c.Dimension : (row+1)*c.Dimension]
		norm, err := cosineNorm(x)
		if err != nil {
			return nil, err
		}
		for d, v := range x {
			normalized[row*c.Dimension+d] = v / norm
		}
	}
	n := c.Rows
	distances := make([]float64, n*(n-1)/2)
	for i := 0; i < n-1; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		for j := i + 1; j < n; j++ {
			if j%32 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			sum := float64(0)
			for d := 0; d < c.Dimension; d++ {
				delta := normalized[i*c.Dimension+d] - normalized[j*c.Dimension+d]
				sum += delta * delta
			}
			distances[ahcIndex(n, i, j)] = math.Sqrt(sum)
		}
	}
	links, err := centroidLinkage(ctx, distances, n)
	if err != nil {
		return nil, err
	}
	labels, count, err := cutCentroidLinkage(ctx, links, n, c.Threshold)
	if err != nil {
		return nil, err
	}
	return &CentroidAHCResult{links, labels, count}, nil
}
func ahcIndex(n, i, j int) int {
	if i > j {
		i, j = j, i
	}
	return n*i - i*(i+1)/2 + j - i - 1
}

// distances is call-owned condensed storage; the generic source algorithm
// replaces the larger active slot and deletes the smaller one at each merge.
func centroidLinkage(ctx context.Context, distances []float64, n int) ([]AHCLink, error) {
	sizes, ids := make([]int, n), make([]int, n)
	neighbor := make([]int, n-1)
	lower := make([]float64, n-1)
	for i := range sizes {
		sizes[i] = 1
		ids[i] = i
	}
	nearest := func(x int) (int, float64, error) {
		best, minimum := -1, math.Inf(1)
		for i := x + 1; i < n; i++ {
			if i%128 == 0 {
				if err := ctx.Err(); err != nil {
					return -1, 0, err
				}
			}
			if sizes[i] == 0 {
				continue
			}
			v := distances[ahcIndex(n, x, i)]
			if v < minimum {
				minimum = v
				best = i
			}
		}
		if best < 0 {
			return -1, 0, fmt.Errorf("centroid AHC has no finite neighbour")
		}
		return best, minimum, nil
	}
	for x := 0; x < n-1; x++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		y, v, err := nearest(x)
		if err != nil {
			return nil, err
		}
		neighbor[x], lower[x] = y, v
	}
	heap := newAHCHeap(lower)
	links := make([]AHCLink, n-1)
	for k := 0; k < n-1; k++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		x, y := 0, 0
		distance := float64(0)
		resolved := false
		for retry := 0; retry < n-k; retry++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			x, distance = heap.keyByIndex[0], heap.values[0]
			y = neighbor[x]
			if distance == distances[ahcIndex(n, x, y)] {
				resolved = true
				break
			}
			var err error
			y, distance, err = nearest(x)
			if err != nil {
				return nil, err
			}
			neighbor[x], lower[x] = y, distance
			heap.change(x, distance)
		}
		if !resolved {
			return nil, fmt.Errorf("centroid AHC nearest bound did not converge")
		}
		heap.removeMin()
		nx, ny := sizes[x], sizes[y]
		left, right := ids[x], ids[y]
		if left > right {
			left, right = right, left
		}
		links[k] = AHCLink{left, right, distance, nx + ny}
		sizes[x] = 0
		sizes[y] = nx + ny
		ids[y] = n + k
		for z := 0; z < n; z++ {
			if z%128 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			if sizes[z] == 0 || z == y {
				continue
			}
			dx, dy := distances[ahcIndex(n, z, x)], distances[ahcIndex(n, z, y)]
			// Preserve source operation order; no silent clamp of negative radicands.
			total := float64(nx + ny)
			square := (((float64(nx) * dx * dx) + (float64(ny) * dy * dy)) - (float64(nx*ny)*distance*distance)/total) / total
			value := math.Sqrt(square)
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("nonfinite centroid distance update")
			}
			distances[ahcIndex(n, z, y)] = value
		}
		for z := 0; z < x; z++ {
			if sizes[z] > 0 && neighbor[z] == x {
				neighbor[z] = y
			}
		}
		for z := 0; z < y; z++ {
			if sizes[z] == 0 {
				continue
			}
			value := distances[ahcIndex(n, z, y)]
			if value < lower[z] {
				neighbor[z], lower[z] = y, value
				heap.change(z, value)
			}
		}
		if y < n-1 {
			z, value, err := nearest(y)
			if err != nil {
				return nil, err
			}
			neighbor[y], lower[y] = z, value
			heap.change(y, value)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return links, nil
}

// The generated topology always references prior rows. A linear bottom-up
// maximum is equivalent to source get_max_dist_for_each_cluster's postorder.
// Label traversal deliberately visits internal children BEFORE leaf siblings;
// ordinary left-to-right leaf DFS changes fcluster numbering.
func cutCentroidLinkage(ctx context.Context, links []AHCLink, n int, threshold float64) ([]int, int, error) {
	maxima := make([]float64, n-1)
	for i, link := range links {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		v := link.Distance
		if link.Left >= n {
			v = math.Max(v, maxima[link.Left-n])
		}
		if link.Right >= n {
			v = math.Max(v, maxima[link.Right-n])
		}
		maxima[i] = v
	}
	labels := make([]int, n)
	stack := make([]int, n)
	visited := make([]bool, 2*n-1)
	top := 0
	stack[0] = 2*n - 2
	leader := -1
	clusters := 0
	for top >= 0 {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		root := stack[top] - n
		link := links[root]
		if leader == -1 && maxima[root] <= threshold {
			leader = root
			clusters++
		}
		if link.Left >= n && !visited[link.Left] {
			visited[link.Left] = true
			top++
			stack[top] = link.Left
			continue
		}
		if link.Right >= n && !visited[link.Right] {
			visited[link.Right] = true
			top++
			stack[top] = link.Right
			continue
		}
		if link.Left < n {
			if leader == -1 {
				clusters++
			}
			labels[link.Left] = clusters - 1
		}
		if link.Right < n {
			if leader == -1 {
				clusters++
			}
			labels[link.Right] = clusters - 1
		}
		if leader == root {
			leader = -1
		}
		top--
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	return labels, clusters, nil
}
