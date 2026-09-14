// Copyright (c) 2026 Rui Carmo (Go adaptation)
// SPDX-License-Identifier: BSD-3-Clause
// SciPy hierarchy Heap adaptation; attribution and licence in NOTICE.
package community1

// ahcHeap retains SciPy's strict comparisons and implicit natural-number keys.
// Equal values are not ordered by key: adding a secondary key would alter ties.
type ahcHeap struct {
	values                 []float64
	keyByIndex, indexByKey []int
	size                   int
}

func newAHCHeap(values []float64) *ahcHeap {
	n := len(values)
	h := &ahcHeap{append([]float64(nil), values...), make([]int, n), make([]int, n), n}
	for i := 0; i < n; i++ {
		h.keyByIndex[i] = i
		h.indexByKey[i] = i
	}
	for i := n/2 - 1; i >= 0; i-- {
		h.down(i)
	}
	return h
}
func (h *ahcHeap) swap(i, j int) {
	h.values[i], h.values[j] = h.values[j], h.values[i]
	ki, kj := h.keyByIndex[i], h.keyByIndex[j]
	h.keyByIndex[i], h.keyByIndex[j] = kj, ki
	h.indexByKey[ki], h.indexByKey[kj] = j, i
}
func (h *ahcHeap) down(index int) {
	for child := 2*index + 1; child < h.size; child = 2*index + 1 {
		if child+1 < h.size && h.values[child+1] < h.values[child] {
			child++
		}
		if h.values[index] <= h.values[child] {
			break
		}
		h.swap(index, child)
		index = child
	}
}
func (h *ahcHeap) change(key int, value float64) {
	index := h.indexByKey[key]
	old := h.values[index]
	h.values[index] = value
	if value < old {
		for index > 0 {
			parent := (index - 1) / 2
			if h.values[parent] <= h.values[index] {
				break
			}
			h.swap(index, parent)
			index = parent
		}
	} else {
		h.down(index)
	}
}
func (h *ahcHeap) removeMin() { h.swap(0, h.size-1); h.size--; h.down(0) }
