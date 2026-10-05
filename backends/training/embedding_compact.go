package training

import (
	"fmt"
	"math"
	"sort"
)

// Compact embedding index layout: rows token slots, rows+1 bucket offsets,
// then rows occurrence row IDs. Active tokens are ascending; inactive slots
// are -1 and have empty buckets. Output has rows*width elements, so the dense
// parameter table is not needed to materialise this VJP.
func BuildCompactEmbeddingIndex(ids []float32, vocabulary int) ([]float32, error) {
	if _, _, err := EmbeddingShape(len(ids), 1, vocabulary, 3); err != nil {
		return nil, err
	}
	tokens := make([]int, len(ids))
	for i, id := range ids {
		if math.IsNaN(float64(id)) || math.IsInf(float64(id), 0) || id < 0 || id >= float32(vocabulary) || id != float32(int(id)) {
			return nil, fmt.Errorf("invalid compact embedding ID")
		}
		tokens[i] = int(id)
	}
	sort.Ints(tokens)
	active := 0
	for _, id := range tokens {
		if active == 0 || tokens[active-1] != id {
			tokens[active] = id
			active++
		}
	}
	rows := len(ids)
	index := make([]float32, 3*rows+1)
	for slot := 0; slot < rows; slot++ {
		index[slot] = -1
	}
	at := 0
	for slot, token := range tokens[:active] {
		index[slot] = float32(token)
		index[rows+slot] = float32(at)
		for row, id := range ids {
			if int(id) == token {
				index[2*rows+1+at] = float32(row)
				at++
			}
		}
	}
	for slot := active; slot <= rows; slot++ {
		index[rows+slot] = float32(rows)
	}
	return index, nil
}

func ValidateCompactEmbeddingIndex(index []float32, rows, vocabulary int) error {
	if _, _, err := EmbeddingShape(rows, 1, vocabulary, 3); err != nil {
		return err
	}
	if len(index) != 3*rows+1 {
		return fmt.Errorf("compact embedding index shape")
	}
	integer := func(x float32, bound int) bool {
		return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) && x >= 0 && x <= float32(bound) && x == float32(int(x))
	}
	if index[rows] != 0 || index[2*rows] != float32(rows) {
		return fmt.Errorf("compact embedding index extent")
	}
	seen := make([]bool, rows)
	previousToken := -1
	inactive := false
	for slot := 0; slot < rows; slot++ {
		token, lo, hi := index[slot], index[rows+slot], index[rows+slot+1]
		if !integer(lo, rows) || !integer(hi, rows) || lo > hi {
			return fmt.Errorf("compact embedding index offset")
		}
		if token == -1 {
			inactive = true
			if lo != hi {
				return fmt.Errorf("compact embedding padding bucket")
			}
			continue
		}
		if inactive || !integer(token, vocabulary-1) || int(token) <= previousToken || lo == hi {
			return fmt.Errorf("compact embedding token/order")
		}
		previousToken = int(token)
		previousRow := -1
		for at := int(lo); at < int(hi); at++ {
			row := index[2*rows+1+at]
			if !integer(row, rows-1) || int(row) <= previousRow || seen[int(row)] {
				return fmt.Errorf("compact embedding row/order/duplicate")
			}
			seen[int(row)] = true
			previousRow = int(row)
		}
	}
	// Offsets traverse exactly rows entries; in-range uniqueness proves coverage.
	return nil
}
