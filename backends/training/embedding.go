package training

import (
	"fmt"
	"math"
)

// EmbeddingShape bounds gather (mode0) and deterministic dense table VJP
// (mode1), indexed dense VJP (mode2), or compact indexed VJP (mode3).
// Modes0/1 use exactly represented
// F32 token IDs; mode2 uses validated occurrence offsets/row indices. One
// output element owns each reduction; repeated tokens add in sequence order.
// Mode3 writes compact active-token slots padded to rows; its packed index
// has separate admission. Raw callers must supply admitted IDs/indices.
func EmbeddingShape(rows, width, vocabulary int, mode uint32) (source, output int, err error) {
	if rows < 1 || rows > 1024 || width < 1 || width > 16384 || vocabulary < 1 || vocabulary > 65536 || vocabulary > (1<<26)/width || mode > 3 {
		return 0, 0, fmt.Errorf("invalid training embedding shape/mode")
	}
	source, output = vocabulary*width, rows*width
	if mode != 0 {
		source, output = output, source
	}
	if mode == 3 {
		output = source
	} // compact active-token slots, padded to rows
	return source, output, nil
}

// BuildEmbeddingIndex packs vocabulary+1 offsets followed by rows row IDs.
// Token occurrences are stable in sequence order, preserving the mode1 sum.
// F32 represents every admitted offset and row ID exactly.
func BuildEmbeddingIndex(ids []float32, vocabulary int) ([]float32, error) {
	if _, _, err := EmbeddingShape(len(ids), 1, vocabulary, 2); err != nil {
		return nil, err
	}
	offsets := make([]int, vocabulary+1)
	for _, id := range ids {
		if math.IsNaN(float64(id)) || math.IsInf(float64(id), 0) || id < 0 || id >= float32(vocabulary) || id != float32(int(id)) {
			return nil, fmt.Errorf("invalid training embedding ID")
		}
		offsets[int(id)+1]++
	}
	for i := 1; i < len(offsets); i++ {
		offsets[i] += offsets[i-1]
	}
	index := make([]float32, vocabulary+1+len(ids))
	for i, n := range offsets {
		index[i] = float32(n)
	}
	for row, id := range ids {
		slot := offsets[int(id)]
		index[vocabulary+1+slot] = float32(row)
		offsets[int(id)]++
	}
	return index, nil
}

// ValidateEmbeddingIndex admits the whole constant index before dispatch:
// monotonic extents, sequence-ordered unique rows, and exact coverage.
func ValidateEmbeddingIndex(index []float32, rows, vocabulary int) error {
	if _, _, err := EmbeddingShape(rows, 1, vocabulary, 2); err != nil {
		return err
	}
	if len(index) != vocabulary+1+rows {
		return fmt.Errorf("training embedding index shape")
	}
	integer := func(x float32, bound int) bool {
		return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) && x >= 0 && x <= float32(bound) && x == float32(int(x))
	}
	if index[0] != 0 || index[vocabulary] != float32(rows) {
		return fmt.Errorf("training embedding index extent")
	}
	seen := make([]bool, rows)
	for token := 0; token < vocabulary; token++ {
		lo, hi := index[token], index[token+1]
		if !integer(lo, rows) || !integer(hi, rows) || lo > hi {
			return fmt.Errorf("training embedding index offset")
		}
		previous := -1
		for at := int(lo); at < int(hi); at++ {
			row := index[vocabulary+1+at]
			if !integer(row, rows-1) {
				return fmt.Errorf("training embedding index row")
			}
			r := int(row)
			if r <= previous || seen[r] {
				return fmt.Errorf("training embedding index order/duplicate")
			}
			seen[r] = true
			previous = r
		}
	}
	// Monotonic offsets from zero through rows traverse exactly rows entries.
	// With every row in range and unique, coverage is complete.
	return nil
}
