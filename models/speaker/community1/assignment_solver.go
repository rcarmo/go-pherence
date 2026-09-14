// Copyright (c) 2026 Rui Carmo (Go adaptation)
// SPDX-License-Identifier: BSD-3-Clause
// Adapted from SciPy rectangular_lsap.cpp by PM Larsen (see NOTICE).
package community1

import (
	"context"
	"fmt"
	"math"
)

// Fixed-size workspace is reused across windows. Rows are local speakers <=8,
// columns <=64; tall matrices transpose, so augmented rows are always <=8.
// Original SciPy arithmetic and reverse-candidate/free-sink tie rules retained.
type assignmentWorkspace struct {
	cost                       [512]float64
	u                          [8]float64
	v, shortest                [64]float64
	path, rowForCol, remaining [64]int
	colForRow                  [8]int
	seenRows                   [8]bool
	seenCols                   [64]bool
}

func (w *assignmentWorkspace) solve(ctx context.Context, input []float64, rows, columns int, nanFill float64, labels []int) error {
	nr, nc := rows, columns
	transpose := nc < nr
	if transpose {
		nr, nc = nc, nr
	}
	for i := 0; i < nr; i++ {
		for j := 0; j < nc; j++ {
			value := float64(0)
			if transpose {
				value = input[j*columns+i]
			} else {
				value = input[i*columns+j]
			}
			if math.IsNaN(value) {
				value = nanFill
			}
			w.cost[i*nc+j] = -value
		}
	}
	clear(w.u[:])
	clear(w.v[:])
	for i := 0; i < nr; i++ {
		w.colForRow[i] = -1
	}
	for j := 0; j < nc; j++ {
		w.rowForCol[j] = -1
		w.path[j] = -1
	}
	for current := 0; current < nr; current++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for j := 0; j < nc; j++ {
			w.remaining[j] = nc - j - 1
			w.shortest[j] = math.Inf(1)
		}
		clear(w.seenRows[:])
		clear(w.seenCols[:])
		nRemaining := nc
		sink := -1
		row := current
		minimum := float64(0)
		for sink < 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
			index := -1
			lowest := math.Inf(1)
			w.seenRows[row] = true
			for it := 0; it < nRemaining; it++ {
				col := w.remaining[it]
				candidate := minimum + w.cost[row*nc+col] - w.u[row] - w.v[col]
				if candidate < w.shortest[col] {
					w.path[col] = row
					w.shortest[col] = candidate
				}
				if w.shortest[col] < lowest || (w.shortest[col] == lowest && w.rowForCol[col] == -1) {
					lowest = w.shortest[col]
					index = it
				}
			}
			minimum = lowest
			if index < 0 || math.IsInf(minimum, 0) || math.IsNaN(minimum) {
				return fmt.Errorf("infeasible/nonfinite assignment")
			}
			col := w.remaining[index]
			if w.rowForCol[col] == -1 {
				sink = col
			} else {
				row = w.rowForCol[col]
			}
			w.seenCols[col] = true
			nRemaining--
			w.remaining[index] = w.remaining[nRemaining]
		}
		w.u[current] += minimum
		for i := 0; i < nr; i++ {
			if w.seenRows[i] && i != current {
				w.u[i] += minimum - w.shortest[w.colForRow[i]]
			}
		}
		for j := 0; j < nc; j++ {
			if w.seenCols[j] {
				w.v[j] -= minimum - w.shortest[j]
			}
		}
		col := sink
		for {
			row := w.path[col]
			w.rowForCol[col] = row
			previous := w.colForRow[row]
			w.colForRow[row] = col
			col = previous
			if row == current {
				break
			}
		}
	}
	for row := 0; row < nr; row++ {
		if transpose {
			labels[w.colForRow[row]] = row
		} else {
			labels[row] = w.colForRow[row]
		}
	}
	return ctx.Err()
}
