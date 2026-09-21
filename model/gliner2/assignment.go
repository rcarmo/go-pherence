package gliner2

import (
	"fmt"
	"math"
	"math/big"
)

// minimumCostAssignment returns the lexicographically smallest minimum-cost
// injective row-to-column assignment for a finite rectangular cost matrix with
// rows <= columns.
func minimumCostAssignment(cost [][]float64) ([]int, error) {
	n := len(cost)
	if n == 0 {
		return []int{}, nil
	}
	m, err := validateAssignmentCost(cost)
	if err != nil {
		return nil, err
	}

	weights := assignmentLexWeights(n, m)
	u := make([]assignmentCost, n+1)
	v := make([]assignmentCost, m+1)
	p := make([]int, m+1)
	way := make([]int, m+1)
	minv := make([]assignmentCost, m+1)
	used := make([]bool, m+1)
	var delta, cur assignmentCost

	for i := 1; i <= n; i++ {
		p[0] = i
		for j := 1; j <= m; j++ {
			minv[j].setInf()
			used[j] = false
		}
		used[0] = false
		j0 := 0
		for {
			used[j0] = true
			i0 := p[j0]
			delta.setInf()
			j1 := 0
			for j := 1; j <= m; j++ {
				if used[j] {
					continue
				}
				cur.setReduced(cost[i0-1][j-1], &weights[i0-1][j-1], &u[i0], &v[j])
				if cur.cmp(&minv[j]) < 0 {
					minv[j].set(&cur)
					way[j] = j0
				}
				if j1 == 0 || minv[j].cmp(&delta) < 0 || (minv[j].cmp(&delta) == 0 && j < j1) {
					delta.set(&minv[j])
					j1 = j
				}
			}
			if j1 == 0 || delta.inf {
				return nil, fmt.Errorf("assignment search failed")
			}
			for j := 0; j <= m; j++ {
				if used[j] {
					u[p[j]].add(&delta)
					v[j].sub(&delta)
					continue
				}
				minv[j].sub(&delta)
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	assignment := make([]int, n)
	for i := range assignment {
		assignment[i] = -1
	}
	for j := 1; j <= m; j++ {
		if p[j] != 0 {
			assignment[p[j]-1] = j - 1
		}
	}
	for i, col := range assignment {
		if col < 0 {
			return nil, fmt.Errorf("row %d was left unassigned", i)
		}
	}
	return assignment, nil
}

func validateAssignmentCost(cost [][]float64) (int, error) {
	m := len(cost[0])
	if m == 0 {
		return 0, fmt.Errorf("assignment requires at least as many columns as rows")
	}
	if len(cost) > m {
		return 0, fmt.Errorf("assignment requires rows <= columns: %d > %d", len(cost), m)
	}
	for i := range cost {
		if len(cost[i]) != m {
			return 0, fmt.Errorf("assignment row %d has %d columns want %d", i, len(cost[i]), m)
		}
		for j, value := range cost[i] {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return 0, fmt.Errorf("assignment cost[%d][%d] is not finite", i, j)
			}
		}
	}
	return m, nil
}

func assignmentLexWeights(rows, cols int) [][]big.Int {
	weights := make([][]big.Int, rows)
	if rows == 0 || cols == 0 {
		return weights
	}
	base := big.NewInt(int64(cols + 1))
	factors := make([]big.Int, rows)
	factors[rows-1].SetInt64(1)
	for i := rows - 2; i >= 0; i-- {
		factors[i].Mul(&factors[i+1], base)
	}
	var scale big.Int
	for i := 0; i < rows; i++ {
		weights[i] = make([]big.Int, cols)
		for j := 1; j < cols; j++ {
			scale.SetInt64(int64(j))
			weights[i][j].Mul(&factors[i], &scale)
		}
	}
	return weights
}

type assignmentCost struct {
	primary   float64
	secondary big.Int
	inf       bool
}

func (c *assignmentCost) set(other *assignmentCost) {
	c.primary = other.primary
	c.secondary.Set(&other.secondary)
	c.inf = other.inf
}

func (c *assignmentCost) setInf() {
	c.primary = math.Inf(1)
	c.secondary.SetInt64(0)
	c.inf = true
}

func (c *assignmentCost) setReduced(primary float64, weight *big.Int, row, col *assignmentCost) {
	c.primary = primary - row.primary - col.primary
	c.secondary.Sub(weight, &row.secondary)
	c.secondary.Sub(&c.secondary, &col.secondary)
	c.inf = false
}

func (c *assignmentCost) add(other *assignmentCost) {
	c.primary += other.primary
	c.secondary.Add(&c.secondary, &other.secondary)
}

func (c *assignmentCost) sub(other *assignmentCost) {
	if c.inf {
		return
	}
	c.primary -= other.primary
	c.secondary.Sub(&c.secondary, &other.secondary)
	c.inf = false
}

func (c *assignmentCost) cmp(other *assignmentCost) int {
	if c.inf {
		if other.inf {
			return 0
		}
		return 1
	}
	if other.inf {
		return -1
	}
	if c.primary < other.primary {
		return -1
	}
	if c.primary > other.primary {
		return 1
	}
	return c.secondary.Cmp(&other.secondary)
}
