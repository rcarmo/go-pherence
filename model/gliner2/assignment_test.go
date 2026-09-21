package gliner2

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func TestMinimumCostAssignmentBruteForceSmallMatrices(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	values := []float64{-3, -2, -1, 0, 1, 2, 3}
	for n := 1; n <= 4; n++ {
		for m := n; m <= 6; m++ {
			for sample := 0; sample < 256; sample++ {
				cost := make([][]float64, n)
				for i := 0; i < n; i++ {
					cost[i] = make([]float64, m)
					for j := 0; j < m; j++ {
						cost[i][j] = values[rng.Intn(len(values))]
					}
				}
				want := bruteForceMinimumCostAssignment(cost)
				got, err := minimumCostAssignment(cost)
				if err != nil {
					t.Fatalf("n=%d m=%d sample=%d err=%v", n, m, sample, err)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("n=%d m=%d sample=%d got=%v want=%v cost=%v", n, m, sample, got, want, cost)
				}
			}
		}
	}
}

func TestMinimumCostAssignmentDeterministicTies(t *testing.T) {
	for _, tc := range []struct {
		name string
		cost [][]float64
		want []int
	}{
		{name: "all-zero-rectangular", cost: [][]float64{{0, 0, 0}, {0, 0, 0}}, want: []int{0, 1}},
		{name: "cross-tie", cost: [][]float64{{0, -2}, {1, -1}}, want: []int{0, 1}},
		{name: "duplicate-columns", cost: [][]float64{{2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}}, want: []int{0, 1, 2}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for attempt := 0; attempt < 4; attempt++ {
				got, err := minimumCostAssignment(tc.cost)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("attempt=%d got=%v want=%v", attempt, got, tc.want)
				}
			}
		})
	}
}

func TestMinimumCostAssignmentZeroRows(t *testing.T) {
	for _, cost := range [][][]float64{nil, {}} {
		got, err := minimumCostAssignment(cost)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 0 {
			t.Fatalf("got=%v want empty", got)
		}
	}
}

func TestMinimumCostAssignmentRejectsMalformedInput(t *testing.T) {
	cases := []struct {
		name string
		cost [][]float64
		want string
	}{
		{name: "more-rows-than-columns", cost: [][]float64{{1}, {2}}, want: "rows <= columns"},
		{name: "ragged", cost: [][]float64{{1, 2}, {3}}, want: "row 1"},
		{name: "zero-columns", cost: [][]float64{{}}, want: "at least as many columns as rows"},
		{name: "nan", cost: [][]float64{{math.NaN()}}, want: "not finite"},
		{name: "inf", cost: [][]float64{{math.Inf(1)}}, want: "not finite"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := minimumCostAssignment(tc.cost)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got=%v err=%v want substring %q", got, err, tc.want)
			}
		})
	}
}

func bruteForceMinimumCostAssignment(cost [][]float64) []int {
	n := len(cost)
	if n == 0 {
		return []int{}
	}
	m := len(cost[0])
	used := make([]bool, m)
	current := make([]int, n)
	best := make([]int, n)
	bestCost := math.Inf(1)
	found := false
	const tol = 1e-12
	var search func(row int, total float64)
	search = func(row int, total float64) {
		if row == n {
			if !found || total < bestCost-tol || (math.Abs(total-bestCost) <= tol && lexicographicAssignmentLess(current, best)) {
				bestCost = total
				copy(best, current)
				found = true
			}
			return
		}
		for col := 0; col < m; col++ {
			if used[col] {
				continue
			}
			used[col] = true
			current[row] = col
			search(row+1, total+cost[row][col])
			used[col] = false
		}
	}
	search(0, 0)
	return best
}

func lexicographicAssignmentLess(a, b []int) bool {
	for i := range a {
		if a[i] < b[i] {
			return true
		}
		if a[i] > b[i] {
			return false
		}
	}
	return false
}
