package main

import (
	"math"
	"testing"
)

func TestDiagnosticLimitsBeforeExecution(t *testing.T) {
	check := func(s, d, c, r, p int, a, tol float64, b ...int) error {
		return validateWorkloadFlags(s, d, c, r, p, a, tol, b)
	}
	if err := check(1, 1, 32, 1, 0, .75, 1e-4, 2048, 10600, 256); err != nil {
		t.Fatal(err)
	}
	for _, f := range []func() error{
		func() error { return check(1<<30, 1, 32, 1, 0, .75, 0) }, func() error { return check(1, 1, 0, 1, 0, .75, 0) },
		func() error { return check(1, 1, 32, 0, 0, .75, 0) }, func() error { return check(1, 1, 32, 1, 0, math.NaN(), 0) },
		func() error { return check(1, 1, 32, 1, 0, .75, math.Inf(1)) }, func() error { return check(1, 1, 32, 1, 0, .75, 0, int(^uint(0)>>1)) },
		func() error { return check(1, 1, 32, 1, 0, .75, 0, -1) },
	} {
		if f() == nil {
			t.Fatal("malformed flags accepted")
		}
	}
}
