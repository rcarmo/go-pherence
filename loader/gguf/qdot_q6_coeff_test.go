package gguf

import "testing"

var q6CoeffBenchSink int32

func TestQ6KCoeffDotMatchesScalar(t *testing.T) {
	var q8 [256]int8
	var coeff [256]int16
	var want int32
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
		want += int32(q8[i]) * int32(coeff[i])
	}
	if got := q6KCoeffDot(&q8, &coeff); got != want {
		t.Fatalf("q6KCoeffDot=%d want %d", got, want)
	}
}

func BenchmarkQ6KCoeffDot(b *testing.B) {
	var q8 [256]int8
	var coeff [256]int16
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
	}
	b.ResetTimer()
	var sum int32
	for i := 0; i < b.N; i++ {
		sum += q6KCoeffDot(&q8, &coeff)
	}
	q6CoeffBenchSink += sum
}

func BenchmarkQ6KCoeffDotScalar(b *testing.B) {
	var q8 [256]int8
	var coeff [256]int16
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
	}
	b.ResetTimer()
	var sum int32
	for i := 0; i < b.N; i++ {
		for lane := range q8 {
			sum += int32(q8[lane]) * int32(coeff[lane])
		}
	}
	q6CoeffBenchSink += sum
}

func TestQ6KCoeffDot8MatchesScalarPartitions(t *testing.T) {
	var q8 [256]int8
	var coeff [256]int16
	var want, got [8]int32
	for i := range q8 {
		q8[i] = int8((i*17)%127 - 63)
		coeff[i] = int16((i*29)%3969 - 1984)
		want[(i/2)%8] += int32(q8[i]) * int32(coeff[i])
	}
	q6KCoeffDot8(&q8, &coeff, &got)
	if got != want {
		t.Fatalf("q6KCoeffDot8=%v want %v", got, want)
	}
}

func TestDotQ6KQ8KGemvFastBoundedDifference(t *testing.T) {
	raw, y := syntheticQ6KQ8KDotInputs(2560)
	got := dotQ6KQ8KGemvFast(raw, y, 10)
	want, err := DotQ6KQ8K(raw, y, 2560)
	if err != nil {
		t.Fatal(err)
	}
	const maxDifference = float32(0.00012207031)
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	if diff > maxDifference {
		t.Fatalf("fast dot=%g exact=%g difference=%g > %g", got, want, diff, maxDifference)
	}
}
