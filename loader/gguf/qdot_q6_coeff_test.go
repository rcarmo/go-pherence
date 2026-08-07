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
