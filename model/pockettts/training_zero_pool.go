package pockettts

// trainingZeroPool is local to one privately constructed resident graph. Its
// arrays are immutable upload sources, never execution destinations or caller
// seeds. Sharing does not merge tensor IDs or change logical/native allocation.
// Public builders keep separate owned arrays. No global or cross-graph cache.
type trainingZeroPool map[int][]float32

func (p *trainingZeroPool) values(n int) []float32 {
	if *p == nil {
		*p = make(trainingZeroPool)
	}
	if x, ok := (*p)[n]; ok {
		return x
	}
	x := make([]float32, n)
	(*p)[n] = x
	return x
}
