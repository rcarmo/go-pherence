package main

import "sync/atomic"

type aiBarrier struct {
	count atomic.Int64
	phase atomic.Int64
	n     int64
}

func newAIBarrier(n int) *aiBarrier { return &aiBarrier{n: int64(n)} }

func (b *aiBarrier) wait() {
	p := b.phase.Load()
	if b.count.Add(1) == b.n {
		b.count.Store(0)
		b.phase.Add(1)
		return
	}
	for b.phase.Load() == p {
	}
}
