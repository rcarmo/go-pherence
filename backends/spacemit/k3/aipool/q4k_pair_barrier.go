package aipool

import "sync/atomic"

type Q4KPairBarrier struct {
	count [4]atomic.Int64
	phase [4]atomic.Int64
}

func (b *Q4KPairBarrier) Wait(pair int) {
	if pair < 0 || pair >= len(b.count) {
		return
	}
	p := b.phase[pair].Load()
	if b.count[pair].Add(1) == 2 {
		b.count[pair].Store(0)
		b.phase[pair].Add(1)
		return
	}
	for b.phase[pair].Load() == p {
		// Native uses a tight per-pair barrier around short TCM copy/compute waves.
		// Do not yield here: scheduler handoff latency dwarfs the B/N32 tile work.
	}
}
