package nvidia

import "sync"

var q8ProjectionScratch struct {
	sync.Mutex
	q, d, sums *Buffer
}

func q8ProjectionBuffers(qBytes, dElems, sumBytes int) (q, d, sums *Buffer, unlock func(), err error) {
	q8ProjectionScratch.Lock()
	unlock = q8ProjectionScratch.Unlock
	fail := func(e error) (*Buffer, *Buffer, *Buffer, func(), error) {
		unlock()
		return nil, nil, nil, nil, e
	}
	if q8ProjectionScratch.q == nil || q8ProjectionScratch.q.Size < qBytes {
		if q8ProjectionScratch.q != nil {
			q8ProjectionScratch.q.Free()
		}
		q8ProjectionScratch.q, err = MallocBytes(qBytes)
		if err != nil {
			return fail(err)
		}
	}
	if q8ProjectionScratch.d == nil || q8ProjectionScratch.d.Size < dElems*4 {
		if q8ProjectionScratch.d != nil {
			q8ProjectionScratch.d.Free()
		}
		q8ProjectionScratch.d, err = Malloc(dElems)
		if err != nil {
			return fail(err)
		}
	}
	if sumBytes > 0 && (q8ProjectionScratch.sums == nil || q8ProjectionScratch.sums.Size < sumBytes) {
		if q8ProjectionScratch.sums != nil {
			q8ProjectionScratch.sums.Free()
		}
		q8ProjectionScratch.sums, err = MallocBytes(sumBytes)
		if err != nil {
			return fail(err)
		}
	}
	return q8ProjectionScratch.q, q8ProjectionScratch.d, q8ProjectionScratch.sums, unlock, nil
}

func freeQ8ProjectionScratch() {
	q8ProjectionScratch.Lock()
	defer q8ProjectionScratch.Unlock()
	for _, b := range []*Buffer{q8ProjectionScratch.q, q8ProjectionScratch.d, q8ProjectionScratch.sums} {
		if b != nil {
			b.Free()
		}
	}
	q8ProjectionScratch.q, q8ProjectionScratch.d, q8ProjectionScratch.sums = nil, nil, nil
}
