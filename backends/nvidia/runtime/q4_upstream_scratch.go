package nvidia

import "sync"

var q4UpstreamScratch struct {
	sync.Mutex
	q8 *Buffer
}

func q4UpstreamQ8(bytes int) (*Buffer, func(), error) {
	q4UpstreamScratch.Lock()
	if q4UpstreamScratch.q8 == nil || q4UpstreamScratch.q8.Size < bytes {
		if q4UpstreamScratch.q8 != nil {
			q4UpstreamScratch.q8.Free()
		}
		b, err := MallocBytes(bytes)
		if err != nil {
			q4UpstreamScratch.Unlock()
			return nil, nil, err
		}
		q4UpstreamScratch.q8 = b
	}
	return q4UpstreamScratch.q8, q4UpstreamScratch.Unlock, nil
}
func freeQ4UpstreamScratch() {
	q4UpstreamScratch.Lock()
	defer q4UpstreamScratch.Unlock()
	if q4UpstreamScratch.q8 != nil {
		q4UpstreamScratch.q8.Free()
		q4UpstreamScratch.q8 = nil
	}
}
