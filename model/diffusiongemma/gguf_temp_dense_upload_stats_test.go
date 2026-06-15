package diffusiongemma

import "testing"

func TestGGUFTempDenseUploadStatsSub(t *testing.T) {
	base := ggufTempDenseUploadStats{Calls: 2, Bytes: 100, TransposeNS: 10, UploadNS: 20, CacheHits: 1, ForwardAttnCalls: 2, ForwardAttnHits: 1}
	now := ggufTempDenseUploadStats{Calls: 5, Bytes: 700, TransposeNS: 110, UploadNS: 220, CacheHits: 4, CacheMisses: 3, ForwardAttnCalls: 7, ForwardAttnHits: 5, EncoderMLPCalls: 2, EncoderMLPHits: 1}
	d := now.Sub(base)
	if d.Calls != 3 || d.Bytes != 600 || d.TransposeNS != 100 || d.UploadNS != 200 || d.CacheHits != 3 || d.CacheMisses != 3 || d.ForwardAttnCalls != 5 || d.ForwardAttnHits != 4 || d.EncoderMLPCalls != 2 || d.EncoderMLPHits != 1 {
		t.Fatalf("unexpected diff: %+v", d)
	}
}
