package diffusiongemma

import "testing"

func TestGGUFChunkedLMHeadStatsSub(t *testing.T) {
	base := ggufChunkedLMHeadStats{Calls: 1, Chunks: 2, Bytes: 3, PrepareNS: 4, UploadNS: 5, SgemmNS: 6, DownloadNS: 7, CopyNS: 8}
	now := ggufChunkedLMHeadStats{Calls: 3, Chunks: 7, Bytes: 13, PrepareNS: 24, UploadNS: 35, SgemmNS: 46, DownloadNS: 57, CopyNS: 68}
	d := now.Sub(base)
	if d.Calls != 2 || d.Chunks != 5 || d.Bytes != 10 || d.PrepareNS != 20 || d.UploadNS != 30 || d.SgemmNS != 40 || d.DownloadNS != 50 || d.CopyNS != 60 {
		t.Fatalf("unexpected diff: %+v", d)
	}
}
