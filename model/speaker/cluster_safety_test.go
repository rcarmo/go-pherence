package speaker

import "testing"

func TestAgglomerativeMergeRelabelsEveryMember(t *testing.T) {
	// 1+2 merge first, then that cluster joins0. The representative's old label
	// must be captured before its own array slot changes, or member2 is stranded.
	got := AgglomerativeCluster([][]float32{{0, 1}, {1, 0}, {1, 0}}, -.1)
	if len(got) != 3 || got[0] != got[1] || got[1] != got[2] {
		t.Fatal("merged cluster split", got)
	}
}
