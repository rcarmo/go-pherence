package lfm2

import "testing"

func TestCurrentRuntimeStatus(t *testing.T) {
	st := CurrentRuntimeStatus()
	if st.RuntimeImplemented || st.CPUGeneration || st.EmbeddingCPU || st.ConvCPU || st.AttentionCPU || st.MoECPU || st.NVIDIA {
		t.Fatalf("unexpected implemented status: %+v", st)
	}
	want := []string{"cpu_generation_runtime", "nvidia_runtime"}
	if len(st.Pending) != len(want) {
		t.Fatalf("pending=%+v", st.Pending)
	}
	for i := range want {
		if st.Pending[i] != want[i] {
			t.Fatalf("pending=%+v", st.Pending)
		}
	}
}
