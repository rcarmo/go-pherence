package qwen3tts

import "testing"

func TestCurrentRuntimeStatus(t *testing.T) {
	st := CurrentRuntimeStatus()
	if st.RuntimeImplemented || st.TalkerCPU || st.CodePredictorCPU || st.Decoder12HzCPU || st.NVIDIA || st.Streaming {
		t.Fatalf("unexpected implemented status: %+v", st)
	}
	want := []string{"cpu_talker_runtime", "cpu_code_predictor_runtime", "decoder12hz_runtime", "nvidia_runtime", "streaming_runtime"}
	if len(st.Pending) != len(want) {
		t.Fatalf("pending=%+v", st.Pending)
	}
	for i := range want {
		if st.Pending[i] != want[i] {
			t.Fatalf("pending=%+v", st.Pending)
		}
	}
}
