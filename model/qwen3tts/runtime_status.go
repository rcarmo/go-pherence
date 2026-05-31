package qwen3tts

// RuntimeStatus reports which execution stages are implemented. It keeps the
// inspector/model-coverage story explicit while the package has validated
// contracts but intentionally no generation yet.
type RuntimeStatus struct {
	TalkerCPU          bool     `json:"talker_cpu"`
	CodePredictorCPU   bool     `json:"code_predictor_cpu"`
	Decoder12HzCPU     bool     `json:"decoder12hz_cpu"`
	NVIDIA             bool     `json:"nvidia"`
	Streaming          bool     `json:"streaming"`
	RuntimeImplemented bool     `json:"runtime_implemented"`
	Pending            []string `json:"pending,omitempty"`
}

func CurrentRuntimeStatus() RuntimeStatus {
	st := RuntimeStatus{}
	if !st.TalkerCPU {
		st.Pending = append(st.Pending, "cpu_talker_runtime")
	}
	if !st.CodePredictorCPU {
		st.Pending = append(st.Pending, "cpu_code_predictor_runtime")
	}
	if !st.Decoder12HzCPU {
		st.Pending = append(st.Pending, "decoder12hz_runtime")
	}
	if !st.NVIDIA {
		st.Pending = append(st.Pending, "nvidia_runtime")
	}
	if !st.Streaming {
		st.Pending = append(st.Pending, "streaming_runtime")
	}
	st.RuntimeImplemented = st.TalkerCPU && st.CodePredictorCPU && st.Decoder12HzCPU
	return st
}
