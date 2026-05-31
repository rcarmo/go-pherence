package lfm2

// RuntimeStatus reports which LFM2 execution paths are implemented. It keeps
// inspector output honest while the package has contracts and request planning
// but intentionally no token generation yet.
type RuntimeStatus struct {
	CPUGeneration      bool     `json:"cpu_generation"`
	EmbeddingCPU       bool     `json:"embedding_cpu"`
	ConvCPU            bool     `json:"conv_cpu"`
	AttentionCPU       bool     `json:"attention_cpu"`
	MoECPU             bool     `json:"moe_cpu"`
	NVIDIA             bool     `json:"nvidia"`
	RuntimeImplemented bool     `json:"runtime_implemented"`
	Pending            []string `json:"pending,omitempty"`
}

func CurrentRuntimeStatus() RuntimeStatus {
	st := RuntimeStatus{}
	if !st.CPUGeneration {
		st.Pending = append(st.Pending, "cpu_generation_runtime")
	}
	if !st.NVIDIA {
		st.Pending = append(st.Pending, "nvidia_runtime")
	}
	st.RuntimeImplemented = st.CPUGeneration
	return st
}
