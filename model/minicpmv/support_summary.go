package minicpmv

type SupportSummary struct {
	SupportVersion      string       `json:"support_version"`
	RuntimeStatus       string       `json:"runtime_status"`
	Capabilities        Capabilities `json:"capabilities"`
	PendingRuntimeSteps []string     `json:"pending_runtime_steps,omitempty"`
}

func CurrentSupportSummary() SupportSummary {
	caps := CurrentCapabilities()
	return SupportSummary{SupportVersion: SupportVersion, RuntimeStatus: RuntimeStatusPending, Capabilities: caps, PendingRuntimeSteps: caps.PendingRuntimeSteps}
}
