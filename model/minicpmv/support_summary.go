package minicpmv

type SupportSummary struct {
	SupportVersion      string       `json:"support_version"`
	RuntimeStatus       string       `json:"runtime_status"`
	RuntimeRoadmapPath  string       `json:"runtime_roadmap_path"`
	Capabilities        Capabilities `json:"capabilities"`
	PendingRuntimeSteps []string     `json:"pending_runtime_steps,omitempty"`
}

func CurrentSupportSummary() SupportSummary {
	caps := CurrentCapabilities()
	return SupportSummary{SupportVersion: SupportVersion, RuntimeStatus: RuntimeStatusPending, RuntimeRoadmapPath: RuntimeRoadmapPath, Capabilities: caps, PendingRuntimeSteps: caps.PendingRuntimeSteps}
}
