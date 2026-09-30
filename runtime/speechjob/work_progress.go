package speechjob

import "context"

// WorkProgress describes an in-memory, best-effort stage counter. It is not a
// checkpoint, never changes stage identity, and disappears on process restart.
// Completed counts durably acknowledged ASR windows or finished diarization
// windows; postprocessing is reported separately and has no percentage.
type WorkProgress struct {
	Stage     string `json:"stage"`
	Completed int64  `json:"completed"`
	Total     int64  `json:"total"`
	Phase     string `json:"phase,omitempty"`
}

type workProgressKey struct{}

// WithWorkProgress attaches an optional in-process observer to a run. The
// callback must be bounded, non-blocking and must not re-enter the Store.
func WithWorkProgress(ctx context.Context, report func(WorkProgress)) context.Context {
	return context.WithValue(ctx, workProgressKey{}, report)
}

func reportWorkProgress(ctx context.Context, stage string, completed, total int64, phase string) {
	if ctx == nil || completed < 0 || total < 1 || completed > total {
		return
	}
	if report, ok := ctx.Value(workProgressKey{}).(func(WorkProgress)); ok && report != nil {
		report(WorkProgress{Stage: stage, Completed: completed, Total: total, Phase: phase})
	}
}

// ReportASRProgress publishes bounded ASR sample/window progress without durable
// acknowledgement semantics. A transcript stage may replay from zero on retry.
func ReportASRProgress(ctx context.Context, completed, total int64, phase string) {
	reportWorkProgress(ctx, "asr-windows", completed, total, phase)
}

// ReportDiarizationProgress lets a model owner publish window completion through
// the caller's context, without coupling the model package to queue or HTTP.
func ReportDiarizationProgress(ctx context.Context, completed, total int64, phase string) {
	reportWorkProgress(ctx, "diarization", completed, total, phase)
}
