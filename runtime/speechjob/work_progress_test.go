package speechjob

import (
	"context"
	"testing"
)

func TestWorkProgressContextBoundsAndDiarization(t *testing.T) {
	var got []WorkProgress
	ctx := WithWorkProgress(context.Background(), func(p WorkProgress) { got = append(got, p) })
	reportWorkProgress(ctx, "asr-windows", 10, 42, "windows")
	reportWorkProgress(ctx, "asr-windows", -1, 42, "windows")
	reportWorkProgress(ctx, "asr-windows", 43, 42, "windows")
	reportWorkProgress(ctx, "asr-windows", 0, 0, "windows")
	ReportDiarizationProgress(ctx, 42, 42, "postprocess")
	if len(got) != 2 || got[0] != (WorkProgress{Stage: "asr-windows", Completed: 10, Total: 42, Phase: "windows"}) || got[1] != (WorkProgress{Stage: "diarization", Completed: 42, Total: 42, Phase: "postprocess"}) {
		t.Fatal(got)
	}
	reportWorkProgress(context.Background(), "asr-windows", 1, 2, "windows")
}
