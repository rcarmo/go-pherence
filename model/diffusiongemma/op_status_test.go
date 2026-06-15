package diffusiongemma

import "testing"

func TestOperationStatusesReferenceCompleteForTextGGUF(t *testing.T) {
	implemented, referenceComplete, total := OperationStatusSummaryForDomain("text")
	if total == 0 {
		t.Fatal("no text operation statuses")
	}
	if implemented != total || referenceComplete != total {
		t.Fatalf("text operation status summary implemented=%d reference=%d total=%d", implemented, referenceComplete, total)
	}
	for _, op := range OperationStatuses() {
		if op.Domain != "text" {
			continue
		}
		if !op.Implemented || !op.ReferenceComplete {
			t.Fatalf("text operation %s not complete: %+v", op.Kind, op)
		}
	}
}

func TestOperationDomainSummaries(t *testing.T) {
	ops := OperationStatuses()
	impl, ref, total := OperationStatusSummaryFromStatuses(ops)
	if impl != total || ref >= total {
		t.Fatalf("summary impl=%d ref=%d total=%d", impl, ref, total)
	}
	summaries := OperationDomainSummaries(ops)
	text := summaries["text"]
	if text.Total == 0 || text.Implemented != text.Total || text.ReferenceComplete != text.Total {
		t.Fatalf("text domain summary=%+v", text)
	}
	vision := summaries["vision"]
	if vision.Total == 0 || vision.Implemented != vision.Total || vision.ReferenceComplete >= vision.Total {
		t.Fatalf("vision domain summary=%+v", vision)
	}
}

func TestOperationStatusSummaryForDomainMatchesDomainSummaries(t *testing.T) {
	domains := OperationDomainSummaries(OperationStatuses())
	for _, domain := range []string{"text", "vision"} {
		impl, ref, total := OperationStatusSummaryForDomain(domain)
		s := domains[domain]
		if impl != s.Implemented || ref != s.ReferenceComplete || total != s.Total {
			t.Fatalf("domain %s summary=(%d,%d,%d) want %+v", domain, impl, ref, total, s)
		}
	}
}

func TestOperationStatusesReportPartialVisionBoundary(t *testing.T) {
	implemented, referenceComplete, total := OperationStatusSummaryForDomain("vision")
	if total == 0 {
		t.Fatal("no vision operation statuses")
	}
	if implemented != total {
		t.Fatalf("vision implementation boundary should now be complete: implemented=%d total=%d", implemented, total)
	}
	if referenceComplete >= total {
		t.Fatalf("vision should not report full reference completion yet: reference=%d total=%d", referenceComplete, total)
	}
	seen := map[OpKind]OpStatus{}
	for _, op := range OperationStatuses() {
		if op.Domain == "vision" {
			seen[op.Kind] = op
		}
	}
	for _, want := range []OpKind{OpImagePreprocess, OpImageSoftTokenPrompt, OpVisionPatchEmbedding, OpVisionEmbeddingInsert, OpVisionEncoderTower} {
		if _, ok := seen[want]; !ok {
			t.Fatalf("missing vision op status %s", want)
		}
	}
	if !seen[OpVisionPatchEmbedding].Implemented || seen[OpVisionPatchEmbedding].ReferenceComplete {
		t.Fatalf("patch embedding boundary should be implemented but not reference-complete: %+v", seen[OpVisionPatchEmbedding])
	}
	if !seen[OpVisionEncoderTower].Implemented || seen[OpVisionEncoderTower].ReferenceComplete {
		t.Fatalf("vision tower should be implemented but not reference-complete: %+v", seen[OpVisionEncoderTower])
	}
}
