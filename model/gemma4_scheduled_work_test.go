package model

import (
	"context"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/inferencesched"
)

type scheduledTestRequest struct {
	id  string
	seq uint64
}

func (r scheduledTestRequest) ID() string         { return r.id }
func (r scheduledTestRequest) ArrivalSeq() uint64 { return r.seq }

func newGemma4ZeroLayerDecodeSessionTestModel() *LlamaModel {
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	return m
}

func newGemma4ScheduledWorkForTest(t *testing.T, m *LlamaModel, prompt []int, maxTokens int) *Gemma4ScheduledWork {
	t.Helper()
	work, err := NewGemma4ScheduledWork(newGemma4DecodeSessionForTest(t, m, maxTokens, Gemma4PromptCacheConfig{}), prompt)
	if err != nil {
		t.Fatalf("NewGemma4ScheduledWork: %v", err)
	}
	return work
}

func assertDecodeResultsEqual(t *testing.T, got, want []DecodeResult) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("decode results=%d want %d", len(got), len(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		if g.Token != w.Token || g.Position != w.Position || g.Generated != w.Generated || g.Finished != w.Finished || g.FinishReason != w.FinishReason {
			t.Fatalf("decode result %d=%+v want %+v", i, g, w)
		}
		if !sameFloat32s(g.Logits, w.Logits) {
			t.Fatalf("decode logits[%d]=%v want %v", i, g.Logits, w.Logits)
		}
	}
}

func findScheduledResult(results []inferencesched.StepResult, id string) inferencesched.StepResult {
	for _, res := range results {
		if res.RequestID == id {
			return res
		}
	}
	return inferencesched.StepResult{}
}

func progressingRequestID(results []inferencesched.StepResult) string {
	for _, res := range results {
		if res.TokensConsumed > 0 || res.TokensEmitted > 0 {
			return res.RequestID
		}
	}
	return ""
}

func decodeTokens(results []DecodeResult) []int {
	out := make([]int, 0, len(results))
	for _, res := range results {
		out = append(out, res.Token)
	}
	return out
}

func TestGemma4ScheduledWorkDecodeFirstOrdering(t *testing.T) {
	m := newGemma4ZeroLayerDecodeSessionTestModel()
	decodePrompt := []int{0}
	prefillPrompt := []int{0, 1, 2}
	decodeWork := newGemma4ScheduledWorkForTest(t, m, decodePrompt, 1)
	prefillWork := newGemma4ScheduledWorkForTest(t, m, prefillPrompt, 1)

	s, err := inferencesched.New(inferencesched.Config{MaxActive: 2, DecodeBudget: 1, PrefillBudget: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Add(context.Background(), scheduledTestRequest{id: "decode", seq: 1}, decodeWork); err != nil {
		t.Fatal(err)
	}
	if err := s.Add(context.Background(), scheduledTestRequest{id: "prefill", seq: 2}, prefillWork); err != nil {
		t.Fatal(err)
	}

	report1, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report1.Stats.DecodeUsed != 0 || report1.Stats.PrefillUsed != 1 {
		t.Fatalf("step1 stats=%+v", report1.Stats)
	}
	if got := decodeWork.RemainingPrefill(); got != 0 {
		t.Fatalf("decode work remaining prefill=%d want 0", got)
	}
	if got, want := prefillWork.RemainingPrefill(), len(m.PreparedGenerateTokens(prefillPrompt)); got != want {
		t.Fatalf("prefill work remaining=%d want %d", got, want)
	}

	report2, err := s.Step(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if report2.Stats.DecodeUsed != 1 || report2.Stats.PrefillUsed != 1 {
		t.Fatalf("step2 stats=%+v", report2.Stats)
	}
	if got, want := []string{report2.Results[0].RequestID, report2.Results[1].RequestID}, []string{"decode", "prefill"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("step2 result order=%v want %v", got, want)
	}
	if got := decodeWork.DrainResults(); len(got) != 1 || !got[0].Finished {
		t.Fatalf("decode drain=%+v", got)
	}
	if got, want := prefillWork.RemainingPrefill(), len(m.PreparedGenerateTokens(prefillPrompt))-1; got != want {
		t.Fatalf("prefill work remaining=%d want %d", got, want)
	}
}

func TestGemma4ScheduledWorkBoundedPrefill222Tail(t *testing.T) {
	m := newGemma4ZeroLayerDecodeSessionTestModel()
	prompt := []int{0, 1, 2, 1, 0}
	prepared := m.PreparedGenerateTokens(prompt)
	if len(prepared) != 5 {
		t.Fatalf("prepared prompt len=%d want 5", len(prepared))
	}
	work := newGemma4ScheduledWorkForTest(t, m, prompt, 1)
	if got := work.RemainingPrefill(); got != len(prepared) {
		t.Fatalf("initial remaining=%d want %d", got, len(prepared))
	}

	s, err := inferencesched.New(inferencesched.Config{MaxActive: 1, DecodeBudget: 0, PrefillBudget: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Add(context.Background(), scheduledTestRequest{id: "req", seq: 1}, work); err != nil {
		t.Fatal(err)
	}

	wantConsumed := []int{2, 2, 1}
	wantRemaining := []int{3, 1, 0}
	for i := range wantConsumed {
		report, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		res := findScheduledResult(report.Results, "req")
		if res.TokensConsumed != wantConsumed[i] {
			t.Fatalf("step %d consumed=%d want %d", i+1, res.TokensConsumed, wantConsumed[i])
		}
		if got := work.RemainingPrefill(); got != wantRemaining[i] {
			t.Fatalf("step %d remaining=%d want %d", i+1, got, wantRemaining[i])
		}
	}
}

func TestGemma4ScheduledWorkExactOutputMatchesOneShot(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	prompt := []int{0, 1, 2, 1, 0}
	prepared := m.PreparedGenerateTokens(prompt)
	want := runGemma4SessionOneShotTrace(t, m, prompt, 3, Gemma4PromptCacheConfig{})
	work := newGemma4ScheduledWorkForTest(t, m, prompt, 3)

	s, err := inferencesched.New(inferencesched.Config{MaxActive: 1, DecodeBudget: 2, PrefillBudget: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Add(context.Background(), scheduledTestRequest{id: "req", seq: 1}, work); err != nil {
		t.Fatal(err)
	}

	var gotDecode []DecodeResult
	for step := 0; step < 8; step++ {
		report, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		gotDecode = append(gotDecode, work.DrainResults()...)
		if report.Stats.Running == 0 && report.Stats.Waiting == 0 {
			break
		}
	}
	assertDecodeResultsEqual(t, gotDecode, want.decode)
	gotOutput := append(append([]int(nil), prepared...), decodeTokens(gotDecode)...)
	if !sameInts(gotOutput, want.output) {
		t.Fatalf("output=%v want %v", gotOutput, want.output)
	}
	if !work.closed {
		t.Fatal("work was not closed after finishing")
	}
}

func TestGemma4ScheduledWorkCancellationAndClose(t *testing.T) {
	t.Run("scheduler_cancel", func(t *testing.T) {
		m := newGemma4ZeroLayerDecodeSessionTestModel()
		work := newGemma4ScheduledWorkForTest(t, m, []int{0, 1, 2, 1, 0}, 2)
		s, err := inferencesched.New(inferencesched.Config{MaxActive: 1, DecodeBudget: 0, PrefillBudget: 1})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := s.Add(ctx, scheduledTestRequest{id: "req", seq: 1}, work); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		cancel()
		report, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		res := findScheduledResult(report.Results, "req")
		if !res.Canceled || res.Err != nil {
			t.Fatalf("cancel result=%+v", res)
		}
		if !work.canceled || !work.closed {
			t.Fatalf("work state canceled=%v closed=%v", work.canceled, work.closed)
		}
		if finished, reason := work.session.Finished(); !finished || reason != FinishReasonClosed {
			t.Fatalf("session finished=%v reason=%q", finished, reason)
		}
		if err := work.Close(); err != nil {
			t.Fatalf("idempotent close: %v", err)
		}
	})

	t.Run("close_idempotent", func(t *testing.T) {
		m := newGemma4ZeroLayerDecodeSessionTestModel()
		work := newGemma4ScheduledWorkForTest(t, m, []int{0, 1}, 1)
		if err := work.Close(); err != nil {
			t.Fatalf("first close: %v", err)
		}
		if err := work.Close(); err != nil {
			t.Fatalf("second close: %v", err)
		}
		if finished, reason := work.session.Finished(); !finished || reason != FinishReasonClosed {
			t.Fatalf("session finished=%v reason=%q", finished, reason)
		}
	})
}

func TestGemma4ScheduledWorkMaxActiveAndFairness(t *testing.T) {
	t.Run("max_active", func(t *testing.T) {
		m := newGemma4ZeroLayerDecodeSessionTestModel()
		w1 := newGemma4ScheduledWorkForTest(t, m, []int{0}, 1)
		w2 := newGemma4ScheduledWorkForTest(t, m, []int{1}, 1)
		w3 := newGemma4ScheduledWorkForTest(t, m, []int{2}, 1)
		s, err := inferencesched.New(inferencesched.Config{MaxActive: 2, DecodeBudget: 2, PrefillBudget: 2})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		for i, work := range []*Gemma4ScheduledWork{w1, w2, w3} {
			if err := s.Add(context.Background(), scheduledTestRequest{id: string(rune('a' + i)), seq: uint64(i + 1)}, work); err != nil {
				t.Fatal(err)
			}
		}

		report1, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if report1.Stats.Admitted != 2 || report1.Stats.Running != 2 || report1.Stats.Waiting != 1 {
			t.Fatalf("step1 stats=%+v", report1.Stats)
		}

		report2, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if report2.Stats.Finished != 2 || report2.Stats.Running != 0 || report2.Stats.Waiting != 1 {
			t.Fatalf("step2 stats=%+v", report2.Stats)
		}
		if got := len(w1.DrainResults()) + len(w2.DrainResults()); got != 2 {
			t.Fatalf("finished decode results=%d want 2", got)
		}

		report3, err := s.Step(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if report3.Stats.Admitted != 1 || report3.Stats.Waiting != 0 || report3.Stats.Running != 1 {
			t.Fatalf("step3 stats=%+v", report3.Stats)
		}
	})

	t.Run("fairness", func(t *testing.T) {
		m := newGemma4ZeroLayerDecodeSessionTestModel()
		works := []*Gemma4ScheduledWork{
			newGemma4ScheduledWorkForTest(t, m, []int{0, 1}, 1),
			newGemma4ScheduledWorkForTest(t, m, []int{1, 2}, 1),
			newGemma4ScheduledWorkForTest(t, m, []int{2, 0}, 1),
		}
		s, err := inferencesched.New(inferencesched.Config{MaxActive: 3, DecodeBudget: 0, PrefillBudget: 1})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()

		for i, work := range works {
			if err := s.Add(context.Background(), scheduledTestRequest{id: string(rune('a' + i)), seq: uint64(i + 1)}, work); err != nil {
				t.Fatal(err)
			}
		}

		var got []string
		for i := 0; i < 4; i++ {
			report, err := s.Step(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, progressingRequestID(report.Results))
		}
		if want := []string{"a", "b", "c", "a"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("prefill order=%v want %v", got, want)
		}
	})
}
