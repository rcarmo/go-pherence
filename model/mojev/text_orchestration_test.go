package mojev

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model/qwen"
)

func TestTextOrchestration(t *testing.T) {
	req := decodeTextOrchestrationRequest(t, `{"model":"m","state":"alpha beta","questions":{"route":{"type":"choice","instructions":"Pick a route.","criteria":{"z":"last","a":"first"}},"risk":{"type":"score","instructions":"Rate the risk.","criteria":["low","high"]}}}`)
	tok := syntheticASCIITokenizer(req)
	tokenizeSegments := 1 + len(req.Fields)
	for _, field := range req.Fields {
		tokenizeSegments += len(field.SortedOptions)
	}
	cancelAtPreScore := int32(2*tokenizeSegments + 2)
	cancelAtPostScore := cancelAtPreScore + 1
	cancelAtPostAssemble := cancelAtPreScore + 2
	newPollCancelContext := func(t *testing.T, polls int32) *pollCancelContext {
		t.Helper()
		ctx, cancel := context.WithCancel(context.Background())
		t.Cleanup(cancel)
		probe := &pollCancelContext{Context: ctx, cancel: cancel}
		probe.remaining.Store(polls)
		return probe
	}

	t.Run("valid_callback_result", func(t *testing.T) {
		calls := 0
		got, err := scoreTextContextWith(context.Background(), req, tok, 128, 128, func(row EncodedRow) ([][]float32, error) {
			calls++
			assertEncodedRowMatchesRequest(t, tok, req, row)
			return [][]float32{{-4, 4}, {6, -6}}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if calls != 1 {
			t.Fatalf("callback calls=%d want 1", calls)
		}
		if got == nil || got.Model != req.Model {
			t.Fatalf("bad decision: %+v", got)
		}
		route := got.Answers["route"]
		if route["type"] != "choice" || route["choice"] != "z" || route["confidence"].(float64) <= 0.99 {
			t.Fatalf("bad route answer: %+v", route)
		}
		risk := got.Answers["risk"]
		if risk["type"] != "score" || risk["score"].(float64) <= 0.99 || risk["confidence"].(float64) <= 0.99 {
			t.Fatalf("bad risk answer: %+v", risk)
		}
		if got.Usage.InputTokens == 0 || got.Usage.OutputTokens != 0 {
			t.Fatalf("bad usage: %+v", got.Usage)
		}
	})

	t.Run("canceled_context_entry", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		calls := 0
		got, err := scoreTextContextWith(ctx, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 0 {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})

	t.Run("cancellation_before_tokenize_encode", func(t *testing.T) {
		probe := newPollCancelContext(t, 2)
		calls := 0
		got, err := scoreTextContextWith(probe, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 0 || probe.remaining.Load() > 0 {
			t.Fatalf("got=%+v err=%v calls=%d remaining=%d", got, err, calls, probe.remaining.Load())
		}
	})

	t.Run("cancellation_before_callback", func(t *testing.T) {
		probe := newPollCancelContext(t, 3)
		calls := 0
		got, err := scoreTextContextWith(probe, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 0 || probe.remaining.Load() > 0 {
			t.Fatalf("got=%+v err=%v calls=%d remaining=%d", got, err, calls, probe.remaining.Load())
		}
	})

	t.Run("cancellation_after_tokenize_before_callback", func(t *testing.T) {
		probe := newPollCancelContext(t, cancelAtPreScore)
		calls := 0
		got, err := scoreTextContextWith(probe, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 0 || probe.remaining.Load() > 0 {
			t.Fatalf("got=%+v err=%v calls=%d remaining=%d", got, err, calls, probe.remaining.Load())
		}
	})

	t.Run("nil_context", func(t *testing.T) {
		got, err := scoreTextContextWith(nil, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) { return nil, nil })
		if err == nil || got != nil {
			t.Fatalf("accepted nil context: %+v %v", got, err)
		}
	})

	t.Run("nil_callback", func(t *testing.T) {
		got, err := scoreTextContextWith(context.Background(), req, tok, 128, 128, nil)
		if err == nil || got != nil {
			t.Fatalf("accepted nil callback: %+v %v", got, err)
		}
	})

	for name, mutate := range map[string]func(TextRequest) TextRequest{
		"malformed_sorted_order": func(in TextRequest) TextRequest {
			in.Fields[0].SortedOptions = append([]string(nil), in.Fields[0].Options...)
			return in
		},
		"reserved_text": func(in TextRequest) TextRequest {
			in.State = "contains <|reserved|> text"
			return in
		},
		"invalid_score_labels": func(in TextRequest) TextRequest {
			in.Fields[1].Keys = []string{"1", "0"}
			return in
		},
	} {
		t.Run(name, func(t *testing.T) {
			calls := 0
			bad := mutate(cloneTextRequest(req))
			got, err := scoreTextContextWith(context.Background(), bad, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
				calls++
				return [][]float32{{1, 2}, {3, 4}}, nil
			})
			if err == nil || got != nil || calls != 0 {
				t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
			}
		})
	}

	t.Run("callback_error_transactional", func(t *testing.T) {
		boom := errors.New("boom")
		calls := 0
		got, err := scoreTextContextWith(context.Background(), req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return nil, boom
		})
		if !errors.Is(err, boom) || got != nil || calls != 1 {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})

	t.Run("malformed_logits_transactional", func(t *testing.T) {
		calls := 0
		got, err := scoreTextContextWith(context.Background(), req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3}}, nil
		})
		if err == nil || !strings.Contains(err.Error(), "logit row") || got != nil || calls != 1 {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})

	t.Run("cancellation_after_callback_before_assemble", func(t *testing.T) {
		probe := newPollCancelContext(t, cancelAtPostScore)
		calls := 0
		got, err := scoreTextContextWith(probe, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{-4, 4}, {6, -6}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 1 || probe.remaining.Load() > 0 {
			t.Fatalf("got=%+v err=%v calls=%d remaining=%d", got, err, calls, probe.remaining.Load())
		}
	})

	t.Run("cancellation_after_assemble", func(t *testing.T) {
		probe := newPollCancelContext(t, cancelAtPostAssemble)
		calls := 0
		got, err := scoreTextContextWith(probe, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{-4, 4}, {6, -6}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 1 || probe.remaining.Load() > 0 {
			t.Fatalf("got=%+v err=%v calls=%d remaining=%d", got, err, calls, probe.remaining.Load())
		}
	})

	t.Run("pack_limit_error", func(t *testing.T) {
		calls := 0
		got, err := scoreTextContextWith(context.Background(), req, tok, 0, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if err == nil || got != nil || calls != 0 {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})

	t.Run("wrapper_nil_context", func(t *testing.T) {
		s := &TextScorer{model: &qwen.Qwen35BaseModel{}}
		got, err := s.scoreTextContext(nil, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) { return nil, nil })
		if err == nil || got != nil {
			t.Fatalf("accepted nil wrapper context: %+v %v", got, err)
		}
	})

	t.Run("wrapper_canceled_context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		s := &TextScorer{model: &qwen.Qwen35BaseModel{}}
		calls := 0
		got, err := s.scoreTextContext(ctx, req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if !errors.Is(err, context.Canceled) || got != nil || calls != 0 {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})

	t.Run("wrapper_uninitialized", func(t *testing.T) {
		var s *TextScorer
		calls := 0
		got, err := s.scoreTextContext(context.Background(), req, tok, 128, 128, func(EncodedRow) ([][]float32, error) {
			calls++
			return [][]float32{{1, 2}, {3, 4}}, nil
		})
		if err == nil || got != nil || calls != 0 {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})

	t.Run("wrapper_delegates", func(t *testing.T) {
		s := &TextScorer{model: &qwen.Qwen35BaseModel{}}
		calls := 0
		got, err := s.scoreTextContext(context.Background(), req, tok, 128, 128, func(row EncodedRow) ([][]float32, error) {
			calls++
			assertEncodedRowMatchesRequest(t, tok, req, row)
			return [][]float32{{-4, 4}, {6, -6}}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if calls != 1 || got == nil || got.Model != req.Model {
			t.Fatalf("got=%+v err=%v calls=%d", got, err, calls)
		}
	})
}

func TestNVIDIATextCloseRegression(t *testing.T) {
	req := decodeTextOrchestrationRequest(t, `{"model":"m","state":"alpha beta","questions":{"route":{"type":"choice","instructions":"Pick a route.","criteria":{"z":"last","a":"first"}},"risk":{"type":"score","instructions":"Rate the risk.","criteria":["low","high"]}}}`)
	tok := syntheticASCIITokenizer(req)

	t.Run("readiness_lock_wait_cancel", func(t *testing.T) {
		g := &NVIDIATextScorer{cpu: &TextScorer{}}
		g.mu.Lock()
		defer g.mu.Unlock()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan error, 1)
		go func() {
			out, err := g.ScoreTextContext(ctx, req, tok, 128, 128)
			if out != nil {
				err = errors.New("partial result")
			}
			done <- err
		}()
		cancel()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("canceled text caller stuck behind readiness lock")
		}
	})

	t.Run("concurrent_scoretext_close_empty_host", func(t *testing.T) {
		g := &NVIDIATextScorer{cpu: &TextScorer{}}
		const workers = 32
		start := make(chan struct{})
		errs := make(chan error, workers+1)
		var wg sync.WaitGroup
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer func() {
					if r := recover(); r != nil {
						errs <- fmt.Errorf("panic: %v", r)
					}
				}()
				<-start
				got, err := g.ScoreText(req, tok, 128, 128)
				if err == nil {
					errs <- errors.New("missing score error")
					return
				}
				if got != nil {
					errs <- errors.New("partial score output")
				}
			}()
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errs <- fmt.Errorf("panic: %v", r)
				}
			}()
			<-start
			if err := g.Close(); err != nil {
				errs <- err
			}
		}()
		close(start)
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil {
				t.Fatal(err)
			}
		}
		if g.cpu != nil {
			t.Fatal("closed scorer retained host view")
		}
		if out, err := g.ScoreText(req, tok, 128, 128); err == nil || out != nil {
			t.Fatalf("closed scorer usable: %+v %v", out, err)
		}
		if err := g.Close(); err != nil {
			t.Fatal(err)
		}
	})
}

func decodeTextOrchestrationRequest(t *testing.T, raw string) TextRequest {
	t.Helper()
	req, err := DecodeTextRequest(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func cloneTextRequest(req TextRequest) TextRequest {
	out := req
	out.Fields = make([]TextRequestField, len(req.Fields))
	for i, field := range req.Fields {
		out.Fields[i] = field
		out.Fields[i].Keys = append([]string(nil), field.Keys...)
		out.Fields[i].Options = append([]string(nil), field.Options...)
		out.Fields[i].SortedOptions = append([]string(nil), field.SortedOptions...)
		out.Fields[i].SortedIndices = append([]int(nil), field.SortedIndices...)
	}
	return out
}

func syntheticASCIITokenizer(req TextRequest) *tokenizer.Tokenizer {
	texts := []string{req.State}
	for _, field := range req.Fields {
		prompt, err := renderChoicePrompt(TextField{Name: field.ID, Description: field.Instructions, Options: field.SortedOptions})
		if err != nil {
			panic(err)
		}
		texts = append(texts, prompt)
		texts = append(texts, field.ID, field.Instructions)
		texts = append(texts, field.Keys...)
		texts = append(texts, field.Options...)
	}
	enc := syntheticByteEncoder()
	vocab := make(map[string]int)
	inv := make(map[int]string)
	add := func(text string) {
		for i := 0; i < len(text); i++ {
			symbol := string(enc[text[i]])
			if _, ok := vocab[symbol]; ok {
				continue
			}
			id := len(vocab) + 1
			vocab[symbol] = id
			inv[id] = symbol
		}
	}
	for _, text := range texts {
		add(text)
	}
	return &tokenizer.Tokenizer{
		Vocab:        vocab,
		InvVocab:     inv,
		AddedSpecial: map[string]int{"<|reserved|>": 999},
	}
}

func syntheticByteEncoder() [256]rune {
	var enc [256]rune
	var keep [256]bool
	for i := int('!'); i <= int('~'); i++ {
		keep[i] = true
		enc[i] = rune(i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		keep[i] = true
		enc[i] = rune(i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		keep[i] = true
		enc[i] = rune(i)
	}
	n := 256
	for i := 0; i < 256; i++ {
		if keep[i] {
			continue
		}
		enc[i] = rune(n)
		n++
	}
	return enc
}

func assertEncodedRowMatchesRequest(t *testing.T, tok *tokenizer.Tokenizer, req TextRequest, row EncodedRow) {
	t.Helper()
	if !reflect.DeepEqual(row.State, tok.Encode(req.State)) {
		t.Fatalf("state tokens=%v want %v", row.State, tok.Encode(req.State))
	}
	if len(row.Questions) != len(req.Fields) || len(row.Candidates) != len(req.Fields) {
		t.Fatalf("bad row geometry: %+v", row)
	}
	for i, field := range req.Fields {
		prompt, err := renderChoicePrompt(TextField{Name: field.ID, Description: field.Instructions, Options: field.SortedOptions})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(row.Questions[i], tok.Encode(prompt)) {
			t.Fatalf("question %d tokens=%v want %v", i, row.Questions[i], tok.Encode(prompt))
		}
		if len(row.Candidates[i]) != len(field.SortedOptions) {
			t.Fatalf("candidate count[%d]=%d want %d", i, len(row.Candidates[i]), len(field.SortedOptions))
		}
		for n, option := range field.SortedOptions {
			if !reflect.DeepEqual(row.Candidates[i][n], tok.Encode(option)) {
				t.Fatalf("candidate[%d][%d]=%v want %v", i, n, row.Candidates[i][n], tok.Encode(option))
			}
		}
	}
}
