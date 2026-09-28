package gosystemone

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/rcarmo/go-pherence/loader/tokenizer"
	"github.com/rcarmo/go-pherence/model"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestLongContextReleasedModelRecovery(t *testing.T) {
	path, dir := os.Getenv("GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B"), os.Getenv("GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER")
	if path == "" || dir == "" {
		t.Skip("set GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B and GO_PHERENCE_GO_SYSTEM_ONE_GEMMA4_12B_TOKENIZER")
	}
	tok, err := tokenizer.LoadWithConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	m, err := model.LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	m.Tok = tok
	gpu, err := model.NewGemma4NVIDIA(m)
	if err != nil {
		t.Fatal(err)
	}
	defer gpu.Close()
	scorer := &Gemma4NVIDIAScorer{Model: m, GPU: gpu, PackedTokenRows: 512}
	defer scorer.Close()
	engine := &Engine{Tokenizer: tok, Scorer: scorer, BOSToken: m.Config.BOSTokenID}
	text := strings.Repeat("This is a routine report. ", 390) + "All production services have now stopped."
	request := SystemOneRequest{State: mustRaw(text), Questions: json.RawMessage(`{"urgent":{"type":"noul","instructions":"Does this require urgent handling?"},"kind":{"type":"choice","instructions":"Choose the incident category","criteria":{"service outage":"Production unavailable","routine report":"Everything healthy"}}}`)}
	schema, _, state, err := compileSystemOne(request)
	if err != nil {
		t.Fatal(err)
	}
	n := 1 + len(tok.Encode(engine.renderShared(schema.SystemText))) + len(tok.Encode(engine.renderContext(state)))
	if n <= 2048 {
		t.Fatalf("fixture only %d tokens", n)
	}
	a, err := engine.SystemOne(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	b, err := engine.SystemOne(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a.Answers, b.Answers) {
		t.Fatal("compacted-prefix repeat changed answers")
	}
	if scorer.cached.CanReusePrefix() {
		t.Fatal("long compacted cache marked reusable")
	}
	short := SystemOneRequest{State: mustRaw("All services are healthy."), Questions: request.Questions}
	if _, err = engine.SystemOne(context.Background(), short); err != nil {
		t.Fatal("short recovery", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = engine.SystemOne(ctx, request); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	// A large logical reservation must fail before partial KV allocation on this
	// 12 GB fixture; use an impossible model context to test explicit refusal.
	if _, err = gpu.PrefillPreparedCapacity(context.Background(), []int{m.Config.BOSTokenID}, 32769, 1, 1); !errors.Is(err, model.ErrGemma4ContextCapacity) {
		t.Fatalf("capacity error=%v", err)
	}
	if _, err = engine.SystemOne(context.Background(), short); err != nil {
		t.Fatal("refusal recovery", err)
	}
	t.Logf("long request prompt tokens=%d; repeated answers identical; short/cancel/refusal recovery passed", n)
}
func mustRaw(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}
