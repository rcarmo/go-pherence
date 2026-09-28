package model

import "testing"

func TestBuildPreparedMTPPromptContextDoesNotWrapAgain(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	m.Config.BOSTokenID = 2
	prepared := []int{2, 1}
	ctx, err := m.BuildPreparedMTPPromptContext(prepared)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(ctx.Tokens, prepared) || ctx.SeqLen != len(prepared) || ctx.PreviousToken != prepared[len(prepared)-1] {
		t.Fatalf("prepared context=%+v", ctx)
	}
	prepared[0] = 0
	if ctx.Tokens[0] != 2 {
		t.Fatal("prepared prompt context aliases caller tokens")
	}
}
