package main

import (
	"encoding/json"
	"testing"
)

func TestSmokeResultJSON(t *testing.T) {
	res := smokeResult{ModelHidden: 5376, DrafterHidden: 1024, PackedEmbedding: true, Token: 1}
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	var got smokeResult
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.ModelHidden != 5376 || !got.PackedEmbedding || got.Token != 1 {
		t.Fatalf("roundtrip=%+v", got)
	}
}
