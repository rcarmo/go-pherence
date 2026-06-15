package diffusiongemma

import "testing"

func TestAppendEncoderKVLayer(t *testing.T) {
	prefix := EncoderKVLayer{SeqLen: 2, KVHeads: 1, HeadDim: 2, Keys: []float32{1, 2, 3, 4}, Values: []float32{10, 20, 30, 40}}
	suffix := EncoderKVLayer{SeqLen: 1, KVHeads: 1, HeadDim: 2, Keys: []float32{5, 6}, Values: []float32{50, 60}}
	out, err := appendEncoderKVLayer(prefix, suffix)
	if err != nil {
		t.Fatal(err)
	}
	if out.SeqLen != 3 || out.KVHeads != 1 || out.HeadDim != 2 {
		t.Fatalf("bad output shape: %+v", out)
	}
	wantK := []float32{1, 2, 3, 4, 5, 6}
	wantV := []float32{10, 20, 30, 40, 50, 60}
	for i := range wantK {
		if out.Keys[i] != wantK[i] || out.Values[i] != wantV[i] {
			t.Fatalf("bad append at %d keys=%v values=%v", i, out.Keys, out.Values)
		}
	}
	out.Keys[0] = 99
	if prefix.Keys[0] == 99 {
		t.Fatalf("append aliased prefix keys")
	}
}

func TestAppendEncoderKVLayerNoSuffixDoesNotAlias(t *testing.T) {
	prefix := EncoderKVLayer{SeqLen: 1, KVHeads: 1, HeadDim: 2, Keys: []float32{1, 2}, Values: []float32{3, 4}}
	out, err := appendEncoderKVLayer(prefix, EncoderKVLayer{})
	if err != nil {
		t.Fatal(err)
	}
	out.Keys[0] = 99
	out.Values[0] = 99
	if prefix.Keys[0] == 99 || prefix.Values[0] == 99 {
		t.Fatalf("no-suffix append aliased prefix buffers")
	}
}

func TestAppendEncoderKVLayerRejectsMismatch(t *testing.T) {
	prefix := EncoderKVLayer{SeqLen: 1, KVHeads: 1, HeadDim: 2, Keys: []float32{1, 2}, Values: []float32{3, 4}}
	suffix := EncoderKVLayer{SeqLen: 1, KVHeads: 2, HeadDim: 2, Keys: []float32{1, 2, 3, 4}, Values: []float32{1, 2, 3, 4}}
	if _, err := appendEncoderKVLayer(prefix, suffix); err == nil {
		t.Fatalf("accepted KV head mismatch")
	}
}

func TestAppendEncoderKVLayerRejectsShortBuffers(t *testing.T) {
	prefix := EncoderKVLayer{SeqLen: 2, KVHeads: 1, HeadDim: 2, Keys: []float32{1, 2}, Values: []float32{1, 2, 3, 4}}
	suffix := EncoderKVLayer{SeqLen: 1, KVHeads: 1, HeadDim: 2, Keys: []float32{5, 6}, Values: []float32{7, 8}}
	if _, err := appendEncoderKVLayer(prefix, suffix); err == nil {
		t.Fatalf("accepted short prefix keys")
	}
}

func TestAppendEncoderKVLayers(t *testing.T) {
	prefix := []EncoderKVLayer{{SeqLen: 1, KVHeads: 1, HeadDim: 1, Keys: []float32{1}, Values: []float32{2}}}
	emptyOut, err := appendEncoderKVLayers(prefix, nil)
	if err != nil {
		t.Fatal(err)
	}
	emptyOut[0].Keys[0] = 99
	if prefix[0].Keys[0] == 99 {
		t.Fatalf("empty suffix append aliased prefix layer")
	}
	suffix := []EncoderKVLayer{{SeqLen: 1, KVHeads: 1, HeadDim: 1, Keys: []float32{3}, Values: []float32{4}}}
	out, err := appendEncoderKVLayers(prefix, suffix)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].SeqLen != 2 || out[0].Keys[1] != 3 || out[0].Values[1] != 4 {
		t.Fatalf("bad append layers: %+v", out)
	}
	if _, err := appendEncoderKVLayers(prefix, append(suffix, suffix[0])); err == nil {
		t.Fatalf("accepted layer count mismatch")
	}
}
