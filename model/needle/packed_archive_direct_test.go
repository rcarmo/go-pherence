package needle

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

func TestLoadArchivePackedTinyParity(t *testing.T) {
	path := "../../loader/needle/testdata/needle3.cact"
	baseline, tok, err := LoadArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	loaded, packedTok, err := LoadArchivePacked(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok == nil || packedTok == nil || tok.VocabSize() != packedTok.VocabSize() {
		t.Fatal("tokenizer mismatch")
	}
	compact, err := baseline.CompactPacked()
	if err != nil {
		t.Fatal(err)
	}
	if !loaded.compactPacked || loaded.Checkpoint() != nil || loaded.DecodedBytes() != compact.DecodedBytes() || loaded.PackedBytes() != compact.PackedBytes() {
		t.Fatalf("direct bytes dense=%d packed=%d compact dense=%d packed=%d", loaded.DecodedBytes(), loaded.PackedBytes(), compact.DecodedBytes(), compact.PackedBytes())
	}
	if _, err := loaded.Forward([]int{2}, Options{}); err == nil {
		t.Fatal("direct model accepted dense execution")
	}
	if _, err := loaded.SliceDepth(2); err == nil {
		t.Fatal("direct model accepted depth slicing")
	}
	ids := []int{2, 7, 4, 9, 3, 6, 5, 8, 7, 3, 5, 2}
	opts := Options{Packed: true}
	for i := 1; i <= len(ids); i++ {
		want, err := baseline.Forward(ids[:i], opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := loaded.Forward(ids[:i], opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "direct full", got, want, 1e-4, 5e-3)
	}
	dec, err := loaded.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	ref, err := baseline.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range ids {
		want, err := ref.Step(context.Background(), token)
		if err != nil {
			t.Fatal(err)
		}
		got, err := dec.Step(context.Background(), token)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "direct cached", got, want, 1e-4, 5e-3)
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		want, err := baseline.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := loaded.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "direct head", got, want, 2e-5, 5e-3)
	}
}

// Opt-in: the released archive is ignored by Git and must be pinned locally.
func TestReleasedDirectPackedTwelveTokenParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEEDLE_PACKED_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEEDLE_PACKED_MODEL to the pinned local archive")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38" {
		t.Fatalf("archive hash mismatch: %s", got)
	}
	baseline, tok, err := LoadArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	direct, packedTok, err := LoadArchivePacked(path)
	if err != nil {
		t.Fatal(err)
	}
	if packedTok == nil || tok == nil || tok.VocabSize() != packedTok.VocabSize() {
		t.Fatal("tokenizer mismatch")
	}
	if !direct.compactPacked || direct.Checkpoint() != nil || direct.DecodedBytes() >= baseline.DecodedBytes() {
		t.Fatalf("decoded retained bytes baseline=%d direct=%d", baseline.DecodedBytes(), direct.DecodedBytes())
	}
	t.Logf("released logical decoded data baseline=%d direct=%d; packed direct=%d (not process RSS)", baseline.DecodedBytes(), direct.DecodedBytes(), direct.PackedBytes())
	ids := []int{2, 7, 4, 9, 3, 6, 5, 8, 7, 3, 5, 2}
	opts := Options{Packed: true}
	want, err := baseline.Forward(ids, opts)
	if err != nil {
		t.Fatal(err)
	}
	got, err := direct.Forward(ids, opts)
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "released direct full", got, want, 1e-4, 5e-3)
	ref, err := baseline.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	dec, err := direct.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		want, err = ref.Step(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		got, err = dec.Step(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "released direct cached", got, want, 1e-4, 5e-3)
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		if _, _, _, err = baseline.headGeometry(kind); err != nil {
			continue
		}
		want, err = baseline.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err = direct.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "released direct head", got, want, 2e-5, 5e-3)
	}
}

// Opt-in released prompt parity against the earlier recorded twelve-token
// continuation. This checks model/tokenizer integration, not answer quality.
func TestReleasedDirectPackedPromptContinuation(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEEDLE_PACKED_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEEDLE_PACKED_MODEL to the pinned local archive")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38" {
		t.Fatalf("archive hash mismatch: %s", got)
	}
	baseline, tok, err := LoadArchive(path)
	if err != nil {
		t.Fatal(err)
	}
	direct, packedTok, err := LoadArchivePacked(path)
	if err != nil {
		t.Fatal(err)
	}
	prompt := "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
	ids, err := tok.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	gotIDs, err := packedTok.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, gotIDs) {
		t.Fatal("tokenizer mismatch")
	}
	_, _, bos, eos := tok.SpecialIDs()
	ids = append([]int{bos}, ids...)
	wantTokens := []int{6, 38, 8141, 1515, 4047, 997, 598, 782, 326, 1118, 296, 1957}
	for _, tc := range []struct {
		name   string
		model  *Model
		packed bool
	}{{"decoded", baseline, false}, {"direct", direct, true}} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.model.GenerateCached(context.Background(), ids, len(wantTokens), eos, DecoderOptions{Execution: Options{Packed: tc.packed}})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got, wantTokens) {
				t.Fatalf("continuation=%v want %v", got, wantTokens)
			}
		})
	}
}

func TestDirectPackedTinyConcurrentCancellationAndOwnership(t *testing.T) {
	m, _, err := LoadArchivePacked("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 7, 4, 9}
	want, err := m.Forward(ids, Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				decoder, e := m.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: Options{Packed: true}})
				if e != nil {
					t.Error(e)
					return
				}
				cancelled, cancel := context.WithCancel(context.Background())
				cancel()
				if result, e := decoder.Step(cancelled, ids[0]); e != context.Canceled || result != nil || decoder.Position() != 0 {
					t.Errorf("cancelled step result=%v err=%v position=%d", result, e, decoder.Position())
					return
				}
				for _, id := range ids {
					if _, e = decoder.Step(context.Background(), id); e != nil {
						t.Error(e)
						return
					}
				}
				got, e := m.Forward(ids, Options{Packed: true})
				if e != nil {
					t.Error(e)
					return
				}
				compare(t, "concurrent direct packed", got, want, 1e-4, 5e-3)
				clear(got)
			}
		}()
	}
	wg.Wait()
	got, err := m.Forward(ids, Options{Packed: true})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "direct caller mutation", got, want, 1e-4, 5e-3)
}

func TestDirectPackedRejectsIncompleteArchiveAndV2(t *testing.T) {
	if _, _, err := LoadArchivePacked("../../loader/needle/testdata/needle2.cact"); err == nil || !strings.Contains(err.Error(), "Needle3") {
		t.Fatalf("accepted Needle2: %v", err)
	}
	bytes, err := os.ReadFile("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	archive, err := checkpoint.ParseArchivePacked(bytes)
	if err != nil {
		t.Fatal(err)
	}
	// A missing packed payload cannot masquerade as a decoded tensor.
	archive.Records[0].CQBlob = nil
	if _, _, err = mapArchiveConfigMode(archive, true, nil, true); err == nil {
		t.Fatal("accepted missing packed embedding")
	}
	archive, err = checkpoint.ParseArchivePacked(bytes)
	if err != nil {
		t.Fatal(err)
	}
	// Record 2 is a layer-0 attention projection. Replace just that CQ
	// record with decoded data and leave the other layers packed.
	if archive.Records[2].DType != 3 {
		t.Fatal("fixture changed: record 2 is not CQ")
	}
	decoded, err := checkpoint.DecodeCQRecord(2, archive.Records[2], archive.Codebook)
	if err != nil {
		t.Fatal(err)
	}
	archive.Records[2].DType = 2
	archive.Records[2].Data = decoded
	archive.Records[2].CQBlob = nil
	if _, _, err = mapArchiveConfigMode(archive, true, nil, true); err == nil || !strings.Contains(err.Error(), "partial direct CQ coverage") {
		t.Fatalf("accepted partial-layer CQ tensor: %v", err)
	}
}
