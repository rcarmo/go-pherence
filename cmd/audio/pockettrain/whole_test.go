//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func wholeData(t *testing.T, entries []wholeEntry) []byte {
	t.Helper()
	var out bytes.Buffer
	for _, e := range entries {
		if err := json.NewEncoder(&out).Encode(e); err != nil {
			t.Fatal(err)
		}
	}
	return out.Bytes()
}
func wholeFixture() []wholeEntry {
	return []wholeEntry{{ID: "a", Path: "audio/a.wav", Transcript: "a", SHA256: strings.Repeat("a", 64), Split: "train"}, {ID: "b", Path: "audio/b.wav", Transcript: "b", SHA256: strings.Repeat("b", 64), Split: "train"}}
}
func TestPreparedWholeManifestAdmission(t *testing.T) {
	if got, err := readWholeManifest(wholeData(t, wholeFixture()), 2); err != nil || len(got) != 2 {
		t.Fatal(got, err)
	}
	for _, mutate := range []func([]wholeEntry){func(e []wholeEntry) { e[1].Split = "dev" }, func(e []wholeEntry) { e[1].ID = e[0].ID }, func(e []wholeEntry) { e[1].Path = e[0].Path }, func(e []wholeEntry) { e[1].SHA256 = e[0].SHA256 }, func(e []wholeEntry) { e[0].Path = "../a.wav" }, func(e []wholeEntry) { e[0].Path = "/a.wav" }, func(e []wholeEntry) { e[0].Transcript = " " }, func(e []wholeEntry) { e[0].SHA256 = "bad" }, func(e []wholeEntry) { e[0].Path = "audio/../a.wav" }} {
		e := wholeFixture()
		mutate(e)
		if _, err := readWholeManifest(wholeData(t, e), 2); err == nil {
			t.Fatal("bad manifest admitted", e)
		}
	}
	for _, data := range [][]byte{nil, []byte(`{}`), []byte(`{"unknown":1}`), []byte(`{"split":"dev","split":"train"}`), append(wholeData(t, wholeFixture()), []byte("{}")...)} {
		if _, err := readWholeManifest(data, 2); err == nil {
			t.Fatal("bad input admitted")
		}
	}
	if _, err := readWholeManifest(wholeData(t, wholeFixture()), 1); err == nil {
		t.Fatal("row ceiling")
	}
}
func TestPreparedWholeReferencePairing(t *testing.T) {
	refs, err := wholeReferences([]int{100000, 96000, 70000, 70000})
	if err != nil || !reflect.DeepEqual(refs, []int{2, 2, 3, 2}) {
		t.Fatal(refs, err)
	}
	for _, samples := range [][]int{{0, 100}, {374*1920 + 1, 100}, {200000, 200000}, {100, 200000}} {
		if _, err := wholeReferences(samples); err == nil {
			t.Fatal("bad target/reference geometry", samples)
		}
	}
	refs, err = wholeReferences([]int{62 * 1920, 374 * 1920, 1920})
	if err != nil || refs[1] != 2 {
		t.Fatal("boundary frames", refs, err)
	}
}
func TestPreparedWholePreflightAndAssembly(t *testing.T) {
	dir := t.TempDir()
	entries := wholeFixture()
	for i, e := range entries {
		p := filepath.Join(dir, e.Path)
		if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
			t.Fatal(err)
		}
		data := wavFixture(1920*(i+1), 1, 16, 1, 24000)
		if err := os.WriteFile(p, data, 0600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		entries[i].SHA256 = hex.EncodeToString(sum[:])
	}
	o := options{root: dir, maxSamples: 24000 * 30, maxTokens: 512, maxResident: 12 << 30}
	r, err := preflightWholeDataset(report{}, configFixture(t), tokFixture(t), entries, o)
	if err != nil || len(r.Rows) != 2 {
		t.Fatal(r, err)
	}
	if r.Rows[0].ReferencePath != entries[1].Path || r.Rows[1].ReferencePath != entries[0].Path {
		t.Fatal("reference pairing", r.Rows)
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()
	if _, err := readWholeWAV(root, r.Rows[0].ReferencePath, r.Rows[0].ReferenceAudio, r.Rows[0].ReferenceSamples, 24000*30); err != nil {
		t.Fatal(err)
	}
	target, reference := make([]float32, 1920), make([]float32, 3840)
	reference[0] = 3
	target[0] = 5
	call := 0
	encode := func(in []float32) ([]float32, int, error) {
		call++
		if call == 1 {
			if len(in) != 3840 || in[0] != 3 {
				t.Fatal("reference not whole")
			}
			return []float32{3, 4, 5, 6}, 2, nil
		}
		if len(in) != 1920 || in[0] != 5 {
			t.Fatal("target not whole")
		}
		return []float32{5, 6}, 1, nil
	}
	prepared, err := assembleWholePreparedRow(target, reference, r.Rows[0], []uint32{1}, []float32{1, 2}, []float32{2, 2}, encode)
	if err != nil || prepared.ValidFrames != 1 || !reflect.DeepEqual(prepared.Batch.NormalizedLatents, []float32{2, 2, 0, 0}) || prepared.Batch.VoiceFrames != 2 {
		t.Fatal(prepared, err)
	}
	bad := r.Rows[0]
	bad.Samples++
	if _, err := assembleWholePreparedRow(target, reference, bad, []uint32{1}, []float32{1, 2}, []float32{2, 2}, encode); err == nil {
		t.Fatal("sample mismatch")
	}
	entries[0].SHA256 = strings.Repeat("0", 64)
	if _, err := preflightWholeDataset(report{}, configFixture(t), tokFixture(t), entries, o); err == nil {
		t.Fatal("pin mismatch")
	}
}
