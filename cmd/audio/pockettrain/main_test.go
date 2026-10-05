//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

func configFixture(t *testing.T) pockettts.Config {
	t.Helper()
	data, err := os.ReadFile("../../../model/pockettts/testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	c, err := pockettts.ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	return c
}
func tokFixture(t *testing.T) *pockettts.Tokenizer {
	t.Helper()
	tok, err := pockettts.LoadTokenizer(bytes.NewReader(tokDataFixture(t, false)))
	if err != nil {
		t.Fatal(err)
	}
	return tok
}
func tokDataFixture(t *testing.T, full bool) []byte {
	t.Helper()
	vocab := []any{[]any{"<unk>", -10}, []any{"▁", -1}, []any{"a", -1}, []any{"b", -1}}
	for i := 0; i < 256; i++ {
		const hexdigits = "0123456789ABCDEF"
		piece := "<0x" + string(hexdigits[i>>4]) + string(hexdigits[i&15]) + ">"
		vocab = append(vocab, []any{piece, -5})
	}
	if full {
		for i := len(vocab); i < 4000; i++ {
			vocab = append(vocab, []any{fmt.Sprintf("<unused%d>", i), -100})
		}
	}
	raw := map[string]any{"normalizer": map[string]any{"type": "Prepend", "prepend": "▁"}, "pre_tokenizer": map[string]any{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}, "post_processor": nil, "model": map[string]any{"type": "Unigram", "unk_id": 0, "byte_fallback": true, "vocab": vocab}}
	data, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
func entryFixture() pockettts.TrainingEntry {
	a, b, c, d := 0.1, 1.0, 1.4, 2.4
	return pockettts.TrainingEntry{Path: "recording.wav", Duration: 3, Transcript: "a b", Words: []pockettts.TrainingWord{{Word: "a", Start: &a, End: &b}, {Word: "b", Start: &c, End: &d}}}
}
func wavFixture(samples int, format, bits, channels, rate int) []byte {
	width := bits / 8
	data := make([]byte, 44+samples*width)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], uint16(format))
	binary.LittleEndian.PutUint16(data[22:], uint16(channels))
	binary.LittleEndian.PutUint32(data[24:], uint32(rate))
	binary.LittleEndian.PutUint32(data[28:], uint32(rate*width*channels))
	binary.LittleEndian.PutUint16(data[32:], uint16(width*channels))
	binary.LittleEndian.PutUint16(data[34:], uint16(bits))
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], uint32(samples*width))
	return data
}
func openFixture(t *testing.T, dir, name string, data []byte) *os.File {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, data, 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	return f
}
func marshalFixture(t *testing.T, v any) []byte {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
func writeFixture(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}
func requireError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("want error containing %q, got %v", want, err)
	}
}
func TestPocketTrainAdmission(t *testing.T) {
	for _, args := range [][]string{nil, {"-backend", "vulkan"}, {"-backend", "ptx"}, {"-mode", "train"}, {"-max-rows", "0"}, {"-max-audio-samples", "0"}} {
		var out bytes.Buffer
		if err := run(args, &out); err == nil || out.Len() != 0 {
			t.Fatalf("admission args=%v err=%v output=%q", args, err, out.String())
		}
	}
	for _, backend := range []string{"vulkan", "ptx"} {
		err := run([]string{"-backend", backend, "-config", "/missing"}, &bytes.Buffer{})
		if err == nil || !strings.Contains(err.Error(), "unsupported") {
			t.Fatalf("backend must reject before files: %v", err)
		}
	}
	if validDigest(strings.Repeat("A", 64)) || validDigest("short") || !validDigest(strings.Repeat("a", 64)) {
		t.Fatal("digest validation")
	}
}
func TestPocketTrainManifestBounds(t *testing.T) {
	good := marshalFixture(t, entryFixture())
	if rows, err := readManifest(good, 1); err != nil || len(rows) != 1 {
		t.Fatalf("valid manifest: %v", err)
	}
	for _, tc := range []struct {
		raw     []byte
		message string
	}{
		{nil, "empty"}, {[]byte("\n"), "JSON"},
		{append(append(append([]byte{}, good...), '\n'), good...), "row limit"},
		{[]byte(strings.Repeat("x", 65537)), "token too long"},
	} {
		_, err := readManifest(tc.raw, 1)
		requireError(t, err, tc.message)
	}
	e := entryFixture()
	e.Transcript = strings.Repeat("x", 4097)
	bad := marshalFixture(t, e)
	_, err := readManifest(bad, 1)
	requireError(t, err, "path/transcript/word limit")
	e = entryFixture()
	e.Words = make([]pockettts.TrainingWord, 513)
	bad = marshalFixture(t, e)
	_, err = readManifest(bad, 1)
	requireError(t, err, "path/transcript/word limit")
}
func TestPocketTrainWAVBounds(t *testing.T) {
	dir := t.TempDir()
	for _, fmtbits := range [][2]int{{1, 16}, {3, 32}} {
		data := wavFixture(24000, fmtbits[0], fmtbits[1], 1, 24000)
		f := openFixture(t, dir, "valid"+string(rune(fmtbits[0]))+".wav", data)
		n, a, err := scanWAV(f, 24000)
		if err != nil || n != 24000 || a.Bytes != int64(len(data)) || !validDigest(a.SHA256) {
			t.Fatalf("valid WAV n=%d err=%v", n, err)
		}
	}
	cases := [][]byte{wavFixture(4, 1, 16, 2, 24000), wavFixture(4, 1, 16, 1, 16000), wavFixture(4, 1, 16, 1, 24000)[:45]}
	forged := wavFixture(4, 1, 16, 1, 24000)
	binary.LittleEndian.PutUint32(forged[40:], math.MaxUint32)
	cases = append(cases, forged)
	clipped := wavFixture(4, 1, 16, 1, 24000)
	binary.LittleEndian.PutUint16(clipped[44:], math.MaxInt16)
	cases = append(cases, clipped)
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), 1.01, 1} {
		raw := wavFixture(4, 3, 32, 1, 24000)
		binary.LittleEndian.PutUint32(raw[44:], math.Float32bits(v))
		cases = append(cases, raw)
	}
	for i, raw := range cases {
		f := openFixture(t, dir, "bad"+string(rune('a'+i)), raw)
		_, _, err := scanWAV(f, 24000)
		message := "nonfinite/out-of-range/full-scale"
		switch i {
		case 0, 1:
			message = "format/rate/channels/alignment"
		case 2, 3:
			message = "declared sample count/size"
		case 4:
			message = "clipping sample"
		}
		requireError(t, err, message)
	}
	f := openFixture(t, dir, "too-long", wavFixture(20, 1, 16, 1, 24000))
	_, _, err := scanWAV(f, 19)
	requireError(t, err, "declared sample count/size")
}
func TestPocketTrainSourceConfinement(t *testing.T) {
	root := t.TempDir()
	outside := t.TempDir()
	raw := wavFixture(72000, 1, 16, 1, 24000)
	openFixture(t, outside, "recording.wav", raw)
	if err := os.Symlink(filepath.Join(outside, "recording.wav"), filepath.Join(root, "escape.wav")); err != nil {
		t.Fatal(err)
	}
	cfg, tok := configFixture(t), tokFixture(t)
	o := options{root: root, maxSamples: 72000, maxTokens: 512, maxResident: 16 << 30}
	for _, path := range []string{"../recording.wav", filepath.Join(outside, "recording.wav"), "escape.wav"} {
		e := entryFixture()
		e.Path = path
		_, err := preflightDataset(report{}, cfg, tok, []pockettts.TrainingEntry{e}, o)
		message := "relative within source-root"
		if path == "escape.wav" {
			message = "escapes from parent"
		}
		requireError(t, err, message)
	}
	openFixture(t, root, "recording.wav", raw)
	e := entryFixture()
	before := marshalFixture(t, e)
	r, err := preflightDataset(report{}, cfg, tok, []pockettts.TrainingEntry{e}, o)
	if err != nil || len(r.Rows) != 1 || r.Rows[0].Samples != 72000 {
		t.Fatalf("valid confined row %v", err)
	}
	after := marshalFixture(t, e)
	if !bytes.Equal(before, after) {
		t.Fatal("entry mutated")
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatal("preflight wrote files")
	}
	disk, err := os.ReadFile(filepath.Join(root, "recording.wav"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, disk) {
		t.Fatal("recording modified")
	}
}
func TestPocketTrainRowResourcePlan(t *testing.T) {
	cfg, tok, e := configFixture(t), tokFixture(t), entryFixture()
	r, err := planRow(cfg, tok, e, 72000, 512, 16<<30)
	if err != nil || r.CutSeconds != 1.2 || r.TargetFrames != 23 || r.VoiceFrames != 15 || r.ResidentUpperBytes <= 0 {
		t.Fatalf("deterministic row plan %+v %v", r, err)
	}
	_, err = planRow(cfg, tok, e, 48000, 512, 16<<30)
	requireError(t, err, "exceeds recording duration")
	_, err = planRow(cfg, tok, e, 72000, 1, 16<<30)
	requireError(t, err, "tokenizer count exceeds limit")
	_, err = planRow(cfg, tok, e, 72000, 512, 1)
	requireError(t, err, "resident upper bytes")
}
func TestPocketTrainHeaderAndAssetBounds(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "model.safetensors")
	var prefix [8]byte
	binary.LittleEndian.PutUint64(prefix[:], 1<<21)
	if err := os.WriteFile(p, prefix[:], 0600); err != nil {
		t.Fatal(err)
	}
	_, err := inspectModel(p, 1<<20, configFixture(t))
	requireError(t, err, "header exceeds")
	_, _, err = boundedBytes(p, 7)
	requireError(t, err, "within 7 bytes")
	_, _, err = boundedBytes(dir, 1<<20)
	requireError(t, err, "not a regular file")
	h := sha256.Sum256(prefix[:])
	if _, a, err := boundedBytes(p, 8); err != nil || a.SHA256 != hex.EncodeToString(h[:]) {
		t.Fatalf("asset provenance %v", err)
	}
}
