//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/rcarmo/go-pherence/model/pockettts"
)

func TestPocketTrainSpecialFiles(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "fifo")
	if err := syscall.Mkfifo(path, 0600); err != nil {
		t.Fatal(err)
	}
	_, _, err := boundedFile(path, 1<<20)
	requireError(t, err, "not a regular file")
	o := options{root: root, maxSamples: 72000, maxTokens: 512, maxResident: 16 << 30}
	e := entryFixture()
	e.Path = "fifo"
	_, err = preflightDataset(report{}, configFixture(t), tokFixture(t), []pockettts.TrainingEntry{e}, o)
	requireError(t, err, "audio is not a regular file")
}
func TestPocketTrainSampleExactGeometry(t *testing.T) {
	cfg, tok := configFixture(t), tokFixture(t)
	for _, tc := range []struct {
		cut, end      float64
		voice, target int
		ok            bool
	}{
		{1.2, 3, 15, 23, true}, {1.2, 1.2 + 1920.0/24000, 15, 1, true},
		{1.2, 1.2 + 1921.0/24000, 15, 2, true}, {4.96, 5.96, 62, 13, true},
		{5, 6, 0, 0, false}, {1.2, 31.2, 15, 375, true}, {1.2, 31.2 + 1.0/24000, 0, 0, false},
	} {
		e := entryFixture()
		e.Duration = tc.end
		left, right := tc.cut, tc.cut
		e.Words[0].End = &left
		e.Words[1].Start = &right
		end := tc.end
		e.Words[1].End = &end
		// Some short target cases bypass manifest minimum1s solely to test encoder
		// geometry through sample conversion; use a direct cut helper below.
		if tc.end-tc.cut < 1 {
			got, err := timestampSample(tc.end, 35*24000)
			if err != nil {
				t.Fatal(err)
			}
			start, err := timestampSample(tc.cut, 35*24000)
			if err != nil {
				t.Fatal(err)
			}
			frames := (got - start + 1919) / 1920
			if frames != tc.target {
				t.Fatalf("tail frames=%d want=%d", frames, tc.target)
			}
			continue
		}
		r, err := planRow(cfg, tok, e, 35*24000, 512, 16<<30)
		if (err == nil) != tc.ok {
			t.Fatalf("cut=%g end=%g result=%+v err=%v", tc.cut, tc.end, r, err)
		}
		if tc.ok && (r.VoiceFrames != tc.voice || r.TargetFrames != tc.target) {
			t.Fatalf("wrong ceil geometry %+v", r)
		}
	}
	if _, err := timestampSample(4, 3*24000); err == nil {
		t.Fatal("out-of-recording boundary")
	}
}
func TestPocketTrainStrictTensorHeader(t *testing.T) {
	valid := []byte(`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}`)
	if infos, err := parseModelHeader(valid, 6); err != nil || len(infos) != 2 {
		t.Fatalf("valid structural header %v", err)
	}
	bad := []string{
		`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}}`,
		`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2,4]}}`,
		`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0]}}`,
		`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`,
		`{"a":{"dtype":"BF16","dtype":"F32","shape":[1],"data_offsets":[0,2]}}`,
	}
	for _, header := range bad {
		if _, err := parseModelHeader([]byte(header), 4); err == nil {
			t.Fatalf("invalid header %s", header)
		}
	}
	if _, err := parseModelHeader(valid, 7); err == nil {
		t.Fatal("uncovered payload")
	}
}
func TestPocketTrainTypedHeaderRejection(t *testing.T) {
	for _, tc := range []struct{ header, err string }{
		{`{"a":{"dtype":null,"shape":[1],"data_offsets":[0,2]}}`, "dtype"},
		{`{"a":{"dtype":"BF16","shape":null,"data_offsets":[0,2]}}`, "shape"},
		{`{"a":{"dtype":"BF16","shape":[null],"data_offsets":[0,2]}}`, "integer syntax"},
		{`{"a":{"dtype":"BF16","shape":[1.0],"data_offsets":[0,2]}}`, "integer syntax"},
		{`{"a":{"dtype":"BF16","shape":[1e0],"data_offsets":[0,2]}}`, "integer syntax"},
		{`{"a":{"dtype":"BF16","shape":[-1],"data_offsets":[0,2]}}`, "negative"},
		{`{"a":{"dtype":"BF16","shape":[9999999999999999999999],"data_offsets":[0,2]}}`, "integer range"},
		{`{"a":{"dtype":"BF16","shape":[1],"data_offsets":null}}`, "offsets"},
		{`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[null,2]}}`, "integer syntax"},
		{`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[-1,1]}}`, "negative"},
		{`{"__metadata__":null}`, "metadata"},
		{`{"__metadata__":{"key":null}}`, "metadata"},
		{`{"__metadata__":{"key":7}}`, "metadata"},
		{`{"__metadata__":{"key":"a","key":"b"}}`, "duplicate"},
	} {
		_, err := parseModelHeader([]byte(tc.header), 2)
		requireError(t, err, tc.err)
	}
}

func TestPocketTrainModelFileStructure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tiny.safetensors")
	header := []byte(`{"__metadata__":{"fixture":"tiny"},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}}`)
	encode := func(h []byte) []byte {
		raw := make([]byte, 8+len(h)+6)
		binary.LittleEndian.PutUint64(raw, uint64(len(h)))
		copy(raw[8:], h)
		return raw
	}
	raw := encode(header)
	writeFixture(t, path, raw)
	infos, a, err := inspectModelStructure(path, 1<<20)
	if err != nil || len(infos) != 2 || a.Bytes != int64(len(raw)) {
		t.Fatalf("real tiny structural reader: %v", err)
	}
	h := sha256.Sum256(raw)
	if a.SHA256 != hex.EncodeToString(h[:]) {
		t.Fatal("structural reader digest")
	}
	_, err = inspectModel(path, 1<<20, configFixture(t))
	requireError(t, err, "model inventory invalid")
	writeFixture(t, path, raw[:len(raw)-1])
	_, _, err = inspectModelStructure(path, 1<<20)
	requireError(t, err, "offset/shape mismatch")
	writeFixture(t, path, raw[:8])
	_, _, err = inspectModelStructure(path, 1<<20)
	requireError(t, err, "header exceeds")
	writeFixture(t, path, encode([]byte(`{"a":{"dtype":"BF16","shape":[1],"data_offsets":[null,2]}}`)))
	_, _, err = inspectModelStructure(path, 1<<20)
	requireError(t, err, "integer syntax")
	writeFixture(t, path, encode([]byte(`{"__metadata__":{"x":null},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}`)))
	_, _, err = inspectModelStructure(path, 1<<20)
	requireError(t, err, "metadata")
}

func TestPocketTrainSyntheticPreflight(t *testing.T) {
	dir := t.TempDir()
	config := filepath.Join(dir, "config.yaml")
	raw, err := os.ReadFile("../../../model/pockettts/testdata/english-upstream.yaml")
	if err != nil {
		t.Fatal(err)
	}
	writeFixture(t, config, raw)
	tokenizer := filepath.Join(dir, "tokenizer.json")
	raw = tokDataFixture(t, true)
	writeFixture(t, tokenizer, raw)
	th := sha256.Sum256(raw)
	manifest := filepath.Join(dir, "manifest.jsonl")
	e := entryFixture()
	raw = marshalFixture(t, e)
	writeFixture(t, manifest, raw)
	audio := wavFixture(72000, 1, 16, 1, 24000)
	writeFixture(t, filepath.Join(dir, "recording.wav"), audio)
	model := filepath.Join(dir, "model.safetensors")
	writeFixture(t, model, []byte("synthetic metadata test seam"))
	// Strict reader is tested separately. Inject only this file's deterministic
	// metadata result so full CLI orchestration need not materialise a released
	// model or manufacture a gigabyte-sized synthetic checkpoint.
	old := preflightModelInfo
	calls := 0
	preflightModelInfo = func(path string, limit int64, cfg pockettts.Config) (asset, error) {
		calls++
		return asset{strings.Repeat("a", 64), 27}, nil
	}
	t.Cleanup(func() { preflightModelInfo = old })
	args := []string{"-config", config, "-model", model, "-tokenizer", tokenizer, "-manifest", manifest, "-source-root", dir, "-model-sha256", strings.Repeat("a", 64), "-tokenizer-sha256", hex.EncodeToString(th[:])}
	var out bytes.Buffer
	if err := run(args, &out); err != nil {
		t.Fatal(err)
	}
	var r report
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatal(err)
	}
	if len(r.Rows) != 1 || r.TrainingImplemented || r.ModelValuesValidated || r.Rows[0].TargetFrames != 23 || r.Rows[0].VoiceFrames != 15 || calls != 1 {
		t.Fatalf("misleading report %+v", r)
	}
	first := append([]byte(nil), out.Bytes()...)
	out.Reset()
	if err := run(args, &out); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, out.Bytes()) {
		t.Fatal("nondeterministic preflight")
	}
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 5 {
		t.Fatal("preflight wrote input directory")
	}
	args[len(args)-1] = strings.Repeat("b", 64)
	out.Reset()
	requireError(t, run(args, &out), "tokenizer digest mismatch")
	if out.Len() != 0 {
		t.Fatal("digest mismatch produced readiness output")
	}
	args[len(args)-1] = hex.EncodeToString(th[:])
	args[len(args)-3] = strings.Repeat("b", 64)
	requireError(t, run(args, &out), "model digest mismatch")
}
