package pockettts

import (
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func runIdentityFixture() TrainingRunIdentity {
	return TrainingRunIdentity{SourceRevision: strings.Repeat("a", 40), GoVersion: "go1.26.3", GOOS: "linux", GOARCH: "amd64", SIMDMode: "avx2-fma", Backend: "simd", CorpusSHA256: strings.Repeat("1", 64), InitialModelSHA256: strings.Repeat("2", 64), TokenizerSHA256: strings.Repeat("3", 64), MimiSHA256: strings.Repeat("4", 64), ConfigSHA256: strings.Repeat("5", 64), OptimizerSHA256: strings.Repeat("6", 64), OptionsSHA256: strings.Repeat("7", 64)}
}
func runBindingFixture(step uint64) TrainingCheckpointBinding {
	return TrainingCheckpointBinding{Name: "generation-0001.bin", SHA256: strings.Repeat("8", 64), Step: step}
}
func newRunFixture(t *testing.T, rows int) *TrainingRunState {
	t.Helper()
	s, err := NewTrainingRunState(runIdentityFixture(), rows, [6]uint64{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}
	return s
}
func snapshotRunFixture(t *testing.T, s *TrainingRunState) []byte {
	t.Helper()
	step, _, _ := s.Progress()
	data, err := s.Snapshot(runBindingFixture(step))
	if err != nil {
		t.Fatal(err)
	}
	return data
}
func expectRunError(t *testing.T, err error, text string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), text) {
		t.Fatalf("expected %q, got %v", text, err)
	}
}
func rowAdvance(t *testing.T, s *TrainingRunState, draws int) (uint32, []uint64) {
	t.Helper()
	p, err := s.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	row, err := p.Row()
	if err != nil {
		t.Fatal(err)
	}
	values := make([]uint64, 0, 2*draws)
	for i := 0; i < draws; i++ {
		v, err := p.TimeUint64()
		if err != nil {
			t.Fatal(err)
		}
		values = append(values, v)
		v, err = p.NoiseUint64()
		if err != nil {
			t.Fatal(err)
		}
		values = append(values, v)
	}
	step, _, _ := s.Progress()
	if err := s.CommitRow(p, step+1); err != nil {
		t.Fatal(err)
	}
	return row, values
}

func TestTrainingRunIndependentStreams(t *testing.T) {
	s := newRunFixture(t, 1)
	p, err := s.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	row, err := p.Row()
	if err != nil || row != 0 {
		t.Fatalf("one-row permutation row=%d err=%v", row, err)
	}
	// Independent standard-library PCGs enforce the discrete raw stream
	// contract. No floating samples/model/audio are generated in these tests.
	timeRef, noiseRef := rand.NewPCG(3, 4), rand.NewPCG(5, 6)
	for i := 0; i < 9; i++ {
		got, err := p.TimeUint64()
		if err != nil || got != timeRef.Uint64() {
			t.Fatal("time PCG state changed")
		}
	}
	for i := 0; i < 7; i++ {
		got, err := p.NoiseUint64()
		if err != nil || got != noiseRef.Uint64() {
			t.Fatal("noise PCG stream coupled to time draws")
		}
	}
	a, b := newRunFixture(t, 7), newRunFixture(t, 7)
	if !bytes.Equal(snapshotRunFixture(t, a), snapshotRunFixture(t, b)) {
		t.Fatal("fixed seed shuffle not deterministic")
	}
	order := a.RowOrder()
	order[0] = 999
	if a.RowOrder()[0] == 999 {
		t.Fatal("RowOrder aliases live state")
	}
}

func TestTrainingRunResumeSequence(t *testing.T) {
	continuous, resumed := newRunFixture(t, 5), newRunFixture(t, 5)
	for step := 0; step < 19; step++ {
		rowA, drawA := rowAdvance(t, continuous, step%4+1)
		rowB, drawB := rowAdvance(t, resumed, step%4+1)
		if rowA != rowB || len(drawA) != len(drawB) {
			t.Fatal("resumed row schedule changed")
		}
		for i := range drawA {
			if drawA[i] != drawB[i] {
				t.Fatalf("draw mismatch step=%d", step)
			}
		}
		if step%3 == 0 {
			data := snapshotRunFixture(t, resumed)
			var err error
			resumed, err = RestoreTrainingRunState(data, runIdentityFixture(), runBindingFixture(uint64(step+1)))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(data, snapshotRunFixture(t, resumed)) {
				t.Fatal("restore drew/reshuffled state")
			}
			clear(data) // caller-owned serialized data is not retained
		}
		if !bytes.Equal(snapshotRunFixture(t, continuous), snapshotRunFixture(t, resumed)) {
			t.Fatalf("state differs step=%d", step)
		}
	}
}

func TestTrainingRunTransactionBoundary(t *testing.T) {
	s := newRunFixture(t, 3)
	before := snapshotRunFixture(t, s)
	p, err := s.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.NoiseUint64(); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, snapshotRunFixture(t, s)) {
		t.Fatal("abandoned proposal changed parent")
	}
	q, err := s.PrepareRow()
	if err != nil {
		t.Fatal(err)
	}
	expectRunError(t, s.CommitRow(q, 2), "exactly one")
	if !bytes.Equal(before, snapshotRunFixture(t, s)) {
		t.Fatal("failed commit mutated parent")
	}
	wrong := newRunFixture(t, 3)
	expectRunError(t, wrong.CommitRow(q, 1), "wrong-parent")
	if err := s.CommitRow(q, 1); err != nil {
		t.Fatal(err)
	}
	expectRunError(t, s.CommitRow(p, 2), "stale")
	expectRunError(t, s.CommitRow(q, 2), "replayed")
	if _, err := q.TimeUint64(); err == nil {
		t.Fatal("committed proposal permits draws")
	}
	if _, err := q.Row(); err == nil {
		t.Fatal("committed proposal permits row")
	}
	for step := 2; step <= 3; step++ {
		rowAdvance(t, s, 1)
	}
	step, epoch, cursor := s.Progress()
	if step != 3 || epoch != 0 || cursor != 3 {
		t.Fatalf("epoch-end normal form %d/%d/%d", step, epoch, cursor)
	}
	data := snapshotRunFixture(t, s)
	r, err := RestoreTrainingRunState(data, runIdentityFixture(), runBindingFixture(3))
	if err != nil {
		t.Fatal(err)
	}
	_, epoch, cursor = r.Progress()
	if epoch != 0 || cursor != 3 {
		t.Fatal("decode normalized epoch by drawing")
	}
	rowAdvance(t, r, 1)
	step, epoch, cursor = r.Progress()
	if step != 4 || epoch != 1 || cursor != 1 {
		t.Fatal("rollover boundary changed")
	}
	expectRunError(t, r.SnapshotErrorForTest(runBindingFixture(3)), "checkpoint step")
}

// Keep snapshot error extraction local to tests; no production wrapper API.
func (s *TrainingRunState) SnapshotErrorForTest(b TrainingCheckpointBinding) error {
	_, err := s.Snapshot(b)
	return err
}

func TestTrainingRunBindingIdentity(t *testing.T) {
	s := newRunFixture(t, 2)
	rowAdvance(t, s, 1)
	data := snapshotRunFixture(t, s)
	for _, mutate := range []func(*TrainingRunIdentity){
		func(i *TrainingRunIdentity) { i.GOARCH = "arm64" }, func(i *TrainingRunIdentity) { i.GoVersion = "go1.27.0" },
		func(i *TrainingRunIdentity) { i.SIMDMode = "scalar" }, func(i *TrainingRunIdentity) { i.CorpusSHA256 = strings.Repeat("9", 64) },
		func(i *TrainingRunIdentity) { i.OptimizerSHA256 = strings.Repeat("9", 64) }, func(i *TrainingRunIdentity) { i.OptionsSHA256 = strings.Repeat("9", 64) },
	} {
		id := runIdentityFixture()
		mutate(&id)
		_, err := RestoreTrainingRunState(data, id, runBindingFixture(1))
		expectRunError(t, err, "identity mismatch")
	}
	for _, binding := range []TrainingCheckpointBinding{{Name: "../model.bin", SHA256: strings.Repeat("8", 64), Step: 1}, {Name: "/model.bin", SHA256: strings.Repeat("8", 64), Step: 1}, {Name: "model\\x.bin", SHA256: strings.Repeat("8", 64), Step: 1}, runBindingFixture(2), {Name: "other.bin", SHA256: strings.Repeat("9", 64), Step: 1}} {
		if _, err := RestoreTrainingRunState(data, runIdentityFixture(), binding); err == nil {
			t.Fatal("mismatched/escaping binding admitted")
		}
	}
	if !bytes.Equal(data, snapshotRunFixture(t, s)) {
		t.Fatal("failed restore mutated parent")
	}
}

func mutateRunJSON(t *testing.T, data []byte, key string, value json.RawMessage) []byte {
	t.Helper()
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatal(err)
	}
	m[key] = value
	out, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return out
}
func TestTrainingRunStrictDecode(t *testing.T) {
	s := newRunFixture(t, 3)
	data := snapshotRunFixture(t, s)
	for _, tc := range []struct{ key, value string }{
		{"rows", "null"}, {"rows", "1.0"}, {"rows", "1e0"}, {"rows", "-1"}, {"rows", "100001"},
		{"step", "18446744073709551616"}, {"epoch", "null"}, {"cursor", "1"}, {"order", "[0,0,2]"},
		{"order", "[0,1,null]"}, {"order", "[0,1,3]"}, {"order", "[0,1,2,0]"},
		{"order_rng", "\"0000000000000000000000000000000000000000\""}, {"noise_rng", "\"short\""},
		{"boundary", "\"pending-gradient-accumulation\""}, {"algorithm", "\"other\""},
		{"identity", "null"}, {"checkpoint", "null"}, {"unknown", "0"},
	} {
		bad := mutateRunJSON(t, data, tc.key, json.RawMessage(tc.value))
		if _, err := RestoreTrainingRunState(bad, runIdentityFixture(), runBindingFixture(0)); err == nil {
			t.Fatalf("invalid %s=%s admitted", tc.key, tc.value)
		}
	}
	for _, bad := range [][]byte{data[:len(data)-1], append(append([]byte(nil), data...), []byte(" {}")...), bytes.Replace(data, []byte(`"version":1`), []byte(`"version":1,"version":1`), 1), bytes.Replace(data, []byte(`"goos":"linux"`), []byte(`"goos":null`), 1), bytes.Replace(data, []byte(`"step":0`), []byte(`"step":0,"step":0`), 1), bytes.Repeat([]byte("x"), MaxTrainingRunStateBytes+1)} {
		if _, err := RestoreTrainingRunState(bad, runIdentityFixture(), runBindingFixture(0)); err == nil {
			t.Fatal("malformed/duplicate/oversized state admitted")
		}
	}
}

func TestTrainingRunBoundsAndStepZero(t *testing.T) {
	for _, rows := range []int{0, -1, MaxTrainingRunRows + 1} {
		if _, err := NewTrainingRunState(runIdentityFixture(), rows, [6]uint64{}); err == nil {
			t.Fatal("row bound")
		}
	}
	id := runIdentityFixture()
	id.Backend = "vulkan"
	if _, err := NewTrainingRunState(id, 1, [6]uint64{}); err == nil {
		t.Fatal("GPU run-state implied support")
	}
	s := newRunFixture(t, 1)
	expectRunError(t, s.SnapshotErrorForTest(runBindingFixture(4)), "checkpoint step") // cannot silently start from model step4
	// Private boundary test avoids a billions-step loop; consistent MaxUint64
	// normal form must reject proposal before overflowing the step counter.
	s.core.step = math.MaxUint64
	s.core.epoch = math.MaxUint64 - 1
	s.core.cursor = 1
	if _, err := s.PrepareRow(); err == nil {
		t.Fatal("step overflow")
	}
	var nilState *TrainingRunState
	if _, err := nilState.PrepareRow(); err == nil {
		t.Fatal("nil prepare")
	}
	if _, err := nilState.Snapshot(runBindingFixture(0)); err == nil {
		t.Fatal("nil snapshot")
	}
	if err := nilState.CommitRow(nil, 1); err == nil {
		t.Fatal("nil commit")
	}
}
