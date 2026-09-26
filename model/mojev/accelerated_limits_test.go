package mojev

import (
	"context"
	"strings"
	"testing"

	cfg "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/model/qwen"
)

func acceleratedLimitsHead() *HeadWeights {
	return &HeadWeights{
		Width:             1024,
		Rank:              1,
		NormWeight:        make([]float32, 1024),
		NormBias:          make([]float32, 1024),
		ContextProjection: make([]float32, 1024),
		OptionProjection:  make([]float32, 1024),
	}
}

func acceleratedLimitsCPU(vocab, maxTokens int) *TextScorer {
	return &TextScorer{
		head:      acceleratedLimitsHead(),
		meta:      cfg.QwenNativeMTPMetadata{HiddenSize: 1024, VocabSize: vocab},
		embedding: make([]float32, vocab*1024),
		norm:      make([]float32, 1024),
		rope:      make([]float32, maxTokens*64),
		eps:       1e-6,
	}
}

func acceleratedLimitsIDs(n, id int) []int {
	ids := make([]int, n)
	for i := range ids {
		ids[i] = id
	}
	return ids
}

func acceleratedLimitsRow(stateLen, questionLen int, candidateLens ...int) EncodedRow {
	candidates := make([][]int, len(candidateLens))
	for i, n := range candidateLens {
		candidates[i] = acceleratedLimitsIDs(n, 0)
	}
	return EncodedRow{
		State:      acceleratedLimitsIDs(stateLen, 0),
		Questions:  [][]int{acceleratedLimitsIDs(questionLen, 0)},
		Candidates: [][][]int{candidates},
	}
}

func acceleratedLimitsQuestionRow(questionCount, candidateCount int) EncodedRow {
	questions := make([][]int, questionCount)
	candidates := make([][][]int, questionCount)
	for i := range questions {
		questions[i] = []int{0}
		candidates[i] = make([][]int, candidateCount)
		for j := range candidates[i] {
			candidates[i][j] = []int{0}
		}
	}
	return EncodedRow{State: []int{0}, Questions: questions, Candidates: candidates}
}

func acceleratedLimitsErr(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("want error containing %q, got %v", want, err)
	}
}

func TestAcceleratedLimitsValidateBranchLocalText(t *testing.T) {
	head := acceleratedLimitsHead()

	t.Run("exact4096", func(t *testing.T) {
		row := acceleratedLimitsRow(1, 1, 2047, 2047)
		if err := validateBranchLocalText(row, head); err != nil {
			t.Fatal(err)
		}
		if allocs := testing.AllocsPerRun(100, func() {
			if err := validateBranchLocalText(row, head); err != nil {
				panic(err)
			}
		}); allocs != 0 {
			t.Fatalf("validateBranchLocalText allocated %v times", allocs)
		}
	})

	t.Run("over4096", func(t *testing.T) {
		row := acceleratedLimitsRow(1, 1, 2048, 2047)
		acceleratedLimitsErr(t, validateBranchLocalText(row, head), "invalid branch-local text length")
	})

	t.Run("questions256", func(t *testing.T) {
		row := acceleratedLimitsQuestionRow(256, 2)
		if err := validateBranchLocalText(row, head); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("questions257", func(t *testing.T) {
		row := acceleratedLimitsQuestionRow(257, 2)
		acceleratedLimitsErr(t, validateBranchLocalText(row, head), "invalid branch-local text request")
	})

	t.Run("candidates64", func(t *testing.T) {
		row := acceleratedLimitsQuestionRow(1, 64)
		if err := validateBranchLocalText(row, head); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("candidates65", func(t *testing.T) {
		row := acceleratedLimitsQuestionRow(1, 65)
		acceleratedLimitsErr(t, validateBranchLocalText(row, head), "invalid candidate count")
	})

	t.Run("negativeID", func(t *testing.T) {
		row := acceleratedLimitsRow(1, 1, 1, 1)
		row.Candidates[0][1][0] = -1
		acceleratedLimitsErr(t, validateBranchLocalText(row, head), "negative token ID")
	})

	t.Run("empty", func(t *testing.T) {
		acceleratedLimitsErr(t, validateBranchLocalText(acceleratedLimitsRow(0, 1, 1, 1), head), "invalid branch-local text request")
		acceleratedLimitsErr(t, validateBranchLocalText(acceleratedLimitsRow(1, 0, 1, 1), head), "invalid branch-local text length")
		acceleratedLimitsErr(t, validateBranchLocalText(acceleratedLimitsRow(1, 1, 0, 1), head), "invalid branch-local text length")
	})
}

func TestAcceleratedLimitsSIMDScoreEncodedRejectsBeforeModelAccess(t *testing.T) {
	const maxTokens = 512
	s := &SIMDTextScorer{
		cpu:    acceleratedLimitsCPU(2, maxTokens),
		branch: &qwen.Qwen35SIMDBranch{},
		rows:   make([][]float32, maxTokens),
		hidden: make([]float32, maxTokens*1024),
	}
	if len(s.rows) != maxTokens || cap(s.rows) != maxTokens {
		t.Fatalf("rows geometry = len %d cap %d", len(s.rows), cap(s.rows))
	}

	t.Run("tokenOutsideVocabulary", func(t *testing.T) {
		row := acceleratedLimitsRow(1, 1, 1, 1)
		row.Candidates[0][1][0] = 2
		got, err := s.ScoreEncoded(row)
		if got != nil {
			t.Fatal("unexpected SIMD logits")
		}
		acceleratedLimitsErr(t, err, "token outside vocabulary")
		if strings.Contains(err.Error(), "qwen:") {
			t.Fatalf("unexpected model-path error: %v", err)
		}
	})

	t.Run("branchExceedsCapacity", func(t *testing.T) {
		row := acceleratedLimitsRow(1, 1, 511, 1)
		got, err := s.ScoreEncoded(row)
		if got != nil {
			t.Fatal("unexpected SIMD logits")
		}
		acceleratedLimitsErr(t, err, "branch exceeds SIMD capacity")
		if strings.Contains(err.Error(), "qwen:") {
			t.Fatalf("unexpected model-path error: %v", err)
		}
	})
}

func TestAcceleratedLimitsGPUScoreTreeRejectsBranchCapacity(t *testing.T) {
	g := &NVIDIATextScorer{cpu: acceleratedLimitsCPU(1, 512), maxTokens: 512}
	row := acceleratedLimitsRow(1, 1, 511, 1)
	got, err := g.scoreTree(context.Background(), row)
	if got != nil {
		t.Fatal("unexpected GPU logits")
	}
	acceleratedLimitsErr(t, err, "branch exceeds GPU token capacity")
}

// This exercises only the validated tree-metadata helper at the exact 512-token
// limit; it does not assert encoder/model parity.
func TestAcceleratedLimitsExact512GPUTreeMetadata(t *testing.T) {
	dst := make([]uint32, 512*4)
	if err := fillGPUTree(dst, 512, 1, 1, []int{257, 512}); err != nil {
		t.Fatal(err)
	}
	if dst[0] != ^uint32(0) || dst[1] != 0 || dst[2] != 0 || dst[3] != 1 {
		t.Fatalf("unexpected root metadata: %v", dst[:4])
	}
	if off := 2 * 4; dst[off] != 1 || dst[off+1] != 2 || dst[off+2] != 2 || dst[off+3] != 257 {
		t.Fatalf("unexpected first candidate metadata: %v", dst[off:off+4])
	}
	if off := 257 * 4; dst[off] != 1 || dst[off+1] != 2 || dst[off+2] != 257 || dst[off+3] != 512 {
		t.Fatalf("unexpected second candidate metadata: %v", dst[off:off+4])
	}
}
