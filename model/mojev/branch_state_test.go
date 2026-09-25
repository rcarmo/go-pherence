package mojev

import (
	"reflect"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/model/qwen"
)

// This tests the existing Qwen3.5 deep-clone primitive at the three-level
// MoJev fork boundaries. No model forward or attention-mask path runs here.
func TestQwen35BranchStateOwnership(t *testing.T) {
	ancestor := qwen.Qwen35BaseForwardState{
		FullK: [][]float32{make([]float32, 2, 8), nil},
		FullV: [][]float32{make([]float32, 2, 8), nil},
		Linear: []qwen.Qwen35LinearAttentionState{
			{}, {Conv: []float32{5, 6, 7}, SSM: []float32{8, 9}, Pos: 4},
		},
		Pos: 4,
	}
	ancestor.FullK[0][0], ancestor.FullK[0][1] = 1, 2
	ancestor.FullV[0][0], ancestor.FullV[0][1] = 3, 4
	original := qwen.CloneQwen35BaseForwardState(ancestor)
	questions := []qwen.Qwen35BaseForwardState{
		qwen.CloneQwen35BaseForwardState(ancestor),
		qwen.CloneQwen35BaseForwardState(ancestor),
	}
	if cap(questions[0].FullK[0]) != 8 || cap(questions[0].FullV[0]) != 8 {
		t.Fatal("full-attention spare capacity was lost")
	}
	// Mutating a question's scratch, including reslicing into spare K/V
	// capacity, must never mutate the shared ancestor or sibling question.
	questions[0].FullK[0] = append(questions[0].FullK[0], 10, 11)
	questions[0].FullV[0] = append(questions[0].FullV[0], 12, 13)
	questions[0].Linear[1].Conv[0] = 14
	questions[0].Linear[1].SSM[1] = 15
	questions[0].Linear[1].Pos, questions[0].Pos = 6, 6
	if !reflect.DeepEqual(ancestor, original) || questions[1].Pos != 4 || questions[1].FullK[0][0] != 1 || questions[1].Linear[1].Conv[0] != 5 {
		t.Fatal("question branch mutated ancestor or sibling")
	}
	// Fork two candidates after their own question is complete. The other
	// question cannot supply a state or observe a candidate's updates.
	candidates := []qwen.Qwen35BaseForwardState{
		qwen.CloneQwen35BaseForwardState(questions[0]),
		qwen.CloneQwen35BaseForwardState(questions[0]),
	}
	candidates[0].FullK[0][0] = 20
	candidates[0].FullV[0] = append(candidates[0].FullV[0], 21)
	candidates[0].Linear[1].Conv[1] = 22
	candidates[0].Linear[1].SSM[0] = 23
	candidates[0].Pos = 7
	if questions[0].FullK[0][0] != 1 || questions[0].Linear[1].Conv[1] != 6 || candidates[1].FullV[0][2] != 12 || candidates[1].Linear[1].SSM[0] != 8 || questions[1].FullK[0][0] != 1 {
		t.Fatal("candidate branch aliased a sibling or ancestor")
	}
}

func TestQwen35ConcurrentCandidateForks(t *testing.T) {
	question := qwen.Qwen35BaseForwardState{
		FullK:  [][]float32{make([]float32, 2, 64)},
		FullV:  [][]float32{make([]float32, 2, 64)},
		Linear: []qwen.Qwen35LinearAttentionState{{Conv: []float32{1, 2}, SSM: []float32{3, 4}, Pos: 2}},
		Pos:    2,
	}
	question.FullK[0][0], question.FullV[0][0] = 5, 6
	original := qwen.CloneQwen35BaseForwardState(question)
	var group sync.WaitGroup
	for n := 0; n < 8; n++ {
		group.Add(1)
		go func(n int) {
			defer group.Done()
			fork := qwen.CloneQwen35BaseForwardState(question)
			fork.FullK[0] = append(fork.FullK[0], float32(n+10))
			fork.FullV[0][0] = float32(n + 20)
			fork.Linear[0].Conv[0] = float32(n + 30)
			fork.Linear[0].SSM[1] = float32(n + 40)
			if fork.FullK[0][2] != float32(n+10) || fork.Linear[0].Conv[0] != float32(n+30) {
				t.Errorf("candidate fork %d lost its own state", n)
			}
		}(n)
	}
	group.Wait()
	if !reflect.DeepEqual(question, original) {
		t.Fatal("concurrent candidate forks mutated question state")
	}
}
