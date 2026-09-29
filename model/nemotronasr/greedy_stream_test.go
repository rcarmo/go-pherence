package nemotronasr

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestGreedyRNNTStreamRejectsMalformed(t *testing.T) {
	if _, _, err := (*GreedyRNNTStream)(nil).Append(make([]float32, rnntHidden), 1); err == nil {
		t.Fatal("accepted nil stream")
	}
	if err := (*GreedyRNNTStream)(nil).Finish(); err == nil {
		t.Fatal("accepted nil finish")
	}
	s := &GreedyRNNTStream{Model: &GreedyRNNT{}}
	for _, tc := range []struct {
		data []float32
		rows int
	}{{nil, 0}, {make([]float32, rnntHidden), 2}, {make([]float32, (maxQualifiedRNNTFrames+1)*rnntHidden), maxQualifiedRNNTFrames + 1}, {append([]float32{float32(math.NaN())}, make([]float32, rnntHidden-1)...), 1}} {
		if _, _, err := s.Append(tc.data, tc.rows); err == nil || s.frames != 0 || s.state.initialized {
			t.Fatal("invalid chunk advanced predictor")
		}
	}
	if err := s.Finish(); err == nil {
		t.Fatal("accepted empty finish")
	}
	s.frames = math.MaxInt64
	if _, _, err := s.Append(make([]float32, rnntHidden), 1); err == nil || s.closed || s.state.initialized {
		t.Fatal("accepted frame count overflow or mutated state")
	}
}

func TestReleasedGreedyRNNTStreamJFKProjectedPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := LoadRNNTProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/asr_jfk_full_decode.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		EncoderRows int   `json:"encoder_rows"`
		Tokens      []int `json:"tokens"`
		Frames      []int `json:"frames"`
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.EncoderRows != 139 || len(ref.Tokens) != len(ref.Frames) {
		t.Fatal("invalid projected decode reference")
	}
	input := readStemFixture(t, "asr_jfk_full_encoder", ref.EncoderRows*rnntHidden)
	for _, chunkSize := range []int{1, 4, 17, 56, 139} {
		stream := &GreedyRNNTStream{Model: &GreedyRNNT{Decoder: decoder, Projection: projection}}
		var tokens []int
		var frames []int64
		for offset := 0; offset < ref.EncoderRows; offset += chunkSize {
			end := min(offset+chunkSize, ref.EncoderRows)
			gotTokens, gotFrames, err := stream.Append(input[offset*rnntHidden:end*rnntHidden], end-offset)
			if err != nil {
				t.Fatalf("chunk=%d offset=%d: %v", chunkSize, offset, err)
			}
			tokens = append(tokens, gotTokens...)
			frames = append(frames, gotFrames...)
		}
		if err := stream.Finish(); err != nil {
			t.Fatal(err)
		}
		if _, _, err := stream.Append(input[:rnntHidden], 1); err == nil {
			t.Fatal("accepted append after finish")
		}
		if err := stream.Finish(); err == nil {
			t.Fatal("accepted second finish")
		}
		if !reflect.DeepEqual(tokens, ref.Tokens) || len(frames) != len(ref.Frames) {
			t.Fatalf("chunk=%d token or frame count mismatch", chunkSize)
		}
		for i, frame := range frames {
			if frame != int64(ref.Frames[i]) {
				t.Fatalf("chunk=%d decision=%d frame=%d want=%d", chunkSize, i, frame, ref.Frames[i])
			}
		}
		t.Logf("chunk=%d rows=%d decisions=%d nonblank=%d", chunkSize, ref.EncoderRows, len(tokens), countNonblank(tokens))
	}
}

func countNonblank(tokens []int) int {
	n := 0
	for _, v := range tokens {
		if v != rnntBlank {
			n++
		}
	}
	return n
}
