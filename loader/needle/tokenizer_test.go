package needle

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func tokenizerFixture(t testing.TB) (*Tokenizer, []byte) {
	t.Helper()
	b, e := os.ReadFile("testdata/tokenizer.bin")
	if e != nil {
		t.Fatal(e)
	}
	tok, e := ParseTokenizer(b)
	if e != nil {
		t.Fatal(e)
	}
	return tok, b
}
func TestTokenizerUpstream(t *testing.T) {
	tok, _ := tokenizerFixture(t)
	b, e := os.ReadFile("testdata/archive-reference.json")
	if e != nil {
		t.Fatal(e)
	}
	var f struct {
		Tokenizer struct {
			Encode []struct {
				Text, Decoded string
				IDs           []int
			}
			Decode []struct {
				Text string
				IDs  []int
			}
		}
	}
	if e = json.Unmarshal(b, &f); e != nil {
		t.Fatal(e)
	}
	for _, c := range f.Tokenizer.Encode {
		ids, e := tok.Encode(c.Text)
		if e != nil {
			t.Fatal(e)
		}
		if !reflect.DeepEqual(ids, c.IDs) {
			t.Errorf("encode %q got %v want %v", c.Text, ids, c.IDs)
		}
		out, e := tok.Decode(ids)
		if e != nil || out != c.Decoded {
			t.Errorf("decode %q got %q want %q err %v", c.Text, out, c.Decoded, e)
		}
	}
	for _, c := range f.Tokenizer.Decode {
		out, e := tok.Decode(c.IDs)
		if e != nil || out != c.Text {
			t.Errorf("invalid bytes %v got %q want %q err %v", c.IDs, out, c.Text, e)
		}
	}
}
func TestTokenizerAdmission(t *testing.T) {
	tok, b := tokenizerFixture(t)
	for _, n := range []int{0, 23, 30, len(b) - 1} {
		if _, e := ParseTokenizer(b[:n]); e == nil {
			t.Error("accepted truncated")
		}
	}
	bad := append([]byte(nil), b...)
	bad[20] = 2
	if _, e := ParseTokenizer(bad); e == nil {
		t.Fatal("invalid flags")
	}
	if _, e := tok.Encode(string([]byte{0xff})); e == nil {
		t.Fatal("invalid UTF8")
	}
	if _, e := tok.Decode([]int{-1}); e == nil {
		t.Fatal("negative token")
	}
}
func FuzzParseTokenizer(f *testing.F) {
	_, b := tokenizerFixture(f)
	f.Add(b)
	f.Add([]byte{0})
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<20 {
			return
		}
		_, _ = ParseTokenizer(b)
	})
}

func TestTokenizerRejectsWrappedSpecialID(t *testing.T) {
	_, blob := tokenizerFixture(t)
	for _, off := range []int{4, 8, 12, 16} {
		b := append([]byte(nil), blob...)
		binary.LittleEndian.PutUint32(b[off:], 0xffffffff)
		if _, err := ParseTokenizer(b); err == nil {
			t.Fatal("wrapped special ID")
		}
	}
}
