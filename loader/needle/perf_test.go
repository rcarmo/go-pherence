package needle

import (
	"strings"
	"testing"
)

func BenchmarkNeedleArchiveParse(b *testing.B) {
	data := archiveFixture(b)
	b.ReportAllocs()
	b.SetBytes(int64(len(data)))
	for b.Loop() {
		if _, err := ParseArchive(data); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkNeedleTokenizer(b *testing.B) {
	tok, _ := tokenizerFixture(b)
	text := strings.Repeat("hello é 🚀 <|im_start|> abc ", 32)
	b.ReportAllocs()
	b.SetBytes(int64(len(text)))
	for b.Loop() {
		ids, err := tok.Encode(text)
		if err != nil {
			b.Fatal(err)
		}
		if _, err = tok.Decode(ids); err != nil {
			b.Fatal(err)
		}
	}
}
