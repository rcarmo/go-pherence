package tokenizer

import (
	"math/rand"
	"slices"
	"strings"
	"sync"
	"testing"
)

var byteLevelBenchSink []int

type byteLevelFixture struct {
	tok *Tokenizer
	sym [256]string
}

func newByteLevelFixture(mode byteLevelPretokenizer) byteLevelFixture {
	enc := getByteEncoder()
	vocab := make(map[string]int, 256+16)
	inv := make(map[int]string, 256+16)
	var sym [256]string
	for b := 0; b < 256; b++ {
		tok := string(enc[byte(b)])
		sym[b] = tok
		id := b + 1
		vocab[tok] = id
		inv[id] = tok
	}
	fixture := byteLevelFixture{sym: sym}
	nextID := 1000
	add := func(token string) int {
		if id, ok := vocab[token]; ok {
			return id
		}
		nextID++
		vocab[token] = nextID
		inv[nextID] = token
		return nextID
	}

	add(fixture.token('a', 'b'))
	add(fixture.token('1', '2'))
	add(fixture.token('a', 'a'))
	add(fixture.token('h', 'e'))
	add(fixture.token('h', 'e', 'l'))
	add(fixture.token('h', 'e', 'l', 'l'))
	add(fixture.token(' ', 'w'))
	add(fixture.token(' ', 'w', 'o'))

	merges := [][2]string{
		{fixture.token('a'), fixture.token('a')},
		{fixture.token('h'), fixture.token('e')},
		{fixture.token('h', 'e'), fixture.token('l')},
		{fixture.token('h', 'e', 'l'), fixture.token('l')},
		{fixture.token(' '), fixture.token('w')},
		{fixture.token(' ', 'w'), fixture.token('o')},
		{fixture.token('~'), fixture.token('~')},
	}

	fixture.tok = &Tokenizer{
		Vocab:         vocab,
		InvVocab:      inv,
		Merges:        merges,
		AddedSpecial:  map[string]int{},
		AddedTokens:   map[string]int{},
		byteLevelMode: mode,
	}
	return fixture
}

func (f byteLevelFixture) token(bs ...byte) string {
	var b strings.Builder
	for _, by := range bs {
		b.WriteString(f.sym[by])
	}
	return b.String()
}

func refMergeRanks(merges [][2]string) map[[2]string]int {
	ranks := make(map[[2]string]int, len(merges))
	for i, merge := range merges {
		ranks[merge] = i
	}
	return ranks
}

func refPieceSymbols(piece string) []string {
	enc := getByteEncoder()
	symbols := make([]string, 0, len(piece))
	for i := 0; i < len(piece); i++ {
		symbols = append(symbols, string(enc[piece[i]]))
	}
	return symbols
}

func refBPEPiece(vocab map[string]int, mergeRank map[[2]string]int, maxRank int, symbols []string) []int {
	if len(symbols) > 1 {
		if id, ok := vocab[strings.Join(symbols, "")]; ok {
			return []int{id}
		}
	}
	for len(symbols) >= 2 {
		bestRank := maxRank
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			rank, ok := mergeRank[[2]string{symbols[i], symbols[i+1]}]
			if ok && (rank < bestRank || (rank == bestRank && (bestIdx < 0 || i < bestIdx))) {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		next := make([]string, 0, len(symbols)-1)
		next = append(next, symbols[:bestIdx]...)
		next = append(next, merged)
		next = append(next, symbols[bestIdx+2:]...)
		symbols = next
	}
	ids := make([]int, 0, len(symbols))
	for _, symbol := range symbols {
		if id, ok := vocab[symbol]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func refEncodeByteLevel(tok *Tokenizer, text string) []int {
	if tok == nil || tok.Vocab == nil {
		return nil
	}
	pieces := splitWhitespaceRuns(tok.byteLevelPattern().FindAllString(text, -1))
	mergeRank := refMergeRanks(tok.Merges)
	var ids []int
	for _, piece := range pieces {
		symbols := refPieceSymbols(piece)
		if len(symbols) == 0 {
			continue
		}
		ids = append(ids, refBPEPiece(tok.Vocab, mergeRank, len(tok.Merges), symbols)...)
	}
	return ids
}

func byteLevelRandomCorpus(n int) []string {
	rng := rand.New(rand.NewSource(0x5eed1234))
	segments := []string{
		"",
		"ab",
		"aaa",
		"~~",
		"hello",
		" world",
		"12",
		"12345",
		" ",
		"\t",
		"\n",
		"\r\n",
		"é",
		"éclair",
		"漢字",
		"🙂",
		"!?.",
		string([]byte{0xff, 0xfe}),
		string([]byte{0x00, 'a', 0x7f}),
		string([]byte{0xc3, 0x28}),
		string([]byte{'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'}),
	}
	cases := make([]string, 0, n)
	for len(cases) < n {
		parts := 1 + rng.Intn(8)
		var b strings.Builder
		for i := 0; i < parts; i++ {
			b.WriteString(segments[rng.Intn(len(segments))])
		}
		cases = append(cases, b.String())
	}
	return cases
}

func checkByteLevelReferenceCase(t *testing.T, tok *Tokenizer, text string) {
	t.Helper()
	want := refEncodeByteLevel(tok, text)
	if got := tok.Encode(text); !slices.Equal(got, want) {
		t.Fatalf("Encode(%q / % x) = %v, want %v", text, []byte(text), got, want)
	}
}

func TestByteLevelManualSemantics(t *testing.T) {
	defaultFixture := newByteLevelFixture(byteLevelDefault)
	tok := defaultFixture.tok
	baseA := tok.Vocab[defaultFixture.token('a')]
	base1 := tok.Vocab[defaultFixture.token('1')]
	base2 := tok.Vocab[defaultFixture.token('2')]

	if got, want := tok.Encode("ab"), []int{tok.Vocab[defaultFixture.token('a', 'b')]}; !slices.Equal(got, want) {
		t.Fatalf("direct whole-piece lookup = %v, want %v", got, want)
	}
	if got, want := tok.Encode("aaa"), []int{tok.Vocab[defaultFixture.token('a', 'a')], baseA}; !slices.Equal(got, want) {
		t.Fatalf("leftmost repeated-pair merge = %v, want %v", got, want)
	}
	if got := tok.Encode("~~"); len(got) != 0 {
		t.Fatalf("unknown merged token = %v, want empty", got)
	}
	if got, want := tok.Encode("hello"), []int{tok.Vocab[defaultFixture.token('h', 'e', 'l', 'l')], tok.Vocab[defaultFixture.token('o')]}; !slices.Equal(got, want) {
		t.Fatalf("merge chain = %v, want %v", got, want)
	}
	if got, want := tok.Encode("12"), []int{tok.Vocab[defaultFixture.token('1', '2')]}; !slices.Equal(got, want) {
		t.Fatalf("default digit run tokenization = %v, want %v", got, want)
	}

	singleDigitFixture := newByteLevelFixture(byteLevelQwenSingleDigits)
	if got, want := singleDigitFixture.tok.Encode("12"), []int{base1, base2}; !slices.Equal(got, want) {
		t.Fatalf("single-digit pretokenization = %v, want %v", got, want)
	}
}

func TestByteLevelReferenceParityFixedCorpus(t *testing.T) {
	fixed := []string{
		"",
		"ab",
		"aaa",
		"~~~~",
		"hello",
		"hello world",
		" hello",
		"a  b",
		"a\tb",
		"12",
		"123",
		"1 23 456",
		"\t12\taaa\n",
		" \r\n\t ",
		"éclair",
		"café",
		"漢字",
		"🙂",
		string([]byte{0xff, 0xfe, 'a', 'b'}),
		string([]byte{'a', 0x00, 'b', '~', '~'}),
		string([]byte{'h', 'e', 'l', 'l', 'o', ' ', 0xc3, 0xa9}),
		string([]byte{0xe2, 0x82, 0xac, '1', '2', 0xf0, 0x9f, 0x99, 0x82}),
		string([]byte{0xc3, 0x28, 'a', '\n'}),
	}
	for _, mode := range []byteLevelPretokenizer{byteLevelDefault, byteLevelQwenSingleDigits} {
		fixture := newByteLevelFixture(mode)
		for _, text := range fixed {
			checkByteLevelReferenceCase(t, fixture.tok, text)
		}
	}
}

func TestByteLevelReferenceParityRandomCorpus(t *testing.T) {
	corpus := byteLevelRandomCorpus(96)
	for _, mode := range []byteLevelPretokenizer{byteLevelDefault, byteLevelQwenSingleDigits} {
		fixture := newByteLevelFixture(mode)
		for _, text := range corpus {
			checkByteLevelReferenceCase(t, fixture.tok, text)
		}
	}
}

func TestByteLevelConcurrentEncodeResultOwnership(t *testing.T) {
	fixture := newByteLevelFixture(byteLevelDefault)
	input := strings.Repeat("aaa hello 12 ~~ café ", 32) + string([]byte{0xff, 0xfe, 'a', 'b'})
	want := refEncodeByteLevel(fixture.tok, input)

	const goroutines = 24
	results := make([][]int, goroutines)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			results[i] = fixture.tok.Encode(input)
		}(i)
	}
	close(start)
	wg.Wait()

	for i, got := range results {
		if !slices.Equal(got, want) {
			t.Fatalf("concurrent result %d = %v, want %v", i, got, want)
		}
	}
	for i := range results {
		if len(results[i]) == 0 {
			continue
		}
		snapshots := make([][]int, len(results))
		for j := range results {
			if j == i {
				continue
			}
			snapshots[j] = slices.Clone(results[j])
		}
		old := results[i][0]
		results[i][0] = -1000 - i
		for j := range results {
			if j == i {
				continue
			}
			if !slices.Equal(results[j], snapshots[j]) {
				t.Fatalf("results %d and %d share backing storage", i, j)
			}
		}
		results[i][0] = old
	}
}

func benchmarkByteLevelEncode(b *testing.B, input string) {
	fixture := newByteLevelFixture(byteLevelDefault)
	tok := fixture.tok
	for i := 0; i < 16; i++ {
		byteLevelBenchSink = tok.Encode(input)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var got []int
	for b.Loop() {
		got = tok.Encode(input)
	}
	byteLevelBenchSink = got
}

func TestByteLevelWarmAllocationBudget(t *testing.T) {
	fixture := newByteLevelFixture(byteLevelQwenSingleDigits)
	fixture.tok.initMergeRank()
	symbols := refPieceSymbols("hello")
	scratch := make([]string, len(symbols))
	ids := make([]int, 0, len(symbols))
	// Isolate BPE from regexp's sync.Pool, which randomly drops entries under
	// -race. One joined lookup and three merged strings; no slice per merge.
	if n := testing.AllocsPerRun(20, func() {
		copy(scratch, symbols)
		byteLevelBenchSink = fixture.tok.bpeMerge(scratch, fixture.tok.mergeRank, ids[:0])
	}); n != 4 {
		t.Fatalf("BPE allocations=%g want 4", n)
	}
}

func BenchmarkByteLevelEncodeShortSentence(b *testing.B) {
	benchmarkByteLevelEncode(b, "hello ab aaa 123 café")
}

func BenchmarkByteLevelEncodeRepeatedLongInput(b *testing.B) {
	benchmarkByteLevelEncode(b, strings.Repeat("hello world aaa 123 café ~~ ", 128))
}

func BenchmarkByteLevelEncodeMergeHeavyTokens(b *testing.B) {
	benchmarkByteLevelEncode(b, strings.Repeat("aaa hello aaa hello ", 256))
}
