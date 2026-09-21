package needle

import (
	"container/heap"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	tokenNormal  = 0
	tokenUnknown = 1
	tokenControl = 2
	tokenUser    = 3
	tokenByte    = 4
)

type tokenPiece struct {
	text  string
	score float32
	kind  byte
}

// Tokenizer reads the self-contained BPE dump exported in Needle3 archives. It
// follows upstream RefTokenizer, not arbitrary SentencePiece normalization.
type Tokenizer struct {
	pieces             []tokenPiece
	ids                map[string]int
	bytes              [256]int
	markers            map[rune][]string
	dummy, fallback    bool
	pad, eos, bos, unk int
}

func (t *Tokenizer) VocabSize() int                       { return len(t.pieces) }
func (t *Tokenizer) SpecialIDs() (pad, eos, bos, unk int) { return t.pad, t.eos, t.bos, t.unk }
func ParseTokenizer(blob []byte) (*Tokenizer, error) {
	if len(blob) < 24 || len(blob) > 32<<20 {
		return nil, fmt.Errorf("needle: invalid tokenizer size")
	}
	n := int(binary.LittleEndian.Uint32(blob))
	if n < 4 || n > 131072 {
		return nil, fmt.Errorf("needle: invalid piece count")
	}
	t := &Tokenizer{pieces: make([]tokenPiece, 0, n), ids: map[string]int{}, markers: map[rune][]string{}, pad: int(binary.LittleEndian.Uint32(blob[4:])), eos: int(binary.LittleEndian.Uint32(blob[8:])), bos: int(binary.LittleEndian.Uint32(blob[12:])), unk: int(binary.LittleEndian.Uint32(blob[16:]))}
	for _, id := range []int{t.pad, t.eos, t.bos, t.unk} {
		if id < 0 || id >= n {
			return nil, fmt.Errorf("needle: special ID outside vocabulary")
		}
	}
	if blob[20] > 1 || blob[21] > 1 || binary.LittleEndian.Uint16(blob[22:]) != 0 {
		return nil, fmt.Errorf("needle: invalid tokenizer flags")
	}
	t.dummy, t.fallback = blob[20] != 0, blob[21] != 0
	for i := range t.bytes {
		t.bytes[i] = -1
	}
	off := 24
	for i := 0; i < n; i++ {
		if off > len(blob)-7 {
			return nil, fmt.Errorf("needle: truncated tokenizer record")
		}
		score := math.Float32frombits(binary.LittleEndian.Uint32(blob[off:]))
		kind := blob[off+4]
		size := int(binary.LittleEndian.Uint16(blob[off+5:]))
		off += 7
		if size == 0 || size > len(blob)-off || kind > 4 || !isFinite32(score) {
			return nil, fmt.Errorf("needle: invalid tokenizer piece")
		}
		text := string(blob[off : off+size])
		off += size
		if !utf8.ValidString(text) {
			return nil, fmt.Errorf("needle: invalid piece UTF-8")
		}
		if _, exists := t.ids[text]; exists {
			return nil, fmt.Errorf("needle: duplicate piece")
		}
		t.ids[text] = i
		t.pieces = append(t.pieces, tokenPiece{text, score, kind})
		if kind == tokenByte {
			if len(text) != 6 || !strings.HasPrefix(text, "<0x") || text[5] != '>' {
				return nil, fmt.Errorf("needle: invalid byte piece")
			}
			b, e := strconv.ParseUint(text[3:5], 16, 8)
			if e != nil || t.bytes[b] >= 0 {
				return nil, fmt.Errorf("needle: invalid/duplicate byte piece")
			}
			t.bytes[b] = i
		}
		if kind == tokenUser {
			r, _ := utf8.DecodeRuneInString(text)
			t.markers[r] = append(t.markers[r], text)
		}
	}
	if off != len(blob) {
		return nil, fmt.Errorf("needle: tokenizer trailing bytes")
	}
	if t.pieces[t.unk].kind != tokenUnknown {
		return nil, fmt.Errorf("needle: unk ID is not an unknown piece")
	}
	if t.fallback {
		for _, id := range t.bytes {
			if id < 0 {
				return nil, fmt.Errorf("needle: incomplete byte fallback")
			}
		}
	}
	for r, markers := range t.markers { // longest marker wins, stable ID order breaks equal lengths
		for i := 1; i < len(markers); i++ {
			for j := i; j > 0 && utf8.RuneCountInString(markers[j]) > utf8.RuneCountInString(markers[j-1]); j-- {
				markers[j], markers[j-1] = markers[j-1], markers[j]
			}
		}
		t.markers[r] = markers
	}
	return t, nil
}

type mergeNode struct {
	start, end, prev, next, version int
	alive                           bool
}
type mergeCandidate struct {
	left, right, lv, rv, id int
	score                   float32
}
type mergeQueue []mergeCandidate

func (q mergeQueue) Len() int { return len(q) }
func (q mergeQueue) Less(i, j int) bool {
	if q[i].score != q[j].score {
		return q[i].score > q[j].score
	}
	return q[i].left < q[j].left
}
func (q mergeQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *mergeQueue) Push(v any)   { *q = append(*q, v.(mergeCandidate)) }
func (q *mergeQueue) Pop() any     { old := *q; v := old[len(old)-1]; *q = old[:len(old)-1]; return v }
func (t *Tokenizer) bpe(text string, out []int) []int {
	if text == "" {
		return out
	}
	nodes := make([]mergeNode, 0, utf8.RuneCountInString(text))
	for off, r := range text {
		nodes = append(nodes, mergeNode{start: off, end: off + utf8.RuneLen(r), prev: len(nodes) - 1, next: len(nodes) + 1, alive: true})
	}
	nodes[len(nodes)-1].next = -1
	q := mergeQueue{}
	add := func(left int) {
		if left < 0 {
			return
		}
		a := nodes[left]
		if !a.alive || a.next < 0 {
			return
		}
		b := nodes[a.next]
		if id, ok := t.ids[text[a.start:b.end]]; ok {
			heap.Push(&q, mergeCandidate{left, a.next, a.version, b.version, id, t.pieces[id].score})
		}
	}
	for i := range nodes {
		add(i)
	}
	for q.Len() > 0 {
		p := heap.Pop(&q).(mergeCandidate)
		a, b := &nodes[p.left], &nodes[p.right]
		if !a.alive || !b.alive || a.next != p.right || a.version != p.lv || b.version != p.rv {
			continue
		}
		a.end = b.end
		a.next = b.next
		a.version++
		b.alive = false
		b.version++
		if a.next >= 0 {
			nodes[a.next].prev = p.left
		}
		add(a.prev)
		add(p.left)
	}
	for i := 0; i >= 0; i = nodes[i].next {
		s := text[nodes[i].start:nodes[i].end]
		if id, ok := t.ids[s]; ok {
			out = append(out, id)
		} else if t.fallback {
			for j := range len(s) {
				out = append(out, t.bytes[s[j]])
			}
		} else {
			out = append(out, t.unk)
		}
	}
	return out
}
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t == nil {
		return nil, fmt.Errorf("needle: nil tokenizer")
	}
	if len(text) > 1<<20 || !utf8.ValidString(text) {
		return nil, fmt.Errorf("needle: text exceeds 1 MiB or invalid UTF-8")
	}
	if text == "" {
		return []int{}, nil
	}
	text = strings.ReplaceAll(text, " ", "▁")
	if t.dummy {
		text = "▁" + text
	}
	out := make([]int, 0, len(text)/2)
	start := 0
	for pos := 0; pos < len(text); {
		r, size := utf8.DecodeRuneInString(text[pos:])
		marker := ""
		for _, m := range t.markers[r] {
			if strings.HasPrefix(text[pos:], m) {
				marker = m
				break
			}
		}
		if marker != "" {
			out = t.bpe(text[start:pos], out)
			out = append(out, t.ids[marker])
			pos += len(marker)
			start = pos
		} else {
			pos += size
		}
	}
	return t.bpe(text[start:], out), nil
}
func (t *Tokenizer) Decode(ids []int) (string, error) {
	if t == nil {
		return "", fmt.Errorf("needle: nil tokenizer")
	}
	if len(ids) > 1<<20 {
		return "", fmt.Errorf("needle: too many tokens")
	}
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			return "", fmt.Errorf("needle: token ID out of range")
		}
		p := t.pieces[id]
		if b.Len() > 16<<20-len(p.text) {
			return "", fmt.Errorf("needle: decoded text exceeds 16 MiB")
		}
		switch p.kind {
		case tokenByte:
			v, _ := strconv.ParseUint(p.text[3:5], 16, 8)
			b.WriteByte(byte(v))
		case tokenControl, tokenUnknown:
		default:
			b.WriteString(p.text)
		}
	}
	text := decodeUTF8Replacement(b.String())
	text = strings.ReplaceAll(text, "▁", " ")
	if t.dummy && strings.HasPrefix(text, " ") {
		text = text[1:]
	}
	return text, nil
}

// Python's errors="replace" consumes a valid UTF-8 prefix of a malformed
// sequence as one replacement, but replaces invalid lead bytes individually.
func decodeUTF8Replacement(s string) string {
	if utf8.ValidString(s) {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		r, n := utf8.DecodeRuneInString(s[i:])
		if r != utf8.RuneError || n != 1 {
			b.WriteString(s[i : i+n])
			i += n
			continue
		}
		want := 0
		lead := s[i]
		switch {
		case lead >= 0xc2 && lead <= 0xdf:
			want = 2
		case lead >= 0xe0 && lead <= 0xef:
			want = 3
		case lead >= 0xf0 && lead <= 0xf4:
			want = 4
		}
		consumed := 1
		for j := 1; j < want && i+j < len(s); j++ {
			c := s[i+j]
			valid := c >= 0x80 && c <= 0xbf
			if j == 1 {
				if lead == 0xe0 {
					valid = c >= 0xa0 && c <= 0xbf
				}
				if lead == 0xed {
					valid = c >= 0x80 && c <= 0x9f
				}
				if lead == 0xf0 {
					valid = c >= 0x90 && c <= 0xbf
				}
				if lead == 0xf4 {
					valid = c >= 0x80 && c <= 0x8f
				}
			}
			if !valid {
				break
			}
			consumed++
		}
		b.WriteRune(utf8.RuneError)
		i += consumed
	}
	return b.String()
}

// PieceBytes returns literal continuation bytes, without dummy-prefix removal
// or UTF-8 replacement. This is required for token masks and incremental output.
// Control, unknown and user-defined special markers are not ordinary JSON text.
func (t *Tokenizer) PieceBytes(id int) ([]byte, bool) {
	if t == nil || id < 0 || id >= len(t.pieces) {
		return nil, false
	}
	p := t.pieces[id]
	switch p.kind {
	case tokenByte:
		b, err := strconv.ParseUint(p.text[3:5], 16, 8)
		if err != nil {
			return nil, false
		}
		return []byte{byte(b)}, true
	case tokenNormal:
		return []byte(strings.ReplaceAll(p.text, "▁", " ")), true
	default:
		return nil, false
	}
}

// MarkerID resolves an exact user-defined marker, without treating user text as
// a prompt template or performing normalization.
func (t *Tokenizer) MarkerID(marker string) (int, bool) {
	if t == nil {
		return 0, false
	}
	id, ok := t.ids[marker]
	return id, ok && t.pieces[id].kind == tokenUser
}
