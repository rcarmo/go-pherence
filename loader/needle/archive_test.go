package needle

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func archiveFixture(t testing.TB) []byte {
	t.Helper()
	b, e := os.ReadFile("testdata/needle3.cact")
	if e != nil {
		t.Fatal(e)
	}
	return b
}
func TestArchiveUpstream(t *testing.T) {
	b := archiveFixture(t)
	a, e := ParseArchive(b)
	if e != nil {
		t.Fatal(e)
	}
	refBytes, e := os.ReadFile("testdata/archive-reference.json")
	if e != nil {
		t.Fatal(e)
	}
	var ref struct {
		Records []struct {
			Shape  []int
			Data   []float32
			RawHex string `json:"raw_hex"`
		}
	}
	if e = json.Unmarshal(refBytes, &ref); e != nil {
		t.Fatal(e)
	}
	if len(a.Records) != len(ref.Records) {
		t.Fatal("record count")
	}
	for i, w := range ref.Records {
		got := a.Records[i]
		if w.RawHex != "" {
			if len(got.Raw)*2 != len(w.RawHex) {
				t.Fatal("raw bytes")
			}
			continue
		}
		if !reflect.DeepEqual(got.Shape, w.Shape) || len(got.Data) != len(w.Data) {
			t.Fatalf("shape record %d", i)
		}
		for j, v := range got.Data {
			if math.Abs(float64(v-w.Data[j])) > 1e-6+1e-5*math.Abs(float64(w.Data[j])) {
				t.Fatalf("record %d value %d got %g want %g", i, j, v, w.Data[j])
			}
		}
	}
	for i := range b {
		b[i] = 0
	}
	if a.Records[len(a.Records)-1].Raw[0] == 0 {
		t.Fatal("raw aliases input")
	}
}
func TestArchiveMalformed(t *testing.T) {
	base := archiveFixture(t)
	dir := 196 + 28*4
	mutations := map[string]func([]byte){"tag": func(b []byte) { b[0] = 0 }, "count": func(b []byte) { binary.LittleEndian.PutUint32(b[4:], 1<<31) }, "rank": func(b []byte) { b[dir+1] = 5 }, "offset wrap": func(b []byte) { binary.LittleEndian.PutUint64(b[dir+20:], math.MaxUint64-63) }, "metadata overlap": func(b []byte) { binary.LittleEndian.PutUint64(b[dir+20:], 64) }, "oversize": func(b []byte) { binary.LittleEndian.PutUint32(b[dir+4:], 1<<30) }, "codebook": func(b []byte) { binary.LittleEndian.PutUint32(b[196:], 0x7fc00000) }, "overlap": func(b []byte) { copy(b[dir+44+20:dir+44+28], b[dir+20:dir+28]) }, "group": func(b []byte) { binary.LittleEndian.PutUint32(b[dir+36:], 0) }, "padding": func(b []byte) { b[dir+2] = 1 }}
	for name, mut := range mutations {
		t.Run(name, func(t *testing.T) {
			b := append([]byte(nil), base...)
			mut(b)
			if _, e := ParseArchive(b); e == nil {
				t.Fatal("accepted invalid archive")
			}
		})
	}
	for _, n := range []int{0, 195, 300, len(base) - 1} {
		if _, e := ParseArchive(base[:n]); e == nil {
			t.Fatalf("accepted truncated %d", n)
		}
	}
}
func TestCQRecordWidths(t *testing.T) {
	cb := make([]float32, 28)
	for _, range_ := range [][2]int{{0, 4}, {4, 12}, {12, 28}} {
		for i := range_[0]; i < range_[1]; i++ {
			cb[i] = float32(i-range_[0]+1) / 100
		}
	}
	for _, bits := range []int{1, 2, 3, 4, 5} {
		size, _, e := archiveCQSizes(0, 1, 128, bits)
		if e != nil {
			t.Fatal(e)
		}
		blob := make([]byte, int(size))
		binary.LittleEndian.PutUint16(blob[len(blob)-2:], 0x3c00)
		out, e := decodeArchiveCQ(0, 1, 128, bits, blob, cb)
		if e != nil {
			t.Fatal(e)
		}
		level := float32(0)
		switch bits {
		case 1:
			level = -archiveBinaryLevel
		case 2:
			level = cb[0]
		case 3:
			level = cb[4]
		case 4:
			level = cb[12]
		}
		want := level * float32(math.Sqrt(128))
		if math.Abs(float64(out[0]-want)) > 1e-6 {
			t.Fatalf("bits %d got %g want %g", bits, out[0], want)
		}
		for _, v := range out[1:] {
			if v != 0 {
				t.Fatal("Walsh constant")
			}
		}
		if bits == 5 {
			blob[0] = 2
			if _, e = decodeArchiveCQ(0, 1, 128, bits, blob, cb); e == nil {
				t.Fatal("invalid ternary crumb")
			}
		}
	}
}
func FuzzParseArchive(f *testing.F) {
	f.Add(archiveFixture(f))
	f.Add([]byte{0, 1, 2})
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<20 {
			return
		}
		_, _ = ParseArchive(b)
	})
}
