package needle

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"slices"
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
	v2, err := os.ReadFile("testdata/needle2.cact")
	if err != nil {
		f.Fatal(err)
	}
	f.Add(v2)
	f.Add([]byte{0, 1, 2})
	f.Fuzz(func(t *testing.T, b []byte) {
		if len(b) > 1<<20 {
			return
		}
		_, _ = ParseArchive(b)
	})
}

func TestCQBlobOwned(t *testing.T) {
	bytes := archiveFixture(t)
	a, err := ParseArchive(bytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Records[0].CQBlob) == 0 {
		t.Fatal("missing CQ bytes")
	}
	saved := append([]byte(nil), a.Records[0].CQBlob...)
	clear(bytes)
	if !reflect.DeepEqual(a.Records[0].CQBlob, saved) {
		t.Fatal("CQ blob aliases caller bytes")
	}
}

func TestNeedle2ArchiveRecords(t *testing.T) {
	var ref struct {
		Records []struct {
			Name  string    `json:"name"`
			Shape []int     `json:"shape"`
			Data  []float32 `json:"data"`
		} `json:"records"`
	}
	raw, err := os.ReadFile("testdata/needle2-archive-reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/needle2.cact")
	if err != nil {
		t.Fatal(err)
	}
	a, err := ParseArchive(data)
	if err != nil {
		t.Fatal(err)
	}
	if a.Header[0] != archiveTagV2 || len(a.Records) != len(ref.Records)+1 {
		t.Fatal("v2 record count")
	}
	for i, w := range ref.Records {
		got := a.Records[i]
		if !slices.Equal(got.Shape, w.Shape) || len(got.Data) != len(w.Data) {
			t.Fatal("shape " + w.Name)
		}
		for j, x := range got.Data {
			if math.Abs(float64(x-w.Data[j])) > 1e-6+1e-5*math.Abs(float64(w.Data[j])) {
				t.Fatalf("%s[%d] %g != %g", w.Name, j, x, w.Data[j])
			}
		}
	}
	before := append([]float32(nil), a.Records[0].Data...)
	clear(data)
	if !slices.Equal(before, a.Records[0].Data) {
		t.Fatal("input alias")
	}
}

func TestNeedle2ArchiveMalformed(t *testing.T) {
	base, err := os.ReadFile("testdata/needle2.cact")
	if err != nil {
		t.Fatal(err)
	}
	dir := 20 + 28*4
	cases := map[string]func([]byte){
		"count":    func(b []byte) { binary.LittleEndian.PutUint32(b[4:], 4097) },
		"codebook": func(b []byte) { binary.LittleEndian.PutUint32(b[8:], 27) },
		"window":   func(b []byte) { binary.LittleEndian.PutUint32(b[12:], 65537) },
		"kvbits":   func(b []byte) { binary.LittleEndian.PutUint32(b[16:], 7) },
		"rank":     func(b []byte) { b[dir+1] = 5 },
		"offset":   func(b []byte) { binary.LittleEndian.PutUint64(b[dir+20:], ^uint64(0)) },
		"overlap":  func(b []byte) { copy(b[dir+44+20:dir+44+28], b[dir+20:dir+28]) },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			b := append([]byte(nil), base...)
			mutate(b)
			if _, err := ParseArchive(b); err == nil {
				t.Fatal("malformed accepted")
			}
		})
	}
	for _, n := range []int{0, 4, 19, 20, dir, dir + 43, len(base) - 1} {
		if _, err := ParseArchive(base[:n]); err == nil {
			t.Fatalf("truncated at %d", n)
		}
	}
}
