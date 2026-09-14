package numpy

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

func makeNPY(header string, version byte, payload []byte) []byte {
	header += "\n"
	data := []byte("\x93NUMPY")
	data = append(data, version, 0)
	if version == 1 {
		data = binary.LittleEndian.AppendUint16(data, uint16(len(header)))
	} else {
		data = binary.LittleEndian.AppendUint32(data, uint32(len(header)))
	}
	return append(append(data, []byte(header)...), payload...)
}
func archiveNPY(t *testing.T, names []string, values [][]byte, method uint16) []byte {
	t.Helper()
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	for i, name := range names {
		w, err := z.CreateHeader(&zip.FileHeader{Name: name, Method: method})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(values[i]); err != nil {
			t.Fatal(err)
		}
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}
func TestNumericNPZVersionsAndEndian(t *testing.T) {
	for _, v := range []byte{1, 2, 3} {
		for _, dtype := range []string{"<f4", ">f4", "<f8", ">f8"} {
			for _, method := range []uint16{zip.Store, zip.Deflate} {
				var order binary.AppendByteOrder = binary.LittleEndian
				if dtype[0] == '>' {
					order = binary.BigEndian
				}
				var payload []byte
				for _, value := range []float64{1.5, -2, .25, 0} {
					if dtype[2] == '4' {
						payload = order.AppendUint32(payload, math.Float32bits(float32(value)))
					} else {
						payload = order.AppendUint64(payload, math.Float64bits(value))
					}
				}
				header := fmt.Sprintf(`{"shape": (2, 2), 'fortran_order': False, 'descr': '%s', }`, dtype)
				npy := makeNPY(header, v, payload)
				data := archiveNPY(t, []string{"values.npy"}, [][]byte{npy}, method)
				result, err := ReadNPZ(context.Background(), bytes.NewReader(data), int64(len(data)), map[string][]int{"values": {2, 2}})
				if err != nil {
					t.Fatal(v, dtype, method, err)
				}
				want := []float64{1.5, -2, .25, 0}
				for i, value := range result["values"].Values {
					if value != want[i] {
						t.Fatal("decode mismatch")
					}
				}
				for i := range data {
					data[i] = 0
				}
				if result["values"].Values[0] != 1.5 {
					t.Fatal("source alias")
				}
			}
		}
	}
}
func TestNumericNPZRejectsUnsafeHeadersAndContainers(t *testing.T) {
	base := `{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }`
	payload := binary.LittleEndian.AppendUint64(nil, math.Float64bits(2))
	for _, kind := range []string{"object", "fortran", "structured", "native_endian", "rank3", "scalar", "zero_dim", "huge_dim", "expression", "unknown", "duplicate", "trailing", "missing", "version", "extent_short", "extent_long", "nan", "inf", "path", "name", "duplicate_member", "count", "crc", "huge_directory"} {
		header := base
		names := []string{"a.npy"}
		version := byte(1)
		p := append([]byte(nil), payload...)
		switch kind {
		case "object":
			header = strings.Replace(base, "<f8", "|O", 1)
		case "fortran":
			header = strings.Replace(base, "False", "True", 1)
		case "structured":
			header = strings.Replace(base, "'<f8'", "[('x', '<f8')]", 1)
		case "native_endian":
			header = strings.Replace(base, "<f8", "=f8", 1)
		case "rank3":
			header = strings.Replace(base, "(1,)", "(1, 1, 1)", 1)
		case "scalar":
			header = strings.Replace(base, "(1,)", "()", 1)
		case "zero_dim":
			header = strings.Replace(base, "(1,)", "(0,)", 1)
		case "huge_dim":
			header = strings.Replace(base, "(1,)", "(999999999999999999999,)", 1)
		case "expression":
			header = strings.Replace(base, "(1,)", "(1+0,)", 1)
		case "unknown":
			header = strings.Replace(base, "'shape'", "'shapes'", 1)
		case "duplicate":
			header = strings.Replace(base, "}", "'descr':'<f8'}", 1)
		case "trailing":
			header += "__import__('os')"
		case "missing":
			header = "{}"
		case "version":
			version = 4
		case "extent_short":
			p = p[:7]
		case "extent_long":
			p = append(p, 0)
		case "nan":
			p = binary.LittleEndian.AppendUint64(nil, math.Float64bits(math.NaN()))
		case "inf":
			p = binary.LittleEndian.AppendUint64(nil, math.Float64bits(math.Inf(1)))
		case "path":
			names[0] = "../a.npy"
		case "name":
			names[0] = "b.npy"
		}
		npy := makeNPY(header, version, p)
		values := [][]byte{npy}
		if kind == "duplicate_member" {
			names = append(names, "a.npy")
			values = append(values, npy)
		}
		if kind == "count" {
			for i := 1; i < 17; i++ {
				names = append(names, fmt.Sprintf("a%d.npy", i))
				values = append(values, npy)
			}
		}
		data := archiveNPY(t, names, values, zip.Store)
		if kind == "crc" {
			i := bytes.Index(data, []byte("PK\x01\x02"))
			data[i+16] ^= 0xff
		}
		if kind == "huge_directory" {
			i := bytes.LastIndex(data, []byte("PK\x05\x06"))
			binary.LittleEndian.PutUint16(data[i+8:], 60000)
			binary.LittleEndian.PutUint16(data[i+10:], 60000)
		}
		result, err := ReadNPZ(context.Background(), bytes.NewReader(data), int64(len(data)), map[string][]int{"a": {1}})
		if err == nil || result != nil {
			t.Fatal("accepted", kind)
		}
	}
	for _, shape := range [][]int{nil, {0}, {513}, {1, 1, 1}} {
		data := archiveNPY(t, []string{"a.npy"}, [][]byte{makeNPY(base, 1, payload)}, zip.Store)
		if out, err := ReadNPZ(context.Background(), bytes.NewReader(data), int64(len(data)), map[string][]int{"a": shape}); err == nil || out != nil {
			t.Fatal("bad schema")
		}
	}
}
func TestNumericNPZDirectoryAndHeaderBounds(t *testing.T) {
	npy := makeNPY(`{'descr':'<f8','shape':(1,),'fortran_order':False}`, 1, make([]byte, 8))
	data := archiveNPY(t, []string{"a.npy", "b.npy"}, [][]byte{npy, npy}, zip.Store)
	end := bytes.LastIndex(data, []byte("PK\x05\x06"))
	binary.LittleEndian.PutUint16(data[end+8:], 1)
	binary.LittleEndian.PutUint16(data[end+10:], 1)
	if err := boundedDirectory(bytes.NewReader(data), int64(len(data))); err == nil {
		t.Fatal("trusted false directory count")
	}
	long := makeNPY(strings.Repeat(" ", 4096), 1, nil)
	if _, _, _, err := readHeader(bytes.NewReader(long)); err == nil {
		t.Fatal("oversize header")
	}
	// Metadata declaring too much expanded data must fail before payload decode.
	data = archiveNPY(t, []string{"a.npy"}, [][]byte{npy}, zip.Store)
	central := bytes.Index(data, []byte("PK\x01\x02"))
	binary.LittleEndian.PutUint32(data[central+24:], 9<<20)
	if out, err := ReadNPZ(context.Background(), bytes.NewReader(data), int64(len(data)), map[string][]int{"a": {1}}); err == nil || out != nil {
		t.Fatal("expanded bound")
	}
}

func TestNumericNPZCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if out, err := ReadNPZ(ctx, nil, 22, map[string][]int{"a": {1}}); out != nil || err != context.Canceled {
		t.Fatal("cancellation", err)
	}
}
func FuzzNumericNPZ(f *testing.F) {
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	w, _ := z.Create("a.npy")
	w.Write(makeNPY(`{'descr':'<f8','shape':(1,),'fortran_order':False}`, 1, make([]byte, 8)))
	z.Close()
	f.Add(b.Bytes())
	f.Add([]byte("bad"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip()
		}
		result, err := ReadNPZ(context.Background(), bytes.NewReader(data), int64(len(data)), map[string][]int{"a": {1}})
		if err == nil && (len(result) != 1 || len(result["a"].Values) != 1) {
			t.Fatal("invalid successful result")
		}
	})
}
