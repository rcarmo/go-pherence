package audio

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func minimalAuditWAV(channels, bits, format uint16, data []byte) []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+len(data)))
	b.WriteString("WAVEfmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, format)
	binary.Write(&b, binary.LittleEndian, channels)
	binary.Write(&b, binary.LittleEndian, uint32(16000))
	binary.Write(&b, binary.LittleEndian, uint32(16000)*uint32(channels)*uint32(bits/8))
	binary.Write(&b, binary.LittleEndian, channels*(bits/8))
	binary.Write(&b, binary.LittleEndian, bits)
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(len(data)))
	b.Write(data)
	return b.Bytes()
}
func TestWAVMalformedDoesNotPanicOrAcceptTruncation(t *testing.T) {
	cases := [][]byte{minimalAuditWAV(0, 16, 1, []byte{0, 0}), minimalAuditWAV(1, 0, 1, []byte{0, 0}), minimalAuditWAV(1, 16, 1, []byte{0, 0})[:45], minimalAuditWAV(1, 16, 1, []byte{0})}
	for i, data := range cases {
		func() {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("case%d panic: %v", i, p)
				}
			}()
			if _, _, err := ReadWAV(bytes.NewReader(data)); err == nil {
				t.Errorf("case%d malformed accepted", i)
			}
		}()
	}
}
func TestWAVOddAncillaryPadding(t *testing.T) {
	normal := minimalAuditWAV(1, 16, 1, []byte{0, 0})
	var data bytes.Buffer
	data.Write(normal[:12])
	data.WriteString("JUNK")
	binary.Write(&data, binary.LittleEndian, uint32(1))
	data.Write([]byte{'x', 0})
	data.Write(normal[12:])
	b := data.Bytes()
	binary.LittleEndian.PutUint32(b[4:], uint32(len(b)-8))
	samples, rate, err := ReadWAV(bytes.NewReader(b))
	if err != nil || rate != 16000 || len(samples) != 1 {
		t.Fatal("padding corrupted chunk alignment", rate, len(samples), err)
	}
}
