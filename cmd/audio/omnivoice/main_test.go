package main

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestAudioNativeWAV(t *testing.T) {
	// Three seconds at 16 kHz exercises go-264 resampling to 24 kHz.
	const frames = 48000
	data := make([]byte, 44+frames*2)
	copy(data, "RIFF")
	binary.LittleEndian.PutUint32(data[4:], uint32(len(data)-8))
	copy(data[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(data[16:], 16)
	binary.LittleEndian.PutUint16(data[20:], 1)
	binary.LittleEndian.PutUint16(data[22:], 1)
	binary.LittleEndian.PutUint32(data[24:], 16000)
	binary.LittleEndian.PutUint32(data[28:], 32000)
	binary.LittleEndian.PutUint16(data[32:], 2)
	binary.LittleEndian.PutUint16(data[34:], 16)
	copy(data[36:], "data")
	binary.LittleEndian.PutUint32(data[40:], frames*2)
	for i := 0; i < frames; i++ {
		binary.LittleEndian.PutUint16(data[44+i*2:], uint16(int16((i%100)*100-5000)))
	}
	path := filepath.Join(t.TempDir(), "reference.wav")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"-mode", "audio", "-reference", path, "-threads", "1"}); err != nil {
		t.Fatal(err)
	}
}
func TestCLIRejectsInvalid(t *testing.T) {
	for _, args := range [][]string{{}, {"-mode", "speak"}, {"-threads", "0"}, {"-mode", "audio"}} {
		if err := run(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

func TestCLIResidentBudgetValidation(t *testing.T) {
	for _, args := range [][]string{{"-resident-mib", "-1"}, {"-resident-mib", "65537"}, {"-resident-mib", "1", "-mode", "audio"}, {"-resident-mib", "1", "-mode", "plan-chunks"}} {
		if err := run(args); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
