package needle

import (
	"encoding/binary"
	"math"
	"reflect"
	"testing"
)

func TestParseArchivePackedRetainsValidatedCQWithoutDecodedData(t *testing.T) {
	bytes := archiveFixture(t)
	dense, err := ParseArchive(bytes)
	if err != nil {
		t.Fatal(err)
	}
	packed, err := ParseArchivePacked(bytes)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed.Records) != len(dense.Records) {
		t.Fatal("record count")
	}
	cqCount := 0
	for i, record := range packed.Records {
		reference := dense.Records[i]
		if record.DType == archiveDTypeCQ {
			cqCount++
			if record.Data != nil || len(record.CQBlob) == 0 || !reflect.DeepEqual(record.CQBlob, reference.CQBlob) {
				t.Fatalf("CQ record %d retains decoded data or lost owned blob", i)
			}
			got, err := DecodeCQRecord(i, record, packed.Codebook)
			if err != nil || !reflect.DeepEqual(got, reference.Data) {
				t.Fatalf("CQ record %d dense parity err=%v", i, err)
			}
		} else if !reflect.DeepEqual(record, reference) {
			t.Fatalf("non-CQ record %d changed", i)
		}
	}
	if cqCount == 0 {
		t.Fatal("fixture lacks CQ records")
	}
	if _, err := DecodeCQRecord(1, ArchiveRecord{DType: archiveDTypeFP32}, packed.Codebook); err == nil {
		t.Fatal("accepted non-CQ decode")
	}
	if _, err := DecodeCQRecord(0, ArchiveRecord{DType: archiveDTypeCQ, Shape: []int{1, 128}, Group: 128, Bits: 6, CQBlob: make([]byte, 1)}, packed.Codebook); err == nil {
		t.Fatal("accepted invalid CQ bits")
	}
	before := append([]byte(nil), packed.Records[0].CQBlob...)
	clear(bytes)
	if !reflect.DeepEqual(packed.Records[0].CQBlob, before) {
		t.Fatal("CQ blob aliases caller")
	}
}

func TestParseArchivePackedRejectsMalformedAndNonFiniteCQ(t *testing.T) {
	original := archiveFixture(t)
	if _, err := ParseArchivePacked(original[:len(original)-1]); err == nil {
		t.Fatal("accepted truncated archive")
	}
	directory := 196 + 28*4
	// Record 0 is a CQ embedding; mutate its packed norm without changing
	// layout. Both modes must reject invalid values before exposing records.
	offset := binary.LittleEndian.Uint64(original[directory+20:])
	rows := int(binary.LittleEndian.Uint32(original[directory+4:]))
	cols := int(binary.LittleEndian.Uint32(original[directory+8:]))
	bits := int(binary.LittleEndian.Uint32(original[directory+40:]))
	_, _, totalErr := archiveCQSizes(0, rows, cols, bits)
	if totalErr != nil {
		t.Fatal(totalErr)
	}
	groups := (cols + 127) / 128
	packedPerGroup := 128 * bits / 8
	if bits == 5 {
		packedPerGroup = 32
	}
	for name, bits16 := range map[string]uint16{"negative norm": 0xbc00, "nonfinite norm": 0x7c00} {
		t.Run(name, func(t *testing.T) {
			bytes := append([]byte(nil), original...)
			binary.LittleEndian.PutUint16(bytes[int(offset)+rows*groups*packedPerGroup:], bits16)
			if _, err := ParseArchivePacked(bytes); err == nil {
				t.Fatal("accepted invalid CQ norm")
			}
			if _, err := ParseArchive(bytes); err == nil {
				t.Fatal("decoded parser accepted invalid CQ norm")
			}
		})
	}
	if bits == 5 {
		bytes := append([]byte(nil), original...)
		bytes[int(offset)] = (bytes[int(offset)] &^ 3) | 2
		if _, err := ParseArchivePacked(bytes); err == nil {
			t.Fatal("accepted ternary crumb 2")
		}
	}
	for name, value := range map[string]uint32{"NaN codebook": 0x7fc00000, "infinite codebook": math.Float32bits(float32(math.Inf(1)))} {
		t.Run(name, func(t *testing.T) {
			bytes := append([]byte(nil), original...)
			binary.LittleEndian.PutUint32(bytes[196:], value)
			if _, err := ParseArchivePacked(bytes); err == nil {
				t.Fatal("accepted invalid codebook")
			}
		})
	}
}
