package needle

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"

	"github.com/rcarmo/go-pherence/half"
)

const (
	archiveTag             uint32 = 0x05E12A84
	archiveHeaderFields           = 48
	archiveHeaderBytes            = archiveHeaderFields*4 + 4 // <48If = 196 bytes.
	archiveRecordBytes            = 44
	archiveAlign                  = 64
	archiveMaxRecords             = 4096
	archiveCodebookLen            = 28
	archiveMaxFileBytes    int64  = 512 << 20
	archiveMaxDecodedBytes int64  = 512 << 20
	archiveMaxRawBytes     int64  = 32 << 20
	archiveCQGroup                = 128
	archiveTernaryBits            = 5

	archiveDTypeFP16 = 1
	archiveDTypeFP32 = 2
	archiveDTypeCQ   = 3
	archiveDTypeRAW  = 4
)

const (
	archiveHdrTag = iota
	archiveHdrNumTensors
	archiveHdrCodebookLen
	archiveHdrKVWindow
	archiveHdrKVBits
	archiveHdrVocab
	archiveHdrOutVocab
	archiveHdrDModel
	archiveHdrNumHeads
	archiveHdrNumKVHeads
	archiveHdrNumLayers
	archiveHdrQKHeadDim
	archiveHdrVHeadDim
	archiveHdrMaxSeq
	archiveHdrHadaN
	archiveHdrMHCLanes
	archiveHdrSlidingWindow
	archiveHdrGlobalMaskLo
	archiveHdrGlobalMaskHi
	archiveHdrQKVConvTaps
	archiveHdrEngramSlots
	archiveHdrEngramSubDim
	archiveHdrNumEngramTables
	archiveHdrEngramConvTaps
	archiveHdrEngramConvDilation
	archiveHdrEngramSeedHeads
	archiveHdrNumEngramOrders
	archiveHdrEngramOrders0
	archiveHdrEngramOrders1
	archiveHdrEngramOrders2
	archiveHdrEngramOrders3
	archiveHdrNumEngramSites
	archiveHdrEngramSites0
)

var (
	archiveWalshScale128 = float32(1 / math.Sqrt(archiveCQGroup))
	archiveBinaryLevel   = float32(math.Sqrt(2/math.Pi) / math.Sqrt(archiveCQGroup))
	archiveTernaryLevel  = float32(1.2240064 / math.Sqrt(archiveCQGroup))
)

// Archive is a raw Needle3 .cact archive.
type Archive struct {
	Header    [48]uint32
	RopeTheta float32
	Codebook  []float32
	Records   []ArchiveRecord
}

// ArchiveRecord is one nameless tensor payload from a .cact archive.
type ArchiveRecord struct {
	DType int
	Shape []int
	Group int
	Bits  int
	Data  []float32
	Raw   []byte
}

type archiveRecordSpec struct {
	index        int
	dtype        int
	shape        []int
	offset       int64
	nbytes       int64
	group        int
	bits         int
	decodedBytes int64
}

// LoadArchive reads and parses a Needle3 .cact archive.
func LoadArchive(path string) (*Archive, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("needle: open %s: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("needle: stat %s: %w", path, err)
	}
	size := fi.Size()
	if size < archiveHeaderBytes {
		return nil, fmt.Errorf("needle: archive too short: %d", size)
	}
	if size > archiveMaxFileBytes {
		return nil, fmt.Errorf("needle: archive size %d exceeds %d", size, archiveMaxFileBytes)
	}
	if size > int64(maxInt()) {
		return nil, fmt.Errorf("needle: archive size %d exceeds platform limit", size)
	}

	buf := make([]byte, int(size))
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, fmt.Errorf("needle: read %s: %w", path, err)
	}
	return ParseArchive(buf)
}

// ParseArchive parses a Needle3 .cact archive from memory.
func ParseArchive(data []byte) (*Archive, error) {
	if len(data) < archiveHeaderBytes {
		return nil, fmt.Errorf("needle: archive too short: %d", len(data))
	}
	if int64(len(data)) > archiveMaxFileBytes {
		return nil, fmt.Errorf("needle: archive size %d exceeds %d", len(data), archiveMaxFileBytes)
	}

	var a Archive
	for i := 0; i < archiveHeaderFields; i++ {
		a.Header[i] = binary.LittleEndian.Uint32(data[i*4:])
	}
	a.RopeTheta = math.Float32frombits(binary.LittleEndian.Uint32(data[archiveHeaderFields*4:]))
	if err := validateArchiveHeader(a.Header, a.RopeTheta); err != nil {
		return nil, err
	}

	numRecords := int(a.Header[archiveHdrNumTensors])
	codebookLen := int(a.Header[archiveHdrCodebookLen])
	codebookBytes, ok := checkedMulInt64(int64(codebookLen), 4)
	if !ok {
		return nil, fmt.Errorf("needle: archive codebook size overflow")
	}
	metadataStart := int64(archiveHeaderBytes)
	metadataEnd, ok := checkedAddInt64(metadataStart, codebookBytes)
	if !ok {
		return nil, fmt.Errorf("needle: archive metadata size overflow")
	}
	directoryBytes, ok := checkedMulInt64(int64(numRecords), archiveRecordBytes)
	if !ok {
		return nil, fmt.Errorf("needle: archive directory size overflow")
	}
	metadataEnd, ok = checkedAddInt64(metadataEnd, directoryBytes)
	if !ok {
		return nil, fmt.Errorf("needle: archive metadata size overflow")
	}
	if metadataEnd > int64(len(data)) {
		return nil, fmt.Errorf("needle: archive metadata truncated")
	}

	a.Codebook = make([]float32, codebookLen)
	codebookOffset := archiveHeaderBytes
	for i := range a.Codebook {
		a.Codebook[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[codebookOffset+i*4:]))
	}
	if err := validateArchiveCodebook(a.Codebook); err != nil {
		return nil, err
	}

	specs := make([]archiveRecordSpec, numRecords)
	var decodedTotal int64
	dirOffset := archiveHeaderBytes + codebookLen*4
	for i := 0; i < numRecords; i++ {
		off := dirOffset + i*archiveRecordBytes
		spec, err := parseArchiveRecordSpec(i, data[off:off+archiveRecordBytes], decodedTotal)
		if err != nil {
			return nil, err
		}
		var ok bool
		decodedTotal, ok = checkedAddInt64(decodedTotal, spec.decodedBytes)
		if !ok {
			return nil, fmt.Errorf("needle: decoded tensor bytes overflow")
		}
		if decodedTotal > archiveMaxDecodedBytes {
			return nil, fmt.Errorf("needle: decoded tensor bytes %d exceed %d", decodedTotal, archiveMaxDecodedBytes)
		}
		specs[i] = spec
	}
	if err := validateArchiveRecordLayout(specs, metadataEnd, int64(len(data))); err != nil {
		return nil, err
	}

	a.Records = make([]ArchiveRecord, numRecords)
	for i, spec := range specs {
		rec, err := decodeArchiveRecord(spec, data[spec.offset:spec.offset+spec.nbytes], a.Codebook)
		if err != nil {
			return nil, err
		}
		a.Records[i] = rec
	}
	return &a, nil
}

func validateArchiveHeader(h [48]uint32, ropeTheta float32) error {
	if h[archiveHdrTag] != archiveTag {
		return fmt.Errorf("needle: bad archive tag 0x%08x", h[archiveHdrTag])
	}
	if h[archiveHdrNumTensors] == 0 || h[archiveHdrNumTensors] > archiveMaxRecords {
		return fmt.Errorf("needle: archive tensor count %d exceeds %d", h[archiveHdrNumTensors], archiveMaxRecords)
	}
	if h[archiveHdrCodebookLen] != archiveCodebookLen {
		return fmt.Errorf("needle: archive codebook length %d want %d", h[archiveHdrCodebookLen], archiveCodebookLen)
	}
	if !isFinite32(ropeTheta) || ropeTheta <= 0 {
		return fmt.Errorf("needle: invalid rope_theta %g", ropeTheta)
	}

	kvBits := h[archiveHdrKVBits]
	if kvBits != 2 && kvBits != 3 && kvBits != 4 && kvBits != 8 {
		return fmt.Errorf("needle: unsupported kv_bits %d", kvBits)
	}

	vocab := h[archiveHdrVocab]
	outVocab := h[archiveHdrOutVocab]
	dModel := h[archiveHdrDModel]
	heads := h[archiveHdrNumHeads]
	kvHeads := h[archiveHdrNumKVHeads]
	layers := h[archiveHdrNumLayers]
	qkDim := h[archiveHdrQKHeadDim]
	vDim := h[archiveHdrVHeadDim]
	maxSeq := h[archiveHdrMaxSeq]
	hadaN := h[archiveHdrHadaN]
	lanes := h[archiveHdrMHCLanes]
	sliding := h[archiveHdrSlidingWindow]
	kvWindow := h[archiveHdrKVWindow]
	qkvConv := h[archiveHdrQKVConvTaps]
	if vocab < 4 || vocab > 131072 || outVocab > vocab || dModel < 2 || dModel > 4096 ||
		heads < 1 || heads > 128 || kvHeads < 1 || kvHeads > heads || heads%kvHeads != 0 ||
		layers < 1 || layers > 64 || qkDim < 2 || qkDim > 512 || qkDim%2 != 0 ||
		vDim < 1 || vDim > 512 || maxSeq < 1 || maxSeq > 65536 ||
		lanes < 1 || lanes > 8 || qkvConv > 32 ||
		sliding > maxSeq || kvWindow > maxSeq {
		return fmt.Errorf("needle: invalid or excessive archive geometry")
	}
	if hadaN < dModel || hadaN > 4096 || hadaN&(hadaN-1) != 0 {
		return fmt.Errorf("needle: invalid archive hada_n %d", hadaN)
	}
	mask := uint64(h[archiveHdrGlobalMaskLo]) | uint64(h[archiveHdrGlobalMaskHi])<<32
	if layers < 64 && mask>>layers != 0 {
		return fmt.Errorf("needle: global mask references layer >= %d", layers)
	}

	ordersN := int(h[archiveHdrNumEngramOrders])
	sitesN := int(h[archiveHdrNumEngramSites])
	if ordersN > 4 {
		return fmt.Errorf("needle: archive engram order count %d exceeds 4", ordersN)
	}
	if sitesN > 16 {
		return fmt.Errorf("needle: archive engram site count %d exceeds 16", sitesN)
	}
	for i := ordersN; i < 4; i++ {
		if h[archiveHdrEngramOrders0+i] != 0 {
			return fmt.Errorf("needle: archive engram order pad %d is nonzero", i)
		}
	}
	for i := sitesN; i < 16; i++ {
		if h[archiveHdrEngramSites0+i] != 0 {
			return fmt.Errorf("needle: archive engram site pad %d is nonzero", i)
		}
	}
	// Upstream emits engram geometry even when there are no active sites.
	if ordersN < 1 {
		return fmt.Errorf("needle: archive engram sites require at least one order")
	}
	engramSlots := h[archiveHdrEngramSlots]
	subDim := h[archiveHdrEngramSubDim]
	tables := h[archiveHdrNumEngramTables]
	convTaps := h[archiveHdrEngramConvTaps]
	dilation := h[archiveHdrEngramConvDilation]
	seedHeads := h[archiveHdrEngramSeedHeads]
	if engramSlots < 1 || engramSlots > 1<<20 || subDim < 1 || subDim > dModel || tables < 1 || tables > 512 || convTaps < 1 || convTaps > 32 || dilation < 1 || dilation > 32 || seedHeads > 128 {
		return fmt.Errorf("needle: invalid archive engram geometry")
	}
	maxOrder := uint32(0)
	for i := 0; i < ordersN; i++ {
		o := h[archiveHdrEngramOrders0+i]
		if o < 1 || o > 32 {
			return fmt.Errorf("needle: invalid archive engram order %d", o)
		}
		if o > maxOrder {
			maxOrder = o
		}
	}
	if dilation < maxOrder {
		return fmt.Errorf("needle: archive engram dilation %d is smaller than max order %d", dilation, maxOrder)
	}
	if tables%uint32(ordersN) != 0 {
		return fmt.Errorf("needle: archive engram tables %d not divisible by %d orders", tables, ordersN)
	}
	product, ok := checkedMulInt64(int64(subDim), int64(tables))
	if !ok || product != int64(dModel) {
		return fmt.Errorf("needle: archive engram tables %d and sub-dim %d do not match d_model %d", tables, subDim, dModel)
	}
	prevSite := uint32(0)
	for i := 0; i < sitesN; i++ {
		site := h[archiveHdrEngramSites0+i]
		if site >= layers {
			return fmt.Errorf("needle: archive engram site %d out of range for %d layers", site, layers)
		}
		if i > 0 && site <= prevSite {
			return fmt.Errorf("needle: archive engram sites are not strictly increasing")
		}
		prevSite = site
	}
	return nil
}

func validateArchiveCodebook(codebook []float32) error {
	if len(codebook) != archiveCodebookLen {
		return fmt.Errorf("needle: archive codebook length %d want %d", len(codebook), archiveCodebookLen)
	}
	if err := validateArchiveCodebookSegment(codebook, 0, 4, 2); err != nil {
		return err
	}
	if err := validateArchiveCodebookSegment(codebook, 4, 12, 3); err != nil {
		return err
	}
	if err := validateArchiveCodebookSegment(codebook, 12, 28, 4); err != nil {
		return err
	}
	return nil
}

func validateArchiveCodebookSegment(codebook []float32, start, end, bits int) error {
	segment := codebook[start:end]
	for i, v := range segment {
		if !isFinite32(v) {
			return fmt.Errorf("needle: archive codebook[%d] for %d-bit CQ is non-finite", start+i, bits)
		}
		if i > 0 && !(segment[i-1] < v) {
			return fmt.Errorf("needle: archive %d-bit codebook is not strictly increasing", bits)
		}
	}
	return nil
}

func parseArchiveRecordSpec(index int, raw []byte, decodedTotal int64) (archiveRecordSpec, error) {
	dtype := int(raw[0])
	ndim := int(raw[1])
	pad := binary.LittleEndian.Uint16(raw[2:])
	if pad != 0 {
		return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d has nonzero pad %d", index, pad)
	}
	if ndim > 4 {
		return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d ndim %d exceeds 4", index, ndim)
	}
	if dtype < archiveDTypeFP16 || dtype > archiveDTypeRAW {
		return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d unsupported dtype %d", index, dtype)
	}

	shapeWords := [4]uint32{
		binary.LittleEndian.Uint32(raw[4:]),
		binary.LittleEndian.Uint32(raw[8:]),
		binary.LittleEndian.Uint32(raw[12:]),
		binary.LittleEndian.Uint32(raw[16:]),
	}
	shape := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		if shapeWords[i] == 0 {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d has zero dimension at axis %d", index, i)
		}
		if shapeWords[i] > uint32(maxInt()) {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d dimension %d exceeds platform limit", index, shapeWords[i])
		}
		shape[i] = int(shapeWords[i])
	}
	for i := ndim; i < 4; i++ {
		if shapeWords[i] != 0 {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d shape pad %d is nonzero", index, i)
		}
	}

	offset := binary.LittleEndian.Uint64(raw[20:])
	nbytes := binary.LittleEndian.Uint64(raw[28:])
	group := binary.LittleEndian.Uint32(raw[36:])
	bits := binary.LittleEndian.Uint32(raw[40:])
	if offset%archiveAlign != 0 {
		return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d offset %d is not %d-byte aligned", index, offset, archiveAlign)
	}
	if offset > uint64(math.MaxInt64) || nbytes > uint64(math.MaxInt64) {
		return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d offsets exceed platform limit", index)
	}

	spec := archiveRecordSpec{
		index:  index,
		dtype:  dtype,
		shape:  shape,
		offset: int64(offset),
		nbytes: int64(nbytes),
		group:  int(group),
		bits:   int(bits),
	}
	switch dtype {
	case archiveDTypeFP16:
		if group != 0 || bits != 0 {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d FP16 group/bits must be zero", index)
		}
		elements, err := archiveShapeElements(index, shape)
		if err != nil {
			return archiveRecordSpec{}, err
		}
		want, ok := checkedMulInt64(int64(elements), 2)
		if !ok {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d byte size overflow", index)
		}
		if spec.nbytes != want {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d FP16 byte size %d want %d", index, spec.nbytes, want)
		}
		decoded, ok := checkedMulInt64(int64(elements), 4)
		if !ok {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d decoded byte size overflow", index)
		}
		spec.decodedBytes = decoded
	case archiveDTypeFP32:
		if group != 0 || bits != 0 {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d FP32 group/bits must be zero", index)
		}
		elements, err := archiveShapeElements(index, shape)
		if err != nil {
			return archiveRecordSpec{}, err
		}
		want, ok := checkedMulInt64(int64(elements), 4)
		if !ok {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d byte size overflow", index)
		}
		if spec.nbytes != want {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d FP32 byte size %d want %d", index, spec.nbytes, want)
		}
		spec.decodedBytes = want
	case archiveDTypeCQ:
		if ndim != 2 {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d CQ rank %d unsupported", index, ndim)
		}
		if group != archiveCQGroup {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d CQ group %d unsupported", index, group)
		}
		if bits != 1 && bits != 2 && bits != 3 && bits != 4 && bits != archiveTernaryBits {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d CQ bits %d unsupported", index, bits)
		}
		want, decoded, err := archiveCQSizes(index, shape[0], shape[1], int(bits))
		if err != nil {
			return archiveRecordSpec{}, err
		}
		if spec.nbytes != want {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d CQ byte size %d want %d", index, spec.nbytes, want)
		}
		if decodedTotal > archiveMaxDecodedBytes-decoded {
			return archiveRecordSpec{}, fmt.Errorf("needle: decoded tensor bytes %d exceed %d", decodedTotal+decoded, archiveMaxDecodedBytes)
		}
		spec.decodedBytes = decoded
	case archiveDTypeRAW:
		if ndim != 0 {
			return archiveRecordSpec{}, fmt.Errorf("needle: RAW record must be scalar-shaped")
		}
		if group != 0 || bits != 0 {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d RAW group/bits must be zero", index)
		}
		spec.decodedBytes = spec.nbytes
		if spec.nbytes > archiveMaxRawBytes {
			return archiveRecordSpec{}, fmt.Errorf("needle: archive record %d RAW byte size %d exceeds %d", index, spec.nbytes, archiveMaxRawBytes)
		}
	}
	return spec, nil
}

func archiveShapeElements(index int, shape []int) (int, error) {
	if len(shape) == 0 {
		return 1, nil
	}
	elements := 1
	for _, dim := range shape {
		if dim <= 0 {
			return 0, fmt.Errorf("needle: archive record %d has invalid dimension %d", index, dim)
		}
		if elements > maxInt()/dim {
			return 0, fmt.Errorf("needle: archive record %d shape %v overflows element count", index, shape)
		}
		elements *= dim
	}
	return elements, nil
}

func archiveCQSizes(index, rows, cols, bits int) (nbytes, decoded int64, err error) {
	if rows <= 0 || cols <= 0 {
		return 0, 0, fmt.Errorf("needle: archive record %d has invalid CQ shape [%d %d]", index, rows, cols)
	}
	if cols > maxInt()-archiveCQGroup {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ width overflows", index)
	}
	inPad := ((cols + archiveCQGroup - 1) / archiveCQGroup) * archiveCQGroup
	groups := inPad / archiveCQGroup
	var packedPerRow int64
	switch bits {
	case archiveTernaryBits:
		packedPerRow = int64(inPad / 4)
	default:
		packedPerRow = int64(inPad) * int64(bits) / 8
	}
	packedBytes, ok := checkedMulInt64(int64(rows), packedPerRow)
	if !ok {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ packed byte size overflow", index)
	}
	normCount, ok := checkedMulInt64(int64(rows), int64(groups))
	if !ok {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ norm count overflow", index)
	}
	normBytes, ok := checkedMulInt64(normCount, 2)
	if !ok {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ norm byte size overflow", index)
	}
	nbytes, ok = checkedAddInt64(packedBytes, normBytes)
	if !ok {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ byte size overflow", index)
	}
	decodedElems, ok := checkedMulInt64(int64(rows), int64(cols))
	if !ok {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ element count overflow", index)
	}
	decoded, ok = checkedMulInt64(decodedElems, 4)
	if !ok {
		return 0, 0, fmt.Errorf("needle: archive record %d CQ decoded byte size overflow", index)
	}
	return nbytes, decoded, nil
}

func validateArchiveRecordLayout(specs []archiveRecordSpec, metadataEnd, fileSize int64) error {
	ordered := append([]archiveRecordSpec(nil), specs...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].offset != ordered[j].offset {
			return ordered[i].offset < ordered[j].offset
		}
		return ordered[i].index < ordered[j].index
	})
	prevEnd := metadataEnd
	prevName := "metadata"
	for _, spec := range ordered {
		if spec.offset < metadataEnd {
			return fmt.Errorf("needle: archive record %d overlaps metadata", spec.index)
		}
		end, ok := checkedAddInt64(spec.offset, spec.nbytes)
		if !ok {
			return fmt.Errorf("needle: archive record %d offset overflow", spec.index)
		}
		if end > fileSize {
			return fmt.Errorf("needle: archive record %d [%d,%d) extends past file size %d", spec.index, spec.offset, end, fileSize)
		}
		if spec.offset < prevEnd {
			return fmt.Errorf("needle: archive record %s and record %d overlap", prevName, spec.index)
		}
		prevEnd = end
		prevName = fmt.Sprintf("%d", spec.index)
	}
	return nil
}

func decodeArchiveRecord(spec archiveRecordSpec, blob []byte, codebook []float32) (ArchiveRecord, error) {
	rec := ArchiveRecord{
		DType: spec.dtype,
		Shape: append([]int(nil), spec.shape...),
		Group: spec.group,
		Bits:  spec.bits,
	}
	switch spec.dtype {
	case archiveDTypeRAW:
		rec.Raw = append([]byte(nil), blob...)
		return rec, nil
	case archiveDTypeFP16:
		data, err := decodeArchiveFP16(spec.index, blob)
		if err != nil {
			return ArchiveRecord{}, err
		}
		rec.Data = data
		return rec, nil
	case archiveDTypeFP32:
		data, err := decodeArchiveFP32(spec.index, blob)
		if err != nil {
			return ArchiveRecord{}, err
		}
		rec.Data = data
		return rec, nil
	case archiveDTypeCQ:
		data, err := decodeArchiveCQ(spec.index, spec.shape[0], spec.shape[1], spec.bits, blob, codebook)
		if err != nil {
			return ArchiveRecord{}, err
		}
		rec.Data = data
		return rec, nil
	default:
		return ArchiveRecord{}, fmt.Errorf("needle: archive record %d unsupported dtype %d", spec.index, spec.dtype)
	}
}

func decodeArchiveFP16(index int, blob []byte) ([]float32, error) {
	if len(blob)%2 != 0 {
		return nil, fmt.Errorf("needle: archive record %d FP16 byte length %d is not divisible by 2", index, len(blob))
	}
	n := len(blob) / 2
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		v := half.F16ToF32(binary.LittleEndian.Uint16(blob[i*2:]))
		if !isFinite32(v) {
			return nil, fmt.Errorf("needle: archive record %d contains non-finite FP16 value", index)
		}
		out[i] = v
	}
	return out, nil
}

func decodeArchiveFP32(index int, blob []byte) ([]float32, error) {
	if len(blob)%4 != 0 {
		return nil, fmt.Errorf("needle: archive record %d FP32 byte length %d is not divisible by 4", index, len(blob))
	}
	n := len(blob) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		v := math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
		if !isFinite32(v) {
			return nil, fmt.Errorf("needle: archive record %d contains non-finite FP32 value", index)
		}
		out[i] = v
	}
	return out, nil
}

func decodeArchiveCQ(index, rows, cols, bits int, blob []byte, codebook []float32) ([]float32, error) {
	want, _, err := archiveCQSizes(index, rows, cols, bits)
	if err != nil {
		return nil, err
	}
	if int64(len(blob)) != want {
		return nil, fmt.Errorf("needle: archive record %d CQ byte size %d want %d", index, len(blob), want)
	}
	cb, err := archiveCodebookForBits(codebook, bits)
	if err != nil {
		return nil, err
	}
	inPad := ((cols + archiveCQGroup - 1) / archiveCQGroup) * archiveCQGroup
	groups := inPad / archiveCQGroup
	packedPerGroup := 0
	switch bits {
	case archiveTernaryBits:
		packedPerGroup = archiveCQGroup / 4
	default:
		packedPerGroup = archiveCQGroup * bits / 8
	}
	packedPerRow := packedPerGroup * groups
	packedBytes := rows * packedPerRow
	out := make([]float32, rows*cols)
	var work [archiveCQGroup]float32
	for r := 0; r < rows; r++ {
		rowPacked := blob[r*packedPerRow : (r+1)*packedPerRow]
		rowNorms := blob[packedBytes+r*groups*2 : packedBytes+(r+1)*groups*2]
		rowOut := out[r*cols : (r+1)*cols]
		for g := 0; g < groups; g++ {
			norm := half.F16ToF32(binary.LittleEndian.Uint16(rowNorms[g*2:]))
			if !isFinite32(norm) || norm < 0 {
				return nil, fmt.Errorf("needle: archive record %d has invalid CQ norm", index)
			}
			groupPacked := rowPacked[g*packedPerGroup : (g+1)*packedPerGroup]
			if bits == archiveTernaryBits {
				if err := decodeArchiveTernaryGroup(&work, groupPacked, norm); err != nil {
					return nil, fmt.Errorf("needle: archive record %d: %w", index, err)
				}
			} else {
				decodeArchiveCodebookGroup(&work, groupPacked, norm, bits, cb)
			}
			archiveWalshInverse128(&work)
			start := g * archiveCQGroup
			width := cols - start
			if width > archiveCQGroup {
				width = archiveCQGroup
			}
			for i := 0; i < width; i++ {
				if !isFinite32(work[i]) {
					return nil, fmt.Errorf("needle: archive record %d decoded non-finite CQ value", index)
				}
				rowOut[start+i] = work[i]
			}
		}
	}
	return out, nil
}

func archiveCodebookForBits(codebook []float32, bits int) ([]float32, error) {
	switch bits {
	case 2:
		return codebook[0:4], nil
	case 3:
		return codebook[4:12], nil
	case 4:
		return codebook[12:28], nil
	case 1:
		return []float32{-archiveBinaryLevel, archiveBinaryLevel}, nil
	case archiveTernaryBits:
		return nil, nil
	default:
		return nil, fmt.Errorf("needle: unsupported CQ bits %d", bits)
	}
}

func decodeArchiveCodebookGroup(dst *[archiveCQGroup]float32, packed []byte, norm float32, bits int, codebook []float32) {
	mask := uint32((1 << bits) - 1)
	for chunk := 0; chunk < archiveCQGroup/8; chunk++ {
		base := chunk * bits
		word := uint32(0)
		for b := 0; b < bits; b++ {
			word |= uint32(packed[base+b]) << (8 * b)
		}
		for i := 0; i < 8; i++ {
			idx := int((word >> (i * bits)) & mask)
			dst[chunk*8+i] = codebook[idx] * norm
		}
	}
}

func decodeArchiveTernaryGroup(dst *[archiveCQGroup]float32, packed []byte, norm float32) error {
	for i, b := range packed {
		for j := 0; j < 4; j++ {
			crumb := (b >> (2 * j)) & 0x3
			pos := i*4 + j
			switch crumb {
			case 3:
				dst[pos] = -archiveTernaryLevel * norm
			case 0:
				dst[pos] = 0
			case 1:
				dst[pos] = archiveTernaryLevel * norm
			case 2:
				return fmt.Errorf("invalid ternary crumb 2")
			}
		}
	}
	return nil
}

func archiveWalshInverse128(x *[archiveCQGroup]float32) {
	for step := 1; step < archiveCQGroup; step <<= 1 {
		block := step << 1
		for base := 0; base < archiveCQGroup; base += block {
			for i := 0; i < step; i++ {
				a := x[base+i]
				b := x[base+step+i]
				x[base+i] = a + b
				x[base+step+i] = a - b
			}
		}
	}
	for i := range x {
		x[i] *= archiveWalshScale128
	}
}

func isFinite32(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}
