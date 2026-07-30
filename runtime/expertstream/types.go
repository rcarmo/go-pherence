package expertstream

import "errors"

const ManifestVersion = 1

var (
	ErrInvalidManifest  = errors.New("expertstream: invalid manifest")
	ErrChecksumMismatch = errors.New("expertstream: checksum mismatch")
	ErrShortRead        = errors.New("expertstream: short read")
	ErrUnknownExpert    = errors.New("expertstream: unknown expert")
	ErrSlotCapacity     = errors.New("expertstream: slot capacity exceeded")
	ErrClosed           = errors.New("expertstream: reader closed")
)

// Manifest describes a backend-neutral expert package.
type Manifest struct {
	Version   int          `json:"version"`
	ModelID   string       `json:"model_id"`
	Layers    int          `json:"layers"`
	Experts   int          `json:"experts"`
	Alignment int64        `json:"alignment"`
	Files     []DataFile   `json:"files"`
	Entries   []ExpertSpec `json:"entries"`
}

// DataFile names a data blob referenced by one or more experts.
type DataFile struct {
	ID     string `json:"id"`
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// ExpertSpec describes one expert's contiguous gate/up/down payloads.
type ExpertSpec struct {
	Key  uint64        `json:"key"`
	File string        `json:"file"`
	Gate ComponentSpec `json:"gate"`
	Up   ComponentSpec `json:"up"`
	Down ComponentSpec `json:"down"`
}

// ComponentSpec describes one expert component inside a data file.
type ComponentSpec struct {
	Offset int64   `json:"offset"`
	Size   int64   `json:"size"`
	DType  string  `json:"dtype"`
	Shape  []int64 `json:"shape"`
}

// Options configures a Reader.
type Options struct {
	Slots   int
	Workers int
}

// Component exposes one loaded component view.
type Component struct {
	DType string
	Shape []int64
	Bytes []byte
}

// SlotView identifies the slot backing a loaded expert.
type SlotView struct {
	Index int
	Bytes []byte
}

// LoadedExpert is one Load result in caller-requested order.
type LoadedExpert struct {
	Key  uint64
	Slot SlotView
	Gate Component
	Up   Component
	Down Component
}
