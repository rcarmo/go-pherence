//go:build !linux || !amd64

package speechjob

import (
	"context"
	"fmt"
	"time"

	c1 "github.com/rcarmo/go-pherence/model/speaker/community1"
)

func community1Unsupported() error {
	return fmt.Errorf("Community-1 speech jobs require Linux/amd64")
}

// Community1StageConfig preserves the public experimental stage contract on
// unsupported platforms so callers fail clearly at construction/validation time.
type Community1StageConfig struct {
	AllowExperimental                                                             bool
	SegmentationSHA256, EmbeddingSHA256, PLDASHA256, FiltersSHA256, RuntimeSHA256 string
	ModelIdentitySHA256                                                           string
	ExecutionBackendSHA256                                                        string
	PCM                                                                           c1.DiarizationPCMConfig
	SegmentationModes                                                             c1.SegmentationModes
	EmbeddingMode                                                                 c1.WeSpeakerBlockMode
	OverlapBranches                                                               bool
	MaxResultBytes                                                                int64
}

func NewCommunity1Stage(*c1.ExperimentalDiarization, Community1StageConfig) (Stage, error) {
	return Stage{}, community1Unsupported()
}
func ValidateCommunity1Config(Community1StageConfig) error { return community1Unsupported() }

// Community1Status is retained for build-tag parity with the Linux/amd64 owner.
type Community1Status struct {
	Running, Stopping, Poisoned, Closed bool
	ErrorCode                           string
}

// Community1Owner is unavailable outside Linux/amd64.
type Community1Owner struct{}

func NewOwnedCommunity1Stage(*c1.ExperimentalDiarization, Community1StageConfig) (*Community1Owner, error) {
	return nil, community1Unsupported()
}
func (o *Community1Owner) Stage() Stage { return Stage{} }
func (o *Community1Owner) Status() Community1Status {
	return Community1Status{Closed: true, ErrorCode: "unsupported_platform"}
}
func (o *Community1Owner) Close(context.Context) error {
	if o == nil {
		return nil
	}
	return community1Unsupported()
}

// VulkanCommunity1StageConfig preserves the explicit unsupported surface.
type VulkanCommunity1StageConfig struct {
	Community         Community1StageConfig
	AllowExperimental bool
	BackendSHA256     string
	DeviceIdentity    string
	DrainPoll         time.Duration
}

type VulkanCommunity1Status struct {
	Running, Draining, Quarantined, Stopping, Closed bool
	ErrorCode                                        string
}

type VulkanCommunity1Stage struct{}

func NewVulkanCommunity1Stage(*c1.VulkanDiarization, VulkanCommunity1StageConfig) (*VulkanCommunity1Stage, error) {
	return nil, community1Unsupported()
}
func (o *VulkanCommunity1Stage) Stage() Stage { return Stage{} }
func (o *VulkanCommunity1Stage) Status() VulkanCommunity1Status {
	return VulkanCommunity1Status{Closed: true, ErrorCode: "unsupported_platform"}
}
func (o *VulkanCommunity1Stage) Close(context.Context) error {
	if o == nil {
		return nil
	}
	return community1Unsupported()
}
