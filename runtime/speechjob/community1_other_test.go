//go:build !linux || !amd64

package speechjob

import (
	"context"
	"strings"
	"testing"
)

func TestCommunity1UnsupportedStubs(t *testing.T) {
	if err := ValidateCommunity1Config(Community1StageConfig{}); err == nil || !strings.Contains(err.Error(), "Linux/amd64") {
		t.Fatal(err)
	}
	if _, err := NewCommunity1Stage(nil, Community1StageConfig{}); err == nil || !strings.Contains(err.Error(), "Linux/amd64") {
		t.Fatal(err)
	}
	if _, err := NewOwnedCommunity1Stage(nil, Community1StageConfig{}); err == nil || !strings.Contains(err.Error(), "Linux/amd64") {
		t.Fatal(err)
	}
	if _, err := NewVulkanCommunity1Stage(nil, VulkanCommunity1StageConfig{}); err == nil || !strings.Contains(err.Error(), "Linux/amd64") {
		t.Fatal(err)
	}
	var owner *Community1Owner
	if err := owner.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	var stage *VulkanCommunity1Stage
	if err := stage.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
}
