package vulkan

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// Save ALL function slots and published state; no native loader is opened.
func offlineInitState(t *testing.T) {
	t.Helper()
	offlineVK(t)
	mockVK(t, &vkReady, false)
	mockVK(t, &vkLimits, VulkanDeviceLimits{})
	mockVK(t, &vkLib, uintptr(0))
	mockVK(t, &vkInstance, VkInstance(0))
	mockVK(t, &vkPhysDev, VkPhysicalDevice(0))
	mockVK(t, &vkDevice, VkDevice(0))
	mockVK(t, &vkQueue, VkQueue(0))
	mockVK(t, &vkCmdPool, VkCommandPool(0))
	mockVK(t, &vkComputeQueueFamily, uint32(0))
	mockVK(t, &vkDevName, "")
	for _, entry := range vkSymbols() {
		slot := reflect.ValueOf(entry.target).Elem()
		old := reflect.New(slot.Type()).Elem()
		old.Set(slot)
		t.Cleanup(func() { slot.Set(old) })
	}
}

func initMock(t *testing.T, failure string) (vkLoader, *[]string) {
	t.Helper()
	var events []string
	unpublished := func() {
		if vkReady || vkLib != 0 || vkInstance != 0 || vkDevice != 0 || vkQueue != 0 || vkCmdPool != 0 || vkPhysDev != 0 || vkDevName != "" || vkLimits != (VulkanDeviceLimits{}) {
			t.Error("partial publication")
		}
	}
	step := func(name string) { events = append(events, name); unpublished() }
	functions := map[string]any{
		"vkCreateInstance": func(p, a unsafe.Pointer, out *VkInstance) VkResult {
			step("instance")
			if *(*uint32)(p) != 1 || *(*uint32)(unsafe.Add(p, 32)) != 0 {
				t.Error("instance ABI")
			}
			app := *(*unsafe.Pointer)(unsafe.Add(p, 24))
			if *(*uint32)(unsafe.Add(app, 44)) != (1<<22 | 3<<12) {
				t.Error("api version")
			}
			*out = 10
			if failure == "instance" {
				return -1
			}
			if failure == "nil-instance" {
				*out = 0
			}
			return VK_SUCCESS
		},
		"vkEnumeratePhysicalDevices": func(i VkInstance, n *uint32, out *VkPhysicalDevice) VkResult {
			if i != 10 {
				t.Error("instance owner")
			}
			if out == nil {
				step("device-count")
				*n = 2
				switch failure {
				case "count-error":
					return -1
				case "count-incomplete":
					return 5
				case "no-devices":
					*n = 0
				case "device-bound":
					*n = 65
				}
				return VK_SUCCESS
			}
			step("device-fill")
			if *n != 2 {
				t.Error("device capacity")
			}
			list := unsafe.Slice(out, 2)
			list[0] = 11
			list[1] = 12
			switch failure {
			case "fill-error":
				return -1
			case "fill-incomplete":
				return 5
			case "fill-grown":
				*n = 3
			case "fill-empty":
				*n = 0
			case "nil-physical":
				list[0] = 0
			case "fill-shrink":
				*n = 1
			}
			return VK_SUCCESS
		},
		"vkGetPhysicalDeviceProperties": func(d VkPhysicalDevice, p unsafe.Pointer) {
			step("properties")
			if uintptr(p)%8 != 0 {
				t.Error("property alignment")
			}
			v := (*vkDeviceProperties)(p)
			*v = offlineProperties()
			v.deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
			copy(v.deviceName[:], "mock-integrated")
			if d == 12 {
				v.deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
				copy(v.deviceName[:], "mock-discrete\x00")
			}
			if failure == "cpu-only" {
				v.deviceType = VK_PHYSICAL_DEVICE_TYPE_CPU
			}
			if failure == "other-only" {
				v.deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER
			}
		},
		"vkGetPhysicalDeviceQueueFamilyProperties": func(d VkPhysicalDevice, n *uint32, p unsafe.Pointer) {
			if p == nil {
				step("queue-count")
				*n = 2
				switch failure {
				case "no-families":
					*n = 0
				case "family-bound":
					*n = 257
				}
				return
			}
			step("queue-fill")
			q := unsafe.Slice((*vkQueueFamilyProperties)(p), 2)
			q[0] = vkQueueFamilyProperties{queueFlags: 1, queueCount: 1}
			q[1] = vkQueueFamilyProperties{queueFlags: VK_QUEUE_COMPUTE_BIT, queueCount: 1}
			switch failure {
			case "queue-fill-empty":
				*n = 0
			case "queue-fill-grown":
				*n = 3
			case "no-compute":
				q[1].queueFlags = 1
			case "zero-queues":
				q[1].queueCount = 0
			}
		},
		"vkCreateDevice": func(d VkPhysicalDevice, p, a unsafe.Pointer, out *VkDevice) VkResult {
			step("device")
			if *(*uint32)(p) != 3 || *(*uint32)(unsafe.Add(p, 20)) != 1 {
				t.Error("device ABI")
			}
			q := *(*unsafe.Pointer)(unsafe.Add(p, 24))
			if *(*uint32)(q) != 2 || *(*uint32)(unsafe.Add(q, 20)) != 1 || *(*uint32)(unsafe.Add(q, 24)) != 1 {
				t.Error("queue create ABI")
			}
			priority := *(*unsafe.Pointer)(unsafe.Add(q, 32))
			if *(*float32)(priority) != 1 {
				t.Error("queue priority")
			}
			*out = 20
			if failure == "device" {
				return -2
			}
			if failure == "nil-device" {
				*out = 0
			}
			return VK_SUCCESS
		},
		"vkGetDeviceQueue": func(d VkDevice, family, index uint32, out *VkQueue) {
			step("queue")
			if d != 20 || family != 1 || index != 0 {
				t.Error("queue owner/index")
			}
			*out = 21
			if failure == "nil-queue" {
				*out = 0
			}
		},
		"vkCreateCommandPool": func(d VkDevice, p, a unsafe.Pointer, out *VkCommandPool) VkResult {
			step("pool")
			if d != 20 || *(*uint32)(p) != 39 || *(*uint32)(unsafe.Add(p, 16)) != 2 || *(*uint32)(unsafe.Add(p, 20)) != 1 {
				t.Error("pool ABI/owner")
			}
			*out = 22
			if failure == "pool" {
				return -3
			}
			if failure == "nil-pool" {
				*out = 0
			}
			return VK_SUCCESS
		},
		"vkDestroyCommandPool": func(d VkDevice, p VkCommandPool, a unsafe.Pointer) {
			step("destroy-pool")
			if d != 20 || p != 22 {
				t.Error("pool destruction owner")
			}
		},
		"vkDestroyDevice": func(d VkDevice, a unsafe.Pointer) {
			step("destroy-device")
			if d != 20 {
				t.Error("device destruction owner")
			}
		},
		"vkDestroyInstance": func(i VkInstance, a unsafe.Pointer) {
			step("destroy-instance")
			if i != 10 {
				t.Error("instance destruction owner")
			}
		},
	}
	loader := vkLoader{
		open: func() (uintptr, error) {
			step("open")
			if failure == "open" {
				return 0, errors.New("missing loader")
			}
			return 9, nil
		},
		close: func(h uintptr) error {
			step("close")
			if h != 9 {
				t.Error("library owner")
			}
			for _, s := range vkSymbols() {
				if !reflect.ValueOf(s.target).Elem().IsNil() {
					t.Error("stale function before unload", s.name)
				}
			}
			if failure == "close" {
				return errors.New("unload failed")
			}
			return nil
		},
		bind: func(target any, lib uintptr, name string) error {
			if lib != 9 {
				t.Error("bind owner")
			}
			if failure == "symbol:"+name {
				return errors.New("missing symbol")
			}
			slot := reflect.ValueOf(target).Elem()
			if failure == "nil-symbol" && name == "vkDestroyInstance" {
				return nil
			}
			if f, ok := functions[name]; ok {
				slot.Set(reflect.ValueOf(f))
			} else {
				slot.Set(reflect.MakeFunc(slot.Type(), func(args []reflect.Value) []reflect.Value {
					t.Errorf("unexpected native entry %s", name)
					out := make([]reflect.Value, slot.Type().NumOut())
					for i := range out {
						out[i] = reflect.Zero(slot.Type().Out(i))
					}
					return out
				}))
			}
			return nil
		},
	}
	// Force a failure after symbols resolve so unload failure can be exercised.
	if failure == "close" {
		functions["vkCreateInstance"] = func(p, a unsafe.Pointer, out *VkInstance) VkResult { step("instance"); return -1 }
	}
	return loader, &events
}
func runInitMock(loader vkLoader) bool {
	_ = vkAcquire(context.Background())
	defer vkRelease()
	return vkInitLocked(loader)
}

func TestVulkanOfflineInitRollback(t *testing.T) {
	for _, failure := range []string{"open", "nil-symbol", "instance", "nil-instance", "count-error", "count-incomplete", "no-devices", "device-bound", "fill-error", "fill-incomplete", "fill-grown", "fill-empty", "nil-physical", "cpu-only", "other-only", "no-families", "family-bound", "queue-fill-empty", "queue-fill-grown", "no-compute", "zero-queues", "device", "nil-device", "nil-queue", "pool", "nil-pool", "close"} {
		t.Run(failure, func(t *testing.T) {
			offlineInitState(t)
			t.Setenv("GO_PHERENCE_VULKAN_ALLOW_CPU", "")
			loader, events := initMock(t, failure)
			if runInitMock(loader) {
				t.Fatal("failure accepted")
			}
			if vkReady || vkInstance != 0 || vkDevice != 0 || vkPhysDev != 0 || vkQueue != 0 || vkCmdPool != 0 || vkDevName != "" {
				t.Fatal("partial global state")
			}
			for _, s := range vkSymbols() {
				if failure != "open" && !reflect.ValueOf(s.target).Elem().IsNil() {
					t.Fatal("dangling symbol", s.name)
				}
			}
			freed := []string{}
			for _, e := range *events {
				if strings.HasPrefix(e, "destroy-") || e == "close" {
					freed = append(freed, e)
				}
			}
			want := []string{"close"}
			switch failure {
			case "open":
				want = []string{}
			case "nil-symbol", "instance", "nil-instance", "close":
			case "nil-queue", "pool", "nil-pool":
				want = []string{"destroy-device", "destroy-instance", "close"}
			default:
				want = []string{"destroy-instance", "close"}
			}
			if !reflect.DeepEqual(freed, want) {
				t.Fatalf("cleanup got%v want%v events%v", freed, want, *events)
			}
			if failure == "close" {
				if vkLib != 9 {
					t.Fatal("unload failure not retained")
				}
				n := len(*events)
				if runInitMock(loader) || len(*events) != n {
					t.Fatal("reopened held library")
				}
			} else if vkLib != 0 {
				t.Fatal("library leaked")
			}
		})
	}
}
func TestVulkanOfflineInitEveryMissingSymbol(t *testing.T) {
	for _, symbol := range vkSymbols() {
		t.Run(symbol.name, func(t *testing.T) {
			offlineInitState(t)
			loader, events := initMock(t, "symbol:"+symbol.name)
			// Stale slots must be cleared before attempting registration.
			slot := reflect.ValueOf(symbol.target).Elem()
			slot.Set(reflect.MakeFunc(slot.Type(), func([]reflect.Value) []reflect.Value { panic("stale symbol invoked") }))
			if runInitMock(loader) {
				t.Fatal("missing symbol accepted")
			}
			if !reflect.DeepEqual(*events, []string{"open", "close"}) {
				t.Fatal("created object before all symbols available", *events)
			}
		})
	}
}
func TestVulkanOfflineInitRetryAndPublication(t *testing.T) {
	for _, mode := range []string{"success", "fill-shrink", "cpu-only", "other-only"} {
		t.Run(mode, func(t *testing.T) {
			offlineInitState(t)
			t.Setenv("GO_PHERENCE_VULKAN_ALLOW_CPU", "1")
			failed, _ := initMock(t, "pool")
			if runInitMock(failed) {
				t.Fatal("failure accepted")
			}
			loader, events := initMock(t, mode)
			if !runInitMock(loader) {
				t.Fatal("retry failed", *events)
			}
			wantPhys := VkPhysicalDevice(12)
			if mode == "fill-shrink" || mode == "cpu-only" || mode == "other-only" {
				wantPhys = 11
			}
			if !vkReady || vkLib != 9 || vkInstance != 10 || vkDevice != 20 || vkQueue != 21 || vkCmdPool != 22 || vkPhysDev != wantPhys || vkComputeQueueFamily != 1 || vkDevName == "" {
				t.Fatal("bad committed state")
			}
			n := len(*events)
			if !runInitMock(loader) || len(*events) != n {
				t.Fatal("success reinitialised")
			}
		})
	}
}
func TestVulkanOfflineInitAdmissionAndLayouts(t *testing.T) {
	offlineInitState(t)
	loader, events := initMock(t, "")
	vkLost = true
	if runInitMock(loader) {
		t.Fatal("lost accepted")
	}
	vkLost = false
	vkPending = &vkPendingSubmission{}
	if runInitMock(loader) {
		t.Fatal("pending accepted")
	}
	if len(*events) != 0 {
		t.Fatal("admission called loader")
	}
	var p vkDeviceProperties
	var q vkQueueFamilyProperties
	if unsafe.Alignof(p) != 8 || unsafe.Offsetof(p.deviceName) != 20 || unsafe.Sizeof(p) < 824 || unsafe.Sizeof(q) != 24 {
		t.Fatal("property ABI")
	}
}
