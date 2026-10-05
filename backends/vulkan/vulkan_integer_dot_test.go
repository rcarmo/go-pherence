package vulkan

import (
	"context"
	"errors"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"
)

func TestVulkanOfflineIntegerDotABI(t *testing.T) {
	if unsafe.Sizeof(vkIntegerDotFeatures{}) != 24 || unsafe.Offsetof(vkIntegerDotFeatures{}.enabled) != 16 || unsafe.Sizeof(vkFeatures2{}) != 240 || unsafe.Sizeof(vkIntegerDotProperties{}) != 136 || unsafe.Offsetof(vkProperties2{}.properties) != 16 || unsafe.Sizeof(vkProperties2{}) != 840 {
		t.Fatal("ABI")
	}
	offlineInitState(t)
	mockVK(t, &vkGetPhysicalDeviceFeatures2, func(_ VkPhysicalDevice, p unsafe.Pointer) {
		q := (*vkFeatures2)(p)
		if q.sType != 1000059000 {
			t.Fatal("query type")
		}
		f := (*vkIntegerDotFeatures)(q.pNext)
		if f.sType != 1000280000 || f.enabled != 0 {
			t.Fatal("zero feature")
		}
		f.enabled = 1
	})
	mockVK(t, &vkGetPhysicalDeviceProperties2, func(_ VkPhysicalDevice, p unsafe.Pointer) {
		q := (*vkProperties2)(p)
		if q.sType != 1000059001 {
			t.Fatal("properties type")
		}
		r := (*vkIntegerDotProperties)(q.pNext)
		if r.sType != 1000280001 {
			t.Fatal("property chain")
		}
		r.accelerated[5] = 1
	})
	if f := vkQueryIntegerDot(12); f == nil || f.enabled != 1 || f.pNext != nil {
		t.Fatal("feature")
	}
	mockVK(t, &vkGetPhysicalDeviceProperties2, func(_ VkPhysicalDevice, p unsafe.Pointer) {
		r := (*vkIntegerDotProperties)((*vkProperties2)(p).pNext)
		r.accelerated[4] = 1
		r.accelerated[5] = 0
	})
	if vkQueryIntegerDot(12) != nil {
		t.Fatal("wrong accelerated signedness")
	}
}
func TestVulkanOfflineIntegerDotInit(t *testing.T) {
	for _, arm := range []string{"ok", "unsupported", "notaccelerated", "missing", "device", "queue", "pool"} {
		t.Run(arm, func(t *testing.T) {
			offlineInitState(t)
			failure := ""
			if arm == "device" || arm == "pool" {
				failure = arm
			}
			if arm == "queue" {
				failure = "nil-queue"
			}
			loader, events := initMock(t, failure)
			bind := loader.bind
			loader.bind = func(target any, lib uintptr, name string) error {
				if name == "vkGetPhysicalDeviceFeatures2" {
					if arm == "missing" {
						return errors.New("missing")
					}
					reflect.ValueOf(target).Elem().Set(reflect.ValueOf(func(_ VkPhysicalDevice, p unsafe.Pointer) {
						if arm != "unsupported" {
							(*vkIntegerDotFeatures)((*vkFeatures2)(p).pNext).enabled = 1
						}
					}))
					return nil
				}
				if name == "vkGetPhysicalDeviceProperties2" {
					reflect.ValueOf(target).Elem().Set(reflect.ValueOf(func(_ VkPhysicalDevice, p unsafe.Pointer) {
						if arm != "notaccelerated" {
							(*vkIntegerDotProperties)((*vkProperties2)(p).pNext).accelerated[5] = 1
						}
					}))
					return nil
				}
				if name == "vkCreateDevice" {
					if e := bind(target, lib, name); e != nil {
						return e
					}
					original := reflect.ValueOf(target).Elem().Interface().(func(VkPhysicalDevice, unsafe.Pointer, unsafe.Pointer, *VkDevice) VkResult)
					reflect.ValueOf(target).Elem().Set(reflect.ValueOf(func(d VkPhysicalDevice, p, a unsafe.Pointer, out *VkDevice) VkResult {
						next := *(*unsafe.Pointer)(unsafe.Add(p, 8))
						if next == nil || (*vkIntegerDotFeatures)(next).enabled != 1 || (*vkIntegerDotFeatures)(next).sType != 1000280000 {
							t.Fatal("enabled chain")
						}
						return original(d, p, a, out)
					}))
					return nil
				}
				return bind(target, lib, name)
			}
			if got := vkInitLockedMode(loader, true); got != (arm == "ok") {
				t.Fatal("init", arm, got, *events)
			}
			if arm == "ok" {
				if !VulkanIntegerDotEnabled() {
					t.Fatal("publish")
				}
			} else {
				if vkReady || vkIntegerDotEnabled || vkGetPhysicalDeviceFeatures2 != nil || vkGetPhysicalDeviceProperties2 != nil {
					t.Fatal("rollback")
				}
			}
		})
	}
	offlineInitState(t)
	mockVK(t, &vkReady, true)
	opened := false
	if vkInitLockedMode(vkLoader{open: func() (uintptr, error) { opened = true; return 0, nil }}, true) || opened {
		t.Fatal("baseline upgrade")
	}
}

func TestVulkanNativeIntegerDotProbe(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_TEST_INTDOT_SPV")
	if path == "" {
		t.Skip("explicit integer-dot probe")
	}
	d, ok := t.Deadline()
	if !ok || time.Until(d) > 120*time.Second {
		t.Fatal("bounded120s")
	}
	if !VulkanInitIntegerDot() || !VulkanIntegerDotEnabled() || os.Getenv("GO_PHERENCE_VULKAN_DEVICE") == "" || !strings.Contains(VulkanDeviceName(), os.Getenv("GO_PHERENCE_VULKAN_DEVICE")) {
		t.Fatal("accelerated physical mixed-dot init")
	}
	code, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := VkKernelCreate(code, 3, 4); err == nil {
		t.Fatal("baseline admits optional kernel")
	}
	words := spirvTestWords(code)
	contract, err := vkInspectSPIRVMode(words, true)
	if err != nil || !contract.IntegerDot {
		t.Fatal(contract, err)
	}
	kernel, err := VkKernelCreateIntegerDot(code, 3, 4)
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, kernel)
	before := VulkanMemoryStats()
	t.Cleanup(func() { nativeEncoderMemoryCheck(t, before) })
	a := nativeArena(t)
	left := nativeGuard(t, a)
	const count = 1025
	rawA, rawB := make([]float32, count), make([]float32, count)
	want := make([]int32, count)
	for i := 0; i < count; i++ {
		var u, s uint32
		for j := 0; j < 4; j++ {
			w := byte((i*17 + j*7) % 32)
			x := int8((i*29+j*13)%256 - 128)
			u |= uint32(w) << uint(j*8)
			s |= uint32(uint8(x)) << uint(j*8)
			want[i] += int32(w) * int32(x)
		}
		rawA[i] = math.Float32frombits(u)
		rawB[i] = math.Float32frombits(s)
	}
	tx, tw, out := nativeTensor(t, a, rawA, count), nativeTensor(t, a, rawB, count), nativeTensor(t, a, nil, count)
	right := nativeGuard(t, a)
	plan, err := NewVkF32Plan(context.Background(), []VkF32Stage{{Kernel: kernel, Groups: [3]uint32{(count + 63) / 64, 1, 1}, Tensors: []*VkTensorF32{tx, tw, out}, PushWords: []uint32{count}}})
	if err != nil {
		t.Fatal(err)
	}
	nativeClose(t, plan)
	for repeat := 0; repeat < 5; repeat++ {
		nativeRun(t, plan.Run)
		got := nativeDownload(t, out)
		for i, v := range got {
			if int32(math.Float32bits(v)) != want[i] {
				t.Fatal("packeddot", i, int32(math.Float32bits(v)), want[i])
			}
		}
		left()
		right()
	}
	t.Logf("INTDOT_PROBE values=%d repeat=5 device=%s", count, VulkanDeviceName())
}

func TestVulkanOfflineIntegerDotShaderContract(t *testing.T) {
	for _, name := range []string{"dot", "q8", "q5q8", "linear"} {
		t.Run(name, func(t *testing.T) {
			file := name + "-stripped.spv"
			if name == "dot" {
				file = "dot.spv"
			}
			code, err := os.ReadFile("testdata/integer-dot/" + file)
			if err != nil {
				t.Fatal(err)
			}
			if _, err = InspectVulkanShader(code); !errors.Is(err, ErrVulkanShaderContract) {
				t.Fatal("baseline admission", err)
			}
			got, err := vkInspectSPIRVMode(spirvTestWords(code), true)
			if err != nil || got.IntegerDot != (name != "q8") {
				t.Fatal(got, err)
			}
			if name == "dot" {
				for _, arm := range []string{"format", "type", "unsigned-only", "signed-only", "duplicate-cap", "missing-cap", "missing-ext", "wrong-ext", "no-enabled-device"} {
					t.Run(arm, func(t *testing.T) {
						modified := append([]byte(nil), code...)
						switch arm {
						case "format":
							modified = editSPIRV(code, 4452, func(w []uint32) { w[5] = 1 })
						case "type":
							modified = editSPIRV(code, 4452, func(w []uint32) { w[3] = w[4] })
						case "unsigned-only":
							modified = editSPIRV(code, 4452, func(w []uint32) { w[0] = 6<<16 | 4451 })
						case "signed-only":
							modified = editSPIRV(code, 4452, func(w []uint32) { w[0] = 6<<16 | 4450 })
						case "duplicate-cap":
							modified = append(modified, spirvTestBytes([]uint32{2<<16 | 17, 6018})...)
						case "missing-cap":
							w := spirvTestWords(code)
							for i := 5; i < len(w); i += int(w[i] >> 16) {
								if w[i]&65535 == 17 && w[i+1] == 6018 {
									w = append(w[:i], w[i+2:]...)
									break
								}
							}
							modified = spirvTestBytes(w)
						case "missing-ext":
							w := spirvTestWords(code)
							for i := 5; i < len(w); i += int(w[i] >> 16) {
								if w[i]&65535 == 10 {
									w = append(w[:i], w[i+int(w[i]>>16):]...)
									break
								}
							}
							modified = spirvTestBytes(w)
						case "wrong-ext":
							modified = editSPIRV(code, 10, func(w []uint32) { w[1] = 0 })
						case "no-enabled-device":
							offlineInitState(t)
							if _, e := VkKernelCreateIntegerDot(code, 3, 4); e == nil {
								t.Fatal("baseline device")
							}
							return
						}
						if _, e := vkInspectSPIRVMode(spirvTestWords(modified), true); !errors.Is(e, ErrVulkanShaderContract) {
							t.Fatal("not rejected", e)
						}
					})
				}
			}
		})
	}
}
