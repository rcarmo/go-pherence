# Community-1 native Vulkan qualification harness

`TestVulkanCommunityNative` and `make speech-vulkan-community-native-check` provide a fail-closed synthetic hardware gate for the newly added Community-1 Vulkan paths. The test is disabled by default and requires both `GO_PHERENCE_TEST_VULKAN_COMMUNITY=1` and an explicit `GO_PHERENCE_VULKAN_DEVICE` substring. Software Vulkan names are rejected, and the test process must have a timeout no greater than five minutes.

When authorized, the harness checks:

1. one projected WeSpeaker BasicBlock against the checked CPU GEMM path;
2. the complete resident ResNet34 trunk and hybrid embedding against CPU output/support;
3. a multi-layer bidirectional LSTM against the checked CPU SIMD path;
4. the hybrid Vulkan-LSTM/CPU-head feature stage against the CPU checkpoint path;
5. the fixed-window CPU-SincNet/Vulkan-LSTM/CPU-head composition against the existing CPU PCM path.

Fixed diagnostic tolerances are `2e-4 + 2e-4*abs(reference)` for the isolated block/LSTM and `5e-4 + 5e-4*abs(reference)` for full trunk/embedding/segmentation composition. These are conservative first-device qualification budgets, not production gates. The harness checks allocation/state return after every subtest and retries close only after an explicit bounded `VulkanDrain`.

The default test run verifies that this gate skips without opening Vulkan. It has **not** been executed on hardware or trained checkpoints in this work. Therefore no native parity, quality, performance or deployment claim follows from its existence.
