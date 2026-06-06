#!/bin/sh
# Clear a wedged SpaceMIT NPU (stale TCM lock from a killed process) without reboot.
rm -f /dev/shm/tcm_sync_standalone && echo "NPU TCM lock cleared (/dev/shm/tcm_sync_standalone removed)"
