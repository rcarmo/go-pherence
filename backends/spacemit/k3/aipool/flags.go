package aipool

import "os"

// Int8TCMBWaveOn enables the int8 TCM B-wave staging path in the worker pool.
var Int8TCMBWaveOn = os.Getenv("IME2_INT8_TCM_B_WAVE") != ""
