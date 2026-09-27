package decider

import "fmt"

// validateTemperaturesByType accepts only the three public answer types.
// A missing entry uses the runtime's scalar temperature. This map is only
// applied by the state-first SystemOne path; schema-first is not implemented.
func validateTemperaturesByType(temperatures map[QuestionType]float32) error {
	for kind, temperature := range temperatures {
		switch kind {
		case Choice, Noul, Score:
		default:
			return fmt.Errorf("decider: unknown temperature_by_type key %q", kind)
		}
		if temperature <= 0 || !isFinite(temperature) {
			return fmt.Errorf("decider: temperature_by_type[%q] must be finite and positive", kind)
		}
	}
	return nil
}

func (r *Runtime) temperatureForType(kind QuestionType) float32 {
	if r != nil {
		if temperature, ok := r.temperatureByType[kind]; ok {
			return temperature
		}
		return r.Temperature
	}
	return 0
}
