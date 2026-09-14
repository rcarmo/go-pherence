package whisper

// ReleaseHostWeights clears this caller-owned encoder's Go tensor references.
// Use only after an independently owned resident encoder has copied all weights
// and while no host inference/observer can access this Encoder. It is idempotent.
// This makes storage eligible for GC; it does not force GC or promise RSS drop.
func (enc *Encoder) ReleaseHostWeights() {
	if enc == nil {
		return
	}
	clear(enc.Conv1Weight)
	enc.Conv1Weight = nil
	clear(enc.Conv1Bias)
	enc.Conv1Bias = nil
	clear(enc.Conv2Weight)
	enc.Conv2Weight = nil
	clear(enc.Conv2Bias)
	enc.Conv2Bias = nil
	clear(enc.PosEmbed)
	enc.PosEmbed = nil
	clear(enc.FinalLNWeight)
	enc.FinalLNWeight = nil
	clear(enc.FinalLNBias)
	enc.FinalLNBias = nil
	for i := range enc.Layers {
		l := &enc.Layers[i]
		clear(l.AttnLNWeight)
		l.AttnLNWeight = nil
		clear(l.AttnLNBias)
		l.AttnLNBias = nil
		clear(l.QWeight)
		l.QWeight = nil
		clear(l.QBias)
		l.QBias = nil
		clear(l.KWeight)
		l.KWeight = nil
		clear(l.KBias)
		l.KBias = nil
		clear(l.VWeight)
		l.VWeight = nil
		clear(l.VBias)
		l.VBias = nil
		clear(l.OWeight)
		l.OWeight = nil
		clear(l.OBias)
		l.OBias = nil
		clear(l.MLPLNWeight)
		l.MLPLNWeight = nil
		clear(l.MLPLNBias)
		l.MLPLNBias = nil
		clear(l.FC1Weight)
		l.FC1Weight = nil
		clear(l.FC1Bias)
		l.FC1Bias = nil
		clear(l.FC2Weight)
		l.FC2Weight = nil
		clear(l.FC2Bias)
		l.FC2Bias = nil
	}
	enc.Layers = nil
}
