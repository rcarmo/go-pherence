package community1

// ReleaseVulkanHostWeights clears host neural tensor references after a
// successful VulkanDiarization construction has copied all parameters. The
// caller must exclusively own both sources and must not run CPU inference or
// observers concurrently. SincNet lowered filters and PLDA stay in the hybrid
// owner; this releases only the source checkpoint's recurrent/head/SincNet
// parameter arrays and the source embedding trunk/projection arrays.
//
// It is idempotent and only makes storage eligible for GC. It does not force GC,
// promise an RSS reduction, close Vulkan resources, or alter the resident owner.
func ReleaseVulkanHostWeights(segmentation *SegmentationCheckpoint, embedding *WeSpeakerResNet34) {
	if segmentation != nil {
		clearSincNetWeights(&segmentation.sincnet)
		if segmentation.recurrent != nil {
			for i := range segmentation.recurrent.layers {
				clearLSTMWeights(&segmentation.recurrent.layers[i].Forward)
				clearLSTMWeights(&segmentation.recurrent.layers[i].Reverse)
			}
			segmentation.recurrent.layers = nil
		}
		if segmentation.head != nil {
			for i := range segmentation.head.layers {
				clearHeadLinear(&segmentation.head.layers[i])
			}
			segmentation.head.layers = nil
			clearHeadLinear(&segmentation.head.classifier)
		}
	}
	if embedding != nil {
		clear(embedding.stem)
		embedding.stem = nil
		clearWeSpeakerBN(&embedding.stemBN)
		for stage := range embedding.stages {
			for _, block := range embedding.stages[stage] {
				if block != nil {
					clearWeSpeakerBlockWeights(&block.weights)
				}
			}
			embedding.stages[stage] = nil
		}
		clearHeadLinear(&embedding.projection)
	}
}

func clearSincNetWeights(w *SincNetWeights) {
	if w == nil {
		return
	}
	clear(w.WaveNorm.Weight)
	clear(w.WaveNorm.Bias)
	w.WaveNorm = SincNetNorm{}
	clear(w.LowHz)
	clear(w.BandHz)
	w.LowHz, w.BandHz = nil, nil
	for i := range w.Conv {
		clear(w.Conv[i].Weight)
		clear(w.Conv[i].Bias)
		w.Conv[i] = SincNetConv{}
	}
	for i := range w.Norm {
		clear(w.Norm[i].Weight)
		clear(w.Norm[i].Bias)
		w.Norm[i] = SincNetNorm{}
	}
}
func clearLSTMWeights(w *LSTMWeights) {
	if w == nil {
		return
	}
	clear(w.WeightIH)
	clear(w.WeightHH)
	clear(w.BiasIH)
	clear(w.BiasHH)
	*w = LSTMWeights{}
}
func clearHeadLinear(w *HeadLinear) {
	if w == nil {
		return
	}
	clear(w.Weight)
	clear(w.Bias)
	*w = HeadLinear{}
}
func clearWeSpeakerBN(w *WeSpeakerBN) {
	if w == nil {
		return
	}
	clear(w.Weight)
	clear(w.Bias)
	clear(w.RunningMean)
	clear(w.RunningVariance)
	*w = WeSpeakerBN{}
}
func clearWeSpeakerBlockWeights(w *WeSpeakerBlockWeights) {
	if w == nil {
		return
	}
	clear(w.Conv1)
	clear(w.Conv2)
	clear(w.Shortcut)
	w.Conv1, w.Conv2, w.Shortcut = nil, nil, nil
	clearWeSpeakerBN(&w.BN1)
	clearWeSpeakerBN(&w.BN2)
	clearWeSpeakerBN(&w.ShortcutBN)
}
