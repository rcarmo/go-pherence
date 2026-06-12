package diffusiongemma

var diffusionGemmaFP8E4M3Table = func() [256]float32 {
	var table [256]float32
	for i := 0; i < len(table); i++ {
		table[i] = fp8DecodeE4M3(byte(i))
	}
	return table
}()
