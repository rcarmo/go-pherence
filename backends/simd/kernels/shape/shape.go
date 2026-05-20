package shape

func shapeSizeChecked(op string, shape []int) int {
	total := 1
	for _, d := range shape {
		if d < 0 || (d != 0 && total > int(^uint(0)>>1)/d) {
			panic(op + ": invalid shape")
		}
		total *= d
	}
	return total
}
