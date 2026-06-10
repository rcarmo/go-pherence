package pathutil

// BaseName returns the final path component for slash- or backslash-separated
// model paths, trimming trailing separators first.
func BaseName(path string) string {
	for len(path) > 0 && (path[len(path)-1] == '/' || path[len(path)-1] == '\\') {
		path = path[:len(path)-1]
	}
	last := 0
	for i := range path {
		if path[i] == '/' || path[i] == '\\' {
			last = i + 1
		}
	}
	if last >= len(path) {
		return path
	}
	return path[last:]
}
