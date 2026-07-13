package mosstranscribe

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

func TranscriptJSON(segments []TranscriptSegment) ([]byte, error) {
	return json.MarshalIndent(segments, "", "  ")
}

func TranscriptSRT(segments []TranscriptSegment) string {
	var out strings.Builder
	for index, segment := range segments {
		fmt.Fprintf(&out, "%d\n%s --> %s\n[%s] %s\n\n", index+1, formatSubtitleTime(segment.Start, ',', 3), formatSubtitleTime(segment.End, ',', 3), segment.Speaker, segment.Text)
	}
	return out.String()
}

func TranscriptASS(segments []TranscriptSegment) string {
	var out strings.Builder
	out.WriteString("[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\n\n[V4+ Styles]\n")
	out.WriteString("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
	out.WriteString("Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1\n\n[Events]\n")
	out.WriteString("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
	for _, segment := range segments {
		text := strings.ReplaceAll(segment.Text, "\\", "\\\\")
		text = strings.ReplaceAll(text, "\n", "\\N")
		text = strings.ReplaceAll(text, "{", "\\{")
		text = strings.ReplaceAll(text, "}", "\\}")
		fmt.Fprintf(&out, "Dialogue: 0,%s,%s,Default,%s,0,0,0,,%s\n", formatSubtitleTime(segment.Start, '.', 2), formatSubtitleTime(segment.End, '.', 2), segment.Speaker, text)
	}
	return out.String()
}

func formatSubtitleTime(seconds float64, decimal rune, precision int) string {
	if math.IsNaN(seconds) || math.IsInf(seconds, 0) || seconds < 0 {
		seconds = 0
	}
	scale := math.Pow10(precision)
	total := int64(math.Round(seconds * scale))
	fraction := total % int64(scale)
	totalSeconds := total / int64(scale)
	hours := totalSeconds / 3600
	minutes := totalSeconds / 60 % 60
	secs := totalSeconds % 60
	return fmt.Sprintf("%02d:%02d:%02d%c%s", hours, minutes, secs, decimal, leftPadInt(fraction, precision))
}

func leftPadInt(value int64, width int) string {
	text := strconv.FormatInt(value, 10)
	if len(text) >= width {
		return text
	}
	return strings.Repeat("0", width-len(text)) + text
}
