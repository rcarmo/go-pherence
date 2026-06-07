// Package promptfile loads benchmark/eval prompts for the cmd/llm speculative
// decoding commands. It is internal to cmd/llm.
package promptfile

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// Load returns the prompts to run. If promptFile is empty it returns the single
// inline prompt; otherwise it reads promptFile, one prompt per line, skipping
// blank lines and lines beginning with '#'. It errors if the file yields none.
func Load(prompt, promptFile string) ([]string, error) {
	if promptFile == "" {
		return []string{prompt}, nil
	}
	f, err := os.Open(promptFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var prompts []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		prompts = append(prompts, line)
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(prompts) == 0 {
		return nil, fmt.Errorf("no prompts in %s", promptFile)
	}
	return prompts, nil
}
