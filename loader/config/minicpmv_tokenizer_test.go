package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadMiniCPMVTokenizerMetadata(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{
  "tokenizer_class":"Qwen2Tokenizer",
  "bos_token":{"content":"<s>"},
  "eos_token":"</s>",
  "pad_token":"<pad>",
  "chat_template":"{% for message in messages %}{{ message['role'] }}{% endfor %}",
  "image_token":"<image>",
  "im_start":"<im_start>",
  "im_end":"<im_end>",
  "im_patch":"<im_patch>"
}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{
  "added_tokens":[
    {"id":151640,"content":"<im_start>"},
    {"id":151641,"content":"<im_end>"},
    {"id":151642,"content":"<im_patch>"},
    {"id":151643,"content":"<image>"}
  ],
  "model":{"vocab":{}}
}`), 0o644); err != nil {
		t.Fatal(err)
	}
	meta, ok, err := ReadMiniCPMVTokenizerMetadata(dir)
	if err != nil || !ok {
		t.Fatalf("ReadMiniCPMVTokenizerMetadata ok=%v err=%v", ok, err)
	}
	if meta.TokenizerClass != "Qwen2Tokenizer" || meta.BOS != "<s>" || meta.EOS != "</s>" || meta.ChatTemplateBytes == 0 {
		t.Fatalf("bad tokenizer fields: %+v", meta)
	}
	if meta.TokenIDs["<im_start>"] != 151640 || meta.TokenIDs["<image>"] != 151643 {
		t.Fatalf("bad token ids: %+v", meta.TokenIDs)
	}
}

func TestReadMiniCPMVTokenizerMetadataMissing(t *testing.T) {
	_, ok, err := ReadMiniCPMVTokenizerMetadata(t.TempDir())
	if err != nil || ok {
		t.Fatalf("missing tokenizer ok=%v err=%v", ok, err)
	}
}
