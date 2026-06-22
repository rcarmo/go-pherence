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
  "chat_template":"{% for message in messages %}{% if message['role'] == 'user' %}<image>{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}",
  "image_token":"<image>",
  "im_start":"<im_start>",
  "im_end":"<im_end>",
  "im_patch":"<im_patch>",
  "audio_token":"<audio>",
  "audio_start":"<audio_start>",
  "audio_end":"<audio_end>",
  "audio_patch":"<audio_patch>"
}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{
  "added_tokens":[
    {"id":151640,"content":"<im_start>"},
    {"id":151641,"content":"<im_end>"},
    {"id":151642,"content":"<im_patch>"},
    {"id":151643,"content":"<image>"},
    {"id":151650,"content":"<audio>"},
    {"id":151651,"content":"<audio_start>"},
    {"id":151652,"content":"<audio_end>"},
    {"id":151653,"content":"<audio_patch>"}
  ],
  "model":{"vocab":{}}
}`), 0o644); err != nil {
		t.Fatal(err)
	}
	meta, ok, err := ReadMiniCPMVTokenizerMetadata(dir)
	if err != nil || !ok {
		t.Fatalf("ReadMiniCPMVTokenizerMetadata ok=%v err=%v", ok, err)
	}
	if meta.TokenizerClass != "Qwen2Tokenizer" || meta.BOS != "<s>" || meta.EOS != "</s>" || meta.ChatTemplateBytes == 0 || meta.ChatTemplate == nil {
		t.Fatalf("bad tokenizer fields: %+v", meta)
	}
	if !meta.ChatTemplate.HasUserRole || !meta.ChatTemplate.HasAssistantRole || !meta.ChatTemplate.HasImageMarker {
		t.Fatalf("bad chat template metadata: %+v", meta.ChatTemplate)
	}
	if meta.Audio != "<audio>" || meta.AudioStart != "<audio_start>" || meta.AudioEnd != "<audio_end>" || meta.AudioPatch != "<audio_patch>" {
		t.Fatalf("bad audio token strings: %+v", meta)
	}
	if meta.TokenIDs["<im_start>"] != 151640 || meta.TokenIDs["<image>"] != 151643 || meta.TokenIDs["<audio_patch>"] != 151653 {
		t.Fatalf("bad token ids: %+v", meta.TokenIDs)
	}
}

func TestReadMiniCPMVTokenizerMetadataMissing(t *testing.T) {
	_, ok, err := ReadMiniCPMVTokenizerMetadata(t.TempDir())
	if err != nil || ok {
		t.Fatalf("missing tokenizer ok=%v err=%v", ok, err)
	}
}
