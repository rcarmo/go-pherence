// Independent capped-sentence oracle for TrevorS/qwen3-tts-rs
// @711ceee07cad92673f86de8997bdf54c30caa49f. CPU only; EOS remains enabled.
// Copy to examples/probe_sentence_64.rs; build --no-default-features --features cpu.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use qwen3_tts::generation::{self, GenerationConfig, SamplingContext};
use qwen3_tts::models::{config::ParsedModelConfig, kv_cache::AnyKVCache,
    talker::{Language, Speaker, TalkerConfig, TalkerModel},
    code_predictor::{CodePredictor, CodePredictorConfig}};
use qwen3_tts::tokenizer::TextTokenizer;
use std::{collections::HashMap, env, fs, path::Path};

const TEXT: &str = "The quick brown fox jumps over the lazy dog and then walks slowly back to the house, where a warm dinner is waiting.";

fn choose(row: &[f32], device: &Device, cfg: &GenerationConfig,
          rng: &mut SamplingContext, selected: usize) -> Result<u32> {
    let raw = Tensor::new(row, device)?.unsqueeze(0)?;
    let mut values: Vec<f32> = generation::apply_token_suppression(&raw, 3072, 2150)?.i(0)?.to_vec1()?;
    if selected < cfg.min_new_tokens { values[2150] = f32::NEG_INFINITY; }
    let filtered = Tensor::new(values.as_slice(), device)?.unsqueeze(0)?;
    Ok(generation::sample(&filtered, cfg, rng)?.to_vec1::<u32>()?[0])
}

fn main() -> Result<()> {
    let root = env::args().nth(1).context("model directory required")?;
    let out = env::args().nth(2).context("output directory required")?;
    fs::create_dir_all(&out)?;
    let cfg = ParsedModelConfig::from_file(&Path::new(&root).join("config.json"))?;
    let tokenizer = TextTokenizer::from_pretrained(&root)?;
    let text_ids = tokenizer.encode(TEXT)?;
    ensure!(!text_ids.is_empty(), "empty text IDs");
    let device = Device::Cpu;
    let all: HashMap<String, Tensor> = candle_core::safetensors::load(Path::new(&root).join("model.safetensors"), &device)?;
    let mut weights = HashMap::new();
    for (name, value) in all.into_iter().filter(|(name, _)| name.starts_with("talker.")) {
        weights.insert(name, if value.dtype() == DType::BF16 { value.to_dtype(DType::F32)? } else { value });
    }
    let cp_weights: HashMap<_, _> = weights.iter().filter_map(|(name, value)|
        name.strip_prefix("talker.code_predictor.").map(|name| (name.to_string(), value.clone()))).collect();
    let predictor = CodePredictor::new(CodePredictorConfig::from_parsed(&cfg), VarBuilder::from_tensors(cp_weights, DType::F32, &device))?;
    let talker = TalkerModel::from_weights_with_config_dtype(&weights, TalkerConfig::from_parsed(&cfg), &device, DType::F32)?;
    let mut cache: Vec<AnyKVCache> = talker.new_kv_caches(0);
    let (prefill_hidden, prefill_logits) = talker.prefill_custom_voice(&text_ids, Speaker::Ryan, Language::English, &mut cache)?;
    let prefix_len = prefill_hidden.dim(1)?;
    ensure!(prefix_len == 10, "unexpected CustomVoice prefix");
    let mut hidden = prefill_hidden.i((.., prefix_len-1..prefix_len, ..))?;
    let mut logits = prefill_logits;
    let eos_text = talker.get_tts_eos_embed()?;
    let pad_text = talker.get_tts_pad_embed()?;
    let remaining: Vec<u32> = text_ids.iter().skip(1).copied().collect();
    let trailing = if remaining.is_empty() { eos_text } else {
        Tensor::cat(&[&talker.get_projected_text_embeddings(&remaining)?, &eos_text], 1)?
    };
    let gen_cfg = GenerationConfig {max_new_tokens:64, temperature:0.7, top_k:50, top_p:0.9,
        repetition_penalty:1.0, eos_token_id:Some(2150), min_new_tokens:2};
    let mut rng = SamplingContext::new(Some(42));
    let mut cp_cache = predictor.new_kv_caches();
    let mut frames: Vec<Vec<u32>> = Vec::new();
    let mut eos_step: Option<usize> = None;
    for step in 0..64 {
        let row: Vec<f32> = logits.i((0, 0, ..))?.to_vec1()?;
        let next = choose(&row, &device, &gen_cfg, &mut rng, step)?;
        if next == 2150 { eos_step = Some(step); break; }
        let semantic = talker.get_codec_embedding(next)?;
        let acoustic: Vec<u32> = predictor.generate_acoustic_codes(&hidden, &semantic, &mut cp_cache)?.to_vec1()?;
        ensure!(acoustic.len() == 15, "acoustic width");
        let mut frame = vec![next]; frame.extend(&acoustic); frames.push(frame);
        println!("frame={step} semantic={next}");
        if step == 63 { break; }
        let text = if step < trailing.dim(1)? { trailing.i((.., step..step+1, ..))? } else { pad_text.clone() };
        let input = semantic.add(&predictor.get_acoustic_embeddings_sum(&acoustic, &device)?)?.add(&text)?;
        (hidden, logits) = talker.generate_step_with_embed(&input, &mut cache, prefix_len+step)?;
    }
    ensure!(!frames.is_empty(), "EOS before first frame");
    let codes = qwen3_tts::codes_to_tensor(&frames, &device)?;
    let tokenizer_weights: HashMap<String, Tensor> = candle_core::safetensors::load(Path::new(&root).join("speech_tokenizer/model.safetensors"), &device)?;
    let decoder_weights: HashMap<_, _> = tokenizer_weights.into_iter().filter(|(k, _)| k.starts_with("decoder."))
        .map(|(k, v)| Ok((k, if v.dtype()==DType::BF16 {v.to_dtype(DType::F32)?} else {v}))).collect::<Result<_>>()?;
    let decoder = qwen3_tts::models::codec::Decoder12Hz::from_weights(&decoder_weights, Default::default())?;
    let wave: Vec<f32> = decoder.decode(&codes)?.flatten_all()?.to_vec1()?;
    let observation = serde_json::json!({"text":TEXT,"speaker":"ryan","language":"en","seed":42,"cap":64,
        "frames":frames.len(),"eos_step":eos_step,"eos_token_id":2150,"samples":wave.len(),"text_ids":text_ids});
    fs::write(Path::new(&out).join("sentence_64_observation.json"), format!("{}\n", serde_json::to_string(&observation)?))?;
    fs::write(Path::new(&out).join("sentence_64_codes.u32le"), frames.iter().flatten().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    fs::write(Path::new(&out).join("sentence_64_waveform.f32le"), wave.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    println!("frames={} eos_step={eos_step:?} samples={}", frames.len(), wave.len());
    Ok(())
}
