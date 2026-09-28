// Independent first-acoustic-frame fixture generator for the pinned Rust/Candle checkout.
// Copy to examples/oracle_acoustic.rs in TrevorS/qwen3-tts-rs@711ceee0 and run
// with --no-default-features --features cpu; keep weights and output outside git.
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use qwen3_tts::models::{config::ParsedModelConfig, kv_cache::AnyKVCache, talker::{Language, Speaker, TalkerConfig, TalkerModel}, code_predictor::{CodePredictor, CodePredictorConfig}};
use std::{collections::HashMap, env, fs, path::Path};

fn main() -> Result<()> {
    let root = env::args().nth(1).context("model directory required")?;
    let out = env::args().nth(2).context("output directory required")?;
    fs::create_dir_all(&out)?;
    let cfg = ParsedModelConfig::from_file(&Path::new(&root).join("config.json"))?;
    let device = Device::Cpu;
    let all: HashMap<String, Tensor> = candle_core::safetensors::load(Path::new(&root).join("model.safetensors"), &device)?;
    let mut talker_weights = HashMap::new();
    for (k, v) in all.into_iter().filter(|(k,_)| k.starts_with("talker.")) {
        talker_weights.insert(k, if v.dtype() == DType::BF16 { v.to_dtype(DType::F32)? } else { v });
    }
    let cp_weights: HashMap<_,_> = talker_weights.iter().filter_map(|(k,v)| k.strip_prefix("talker.code_predictor.").map(|name| (name.to_string(),v.clone()))).collect();
    let predictor = CodePredictor::new(CodePredictorConfig::from_parsed(&cfg), VarBuilder::from_tensors(cp_weights,DType::F32,&device))?;
    let model = TalkerModel::from_weights_with_config_dtype(&talker_weights, TalkerConfig::from_parsed(&cfg), &device, DType::F32)?;
    let mut cache: Vec<AnyKVCache> = model.new_kv_caches(0);
    let (hidden, logits) = model.prefill_custom_voice(&[9707,1879], Speaker::Ryan, Language::English, &mut cache)?;
    let last: Vec<f32> = hidden.i((0, 9, ..))?.to_vec1()?;
    let scores: Vec<f32> = logits.i((0, 0, ..))?.to_vec1()?;
    let eos = 2150usize;
    let low = scores.len() - 1024;
    let first = (0..scores.len()).filter(|&i| i < low || i == eos)
        .max_by(|&a,&b| scores[a].total_cmp(&scores[b])).context("no candidate")?;
    let semantic_embed = model.get_codec_embedding(first as u32)?;
    let last_hidden = hidden.i((..,9..10,..))?;
    let mut cp_caches = predictor.new_kv_caches();
    let cp_input = Tensor::cat(&[&last_hidden, &semantic_embed], 1)?;
    let cp_prefill = predictor.forward_prefill(&cp_input, &[], &mut cp_caches)?;
    let first_logits: Vec<f32> = predictor.get_logits(&cp_prefill, 0, 1)?.i((0, 0, ..))?.to_vec1()?;
    fs::write(Path::new(&out).join("acoustic_first_logits.f32le"), first_logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let acoustic: Vec<u32> = predictor.generate_acoustic_codes(&last_hidden, &semantic_embed, &mut cp_caches)?.to_vec1()?;
    println!("first_acoustic_frame={acoustic:?}");
    let bytes = |floats: &[f32]| floats.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    fs::write(Path::new(&out).join("acoustic.u32le"), acoustic.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    fs::write(Path::new(&out).join("hidden.f32le"), bytes(&last))?;
    fs::write(Path::new(&out).join("logits.f32le"), bytes(&scores))?;
    println!("hidden={} logits={} first_semantic={} first_logit={}", last.len(), scores.len(), first, scores[first]);
    let mut top: Vec<usize> = (0..low).collect();
    top.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    for &i in top.iter().take(8) { println!("rank {} logit {:.9}", i, scores[i]); }
    Ok(())
}
