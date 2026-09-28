// Temporary, untracked independent Talker prefill oracle; output outside the repository.
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use qwen3_tts::models::{config::ParsedModelConfig, kv_cache::AnyKVCache, talker::{Language, Speaker, TalkerConfig, TalkerModel}};
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
    let model = TalkerModel::from_weights_with_config_dtype(&talker_weights, TalkerConfig::from_parsed(&cfg), &device, DType::F32)?;
    let mut cache: Vec<AnyKVCache> = model.new_kv_caches(0);
    let (hidden, logits) = model.prefill_custom_voice(&[9707,1879], Speaker::Ryan, Language::English, &mut cache)?;
    let last: Vec<f32> = hidden.i((0, 9, ..))?.to_vec1()?;
    let scores: Vec<f32> = logits.i((0, 0, ..))?.to_vec1()?;
    let eos = 2150usize;
    let low = scores.len() - 1024;
    let first = (0..scores.len()).filter(|&i| i < low || i == eos)
        .max_by(|&a,&b| scores[a].total_cmp(&scores[b])).context("no candidate")?;
    let bytes = |floats: &[f32]| floats.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    fs::write(Path::new(&out).join("hidden.f32le"), bytes(&last))?;
    fs::write(Path::new(&out).join("logits.f32le"), bytes(&scores))?;
    println!("hidden={} logits={} first_semantic={} first_logit={}", last.len(), scores.len(), first, scores[first]);
    let mut top: Vec<usize> = (0..low).collect();
    top.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    for &i in top.iter().take(8) { println!("rank {} logit {:.9}", i, scores[i]); }
    Ok(())
}
