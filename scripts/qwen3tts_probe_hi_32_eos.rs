// Bounded Rust/Candle CPU observation of natural EOS up to 32 frames for Hi.
// Copy to examples/probe_hi_32_eos.rs in TrevorS/qwen3-tts-rs@711ceee0;
// run with --no-default-features --features cpu, keeping weights outside git.
use anyhow::{ensure, Context, Result};
use qwen3_tts::generation::{self, GenerationConfig, SamplingContext};
fn choose(row:&[f32],dev:&Device,cfg:&GenerationConfig,rng:&mut SamplingContext,selected:usize)->Result<usize>{
    let raw=Tensor::new(row,dev)?.unsqueeze(0)?;
    let filtered=generation::apply_token_suppression(&raw,3072,2150)?;
    let filtered=if selected<cfg.min_new_tokens {
        let mut values:Vec<f32>=filtered.i(0)?.to_vec1()?;
        values[2150]=f32::NEG_INFINITY;
        Tensor::new(values.as_slice(),dev)?.unsqueeze(0)?
    } else {filtered};
    Ok(generation::sample(&filtered,cfg,rng)?.to_vec1::<u32>()?[0] as usize)
}
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
    let (hidden, logits) = model.prefill_custom_voice(&[13048], Speaker::Ryan, Language::English, &mut cache)?;
    let last: Vec<f32> = hidden.i((0, 9, ..))?.to_vec1()?;
    let scores: Vec<f32> = logits.i((0, 0, ..))?.to_vec1()?;
    let gen_cfg=GenerationConfig{max_new_tokens:32,temperature:0.7,top_k:50,top_p:0.9,repetition_penalty:1.0,eos_token_id:Some(2150),min_new_tokens:2};
    let mut rng=SamplingContext::new(Some(42));
    let eos = 2150usize;
    let low = scores.len() - 1024;
    let first = choose(&scores,&device,&gen_cfg,&mut rng,0)?;
    ensure!(first != eos, "EOS before first acoustic frame");
    let semantic_embed = model.get_codec_embedding(first as u32)?;
    let last_hidden = hidden.i((..,9..10,..))?;
    let mut cp_caches = predictor.new_kv_caches();
    let cp_input = Tensor::cat(&[&last_hidden, &semantic_embed], 1)?;
    let cp_prefill = predictor.forward_prefill(&cp_input, &[], &mut cp_caches)?;
    let first_logits: Vec<f32> = predictor.get_logits(&cp_prefill, 0, 1)?.i((0, 0, ..))?.to_vec1()?;
    fs::write(Path::new(&out).join("acoustic_first_logits.f32le"), first_logits.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let acoustic: Vec<u32> = predictor.generate_acoustic_codes(&last_hidden, &semantic_embed, &mut cp_caches)?.to_vec1()?;
    println!("first_acoustic_frame={acoustic:?}");
    let acoustic_sum = predictor.get_acoustic_embeddings_sum(&acoustic, &device)?;
    let text = model.get_tts_eos_embed()?;
    let semantic_sum = model.get_codec_embedding(first as u32)?.add(&acoustic_sum)?;
    let next_input = semantic_sum.add(&text)?;
    let (second_hidden, second_logits) = model.generate_step_with_embed(&next_input, &mut cache, 10)?;
    let second_row: Vec<f32> = second_logits.i((0,0,..))?.to_vec1()?;
    let second = choose(&second_row,&device,&gen_cfg,&mut rng,1)?;
    ensure!(second != eos, "EOS before second acoustic frame");
    println!("second_semantic={second} second_logit={}",second_row[second]);
    fs::write(Path::new(&out).join("second_hidden.f32le"),second_hidden.i((0,0,..))?.to_vec1::<f32>()?.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    fs::write(Path::new(&out).join("second_logits.f32le"),second_row.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let second_embed=model.get_codec_embedding(second as u32)?;
    let second_acoustic: Vec<u32> = predictor.generate_acoustic_codes(&second_hidden,&second_embed,&mut cp_caches)?.to_vec1()?;
    println!("second_acoustic_frame={second_acoustic:?}");
    fs::write(Path::new(&out).join("second_acoustic.u32le"),second_acoustic.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let second_sum=predictor.get_acoustic_embeddings_sum(&second_acoustic,&device)?;
    let eos_text=model.get_tts_pad_embed()?;
    let third_input=model.get_codec_embedding(second as u32)?.add(&second_sum)?.add(&eos_text)?;
    let (third_hidden,third_logits)=model.generate_step_with_embed(&third_input,&mut cache,11)?;
    let third_row: Vec<f32>=third_logits.i((0,0,..))?.to_vec1()?;
    let third=choose(&third_row,&device,&gen_cfg,&mut rng,2)?;
    ensure!(third != eos, "EOS before third acoustic frame");
    println!("third_semantic={third} third_logit={}",third_row[third]);
    fs::write(Path::new(&out).join("third_hidden.f32le"),third_hidden.i((0,0,..))?.to_vec1::<f32>()?.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    fs::write(Path::new(&out).join("third_logits.f32le"),third_row.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let third_codes: Vec<u32>=predictor.generate_acoustic_codes(&third_hidden,&model.get_codec_embedding(third as u32)?,&mut cp_caches)?.to_vec1()?;
    println!("third_acoustic_frame={third_codes:?}");
    fs::write(Path::new(&out).join("third_acoustic.u32le"),third_codes.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let third_sum=predictor.get_acoustic_embeddings_sum(&third_codes,&device)?;
    let pad_text=model.get_tts_pad_embed()?;
    let fourth_input=model.get_codec_embedding(third as u32)?.add(&third_sum)?.add(&pad_text)?;
    let (fourth_hidden,fourth_logits)=model.generate_step_with_embed(&fourth_input,&mut cache,12)?;
    let fourth_row: Vec<f32>=fourth_logits.i((0,0,..))?.to_vec1()?;
    let fourth=choose(&fourth_row,&device,&gen_cfg,&mut rng,3)?;
    ensure!(fourth != eos, "EOS before fourth acoustic frame");
    println!("fourth_semantic={fourth} fourth_logit={}",fourth_row[fourth]);
    fs::write(Path::new(&out).join("fourth_hidden.f32le"),fourth_hidden.i((0,0,..))?.to_vec1::<f32>()?.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    fs::write(Path::new(&out).join("fourth_logits.f32le"),fourth_row.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let fourth_codes:Vec<u32>=predictor.generate_acoustic_codes(&fourth_hidden,&model.get_codec_embedding(fourth as u32)?,&mut cp_caches)?.to_vec1()?;
    println!("fourth_acoustic_frame={fourth_codes:?}");
    fs::write(Path::new(&out).join("fourth_acoustic.u32le"),fourth_codes.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    let mut frame_codes:Vec<Vec<u32>>=Vec::new();
    for (token,codes) in [(first as u32,&acoustic),(second as u32,&second_acoustic),(third as u32,&third_codes),(fourth as u32,&fourth_codes)] {
        let mut frame=Vec::with_capacity(16);frame.push(token);frame.extend(codes);frame_codes.push(frame);
    }
    let mut prev_semantic=fourth as u32;
    let mut prev_codes=fourth_codes;
    for frame_idx in 4..32 {
        let sum=predictor.get_acoustic_embeddings_sum(&prev_codes,&device)?;
        let input=model.get_codec_embedding(prev_semantic)?.add(&sum)?.add(&pad_text)?;
        let (h,logits)=model.generate_step_with_embed(&input,&mut cache,9+frame_idx)?;
        let row:Vec<f32>=logits.i((0,0,..))?.to_vec1()?;
        let next=choose(&row,&device,&gen_cfg,&mut rng,frame_idx)?;
        println!("frame={} semantic={next}",frame_idx);
        if next==eos { println!("EOS at frame={}",frame_idx); break; }
        let codes:Vec<u32>=predictor.generate_acoustic_codes(&h,&model.get_codec_embedding(next as u32)?,&mut cp_caches)?.to_vec1()?;
        println!("frame={} acoustic={codes:?}",frame_idx);
        let mut frame=Vec::with_capacity(16);frame.push(next as u32);frame.extend(&codes);frame_codes.push(frame);
        prev_semantic=next as u32;prev_codes=codes;
    }
    let codes=qwen3_tts::codes_to_tensor(&frame_codes, &device)?;
    let tokenizer_weights: HashMap<String, Tensor>=candle_core::safetensors::load(Path::new(&root).join("speech_tokenizer/model.safetensors"),&device)?;
    let decoder_weights: HashMap<_,_>=tokenizer_weights.into_iter().filter(|(k,_)|k.starts_with("decoder.")).map(|(k,v)|(k,if v.dtype()==DType::BF16{v.to_dtype(DType::F32).unwrap()}else{v})).collect();
    let decoder=qwen3_tts::models::codec::Decoder12Hz::from_weights(&decoder_weights,Default::default())?;
    let audio: Vec<f32>=decoder.decode(&codes)?.flatten_all()?.to_vec1()?;
    println!("probe_hi_32_samples={} frames={} first={:?}",audio.len(),frame_codes.len(),&audio[..audio.len().min(8)]);
    fs::write(Path::new(&out).join("probe_hi_32_waveform.f32le"),audio.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
    fs::write(Path::new(&out).join("probe_hi_32_codes.u32le"),frame_codes.iter().flat_map(|frame|frame.iter()).flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>())?;
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
