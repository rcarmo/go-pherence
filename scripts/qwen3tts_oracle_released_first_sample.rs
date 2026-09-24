// Pinned Rust/Candle CPU sampler applied to the independently pinned prefill logits.
use anyhow::{Context,Result};
use candle_core::{Device,Tensor};
use qwen3_tts::generation::{sample,apply_token_suppression,GenerationConfig,SamplingContext};
use std::{env,fs,path::Path};

fn main()->Result<()> {
    let logits_path=env::args().nth(1).context("F32LE prefill logits path required")?;
    let out=env::args().nth(2).context("output directory required")?;
    fs::create_dir_all(&out)?;
    let bytes=fs::read(&logits_path)?;
    anyhow::ensure!(bytes.len()==3072*4,"expected 3072 F32LE logits");
    let floats:Vec<f32>=bytes.chunks_exact(4).map(|c|f32::from_le_bytes(c.try_into().unwrap())).collect();
    let device=Device::Cpu;
    let mut near_tie=floats.clone();
    let first=1995usize;
    let runner_up=1221usize;
    near_tie[runner_up]=near_tie[first]-0.25;
    let mut rows=Vec::new();
    for &(label,logits) in &[("released",floats.as_slice()),("controlled_near_tie",near_tie.as_slice())] {
        let row=apply_token_suppression(&Tensor::new(logits,&device)?.unsqueeze(0)?,3072,2150)?;
        for &(temperature,top_k,top_p) in &[(0.7,50usize,0.9),(0.0,50,0.9),(0.7,0,1.0)] {
            let cfg=GenerationConfig{max_new_tokens:16,temperature,top_k,top_p,repetition_penalty:1.0,eos_token_id:Some(2150),min_new_tokens:2};
            for &seed in &[42u64,7u64] {
                let mut ctx=SamplingContext::new(Some(seed));
                let mut tokens=Vec::new();
                for _ in 0..8 {tokens.push(sample(&row,&cfg,&mut ctx)?.to_vec1::<u32>()?[0]);}
                rows.push(serde_json::json!({"label":label,"temperature":temperature,"top_k":top_k,"top_p":top_p,"seed":seed,"tokens":tokens}));
            }
        }
    }
    let data=serde_json::json!({"logits_sha256":"a21ec2ea111681b69195b8bedbdd089a5ac18a64eb7eec32b76fa3cf34c90149","controlled_change":{"token_id":1221,"replacement":"f32(logit[1995] - 0.25)"},"rows":rows});
    fs::write(Path::new(&out).join("released_first_sample.json"),serde_json::to_vec_pretty(&data)?)?;
    println!("{}",serde_json::to_string_pretty(&data)?);
    Ok(())
}
