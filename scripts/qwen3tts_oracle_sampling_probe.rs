// Independent bounded sampler fixture for the pinned Rust/Candle CPU implementation.
use anyhow::Result;
use candle_core::{Device, Tensor, IndexOp};
use qwen3_tts::generation::{sample, GenerationConfig, SamplingContext, apply_token_suppression, apply_repetition_penalty};
use std::{env,fs,path::Path};

fn main() -> Result<()> {
    let out=env::args().nth(1).expect("output path required");
    fs::create_dir_all(&out)?;
    let device=Device::Cpu;
    let mut cases:Vec<(&str,Vec<f32>,GenerationConfig,u64,Vec<u32>)>=Vec::new();
    let default=GenerationConfig { max_new_tokens:8, temperature:0.7,top_k:50,top_p:0.9,repetition_penalty:1.0,eos_token_id:Some(2150),min_new_tokens:2 };
    cases.push(("topk_ties",vec![4.,3.,3.,2.,1.], GenerationConfig{top_k:2,top_p:1.0,..default.clone()},42,vec![]));
    cases.push(("topp_order",vec![1.,2.,1.,1.,0.], GenerationConfig{top_k:0,top_p:0.7,..default.clone()},42,vec![]));
    cases.push(("greedy",vec![1.,4.,3.,2.,0.],GenerationConfig{temperature:0.0,top_k:1,top_p:0.1,..default.clone()},42,vec![]));
    cases.push(("repetition",vec![3.,2.,4.,1.,0.],GenerationConfig{repetition_penalty:1.2,top_k:3,top_p:0.9,..default.clone()},7,vec![2]));
    let mut rows=Vec::new();
    for (name, logits, config, seed, generated) in cases {
        let mut row=Tensor::new(logits.as_slice(),&device)?.unsqueeze(0)?;
        if !generated.is_empty() { row=apply_repetition_penalty(&row,&Tensor::new(generated.as_slice(),&device)?,config.repetition_penalty)?; }
        let mut ctx=SamplingContext::new(Some(seed));
        let mut draws=Vec::new();
        for _ in 0..6 { draws.push(sample(&row,&config,&mut ctx)?.to_vec1::<u32>()?[0]); }
        let adjusted:Vec<f32>=row.i(0)?.to_vec1()?;
        rows.push(serde_json::json!({"name":name,"input_logits":logits,"adjusted_logits":adjusted,"temperature":config.temperature,"top_k":config.top_k,"top_p":config.top_p,"repetition_penalty":config.repetition_penalty,"generated":generated,"seed":seed,"tokens":draws}));
    }
    let mut tts=vec![0.0f32;3072];tts[2051]=100.;tts[2150]=10.;tts[42]=2.;
    let suppressed=apply_token_suppression(&Tensor::new(tts.as_slice(),&device)?.unsqueeze(0)?,3072,2150)?;
    let values:Vec<f32>=suppressed.i(0)?.to_vec1()?;
    let meta=serde_json::json!({"cases":rows,"tts_control":{"kept_eos":values[2150],"kept_acoustic":values[42],"suppressed_control_is_negative_infinity":values[2051].is_infinite()&&values[2051].is_sign_negative()}});
    fs::write(Path::new(&out).join("sampling_probe.json"),serde_json::to_vec_pretty(&meta)?)?;
    println!("{}",serde_json::to_string_pretty(&meta)?);
    Ok(())
}
