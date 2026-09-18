#!/usr/bin/env bun

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const readJSON = async (path: string) => JSON.parse(await readFile(resolve(root, path), "utf8"));
const fail = (message: string): never => { throw new Error(message); };

const freeze = await readJSON("docs/speech-quality-freeze.json");
const generation = await readJSON("docs/speech-generation-manifest.json");
const reference = await readJSON("docs/speech-reference-manifest.json");
const segmentation = await readJSON("benchmarks/speech-foundations/community1-trained-segmentation-20260912/manifest.json");
const embedding = await readJSON("benchmarks/speech-foundations/community1-trained-embedding-20260912/manifest.json");
const diarization = await readJSON("benchmarks/speech-foundations/community1-pcm-diarization-20260912/evidence.json");

if (freeze.schema !== 1 || freeze.community1.revision !== reference.source_revisions.community1) fail("Community revision drift");
if (freeze.whisper.models.length !== generation.tokenizers.length) fail("Whisper model count drift");
for (const model of freeze.whisper.models) {
  const pinned = generation.tokenizers.find((x: any) => x.repository === model.id);
  if (!pinned || ["revision", "config_sha256", "tokenizer_sha256", "generation_sha256"].some(k => pinned[k] !== model[k])) fail(`Whisper metadata drift: ${model.id}`);
}
const tensorCounts = (file: any) => Object.values(file.tensors).reduce((out: Record<string, number>, tensor: any) => {
  out[tensor.dtype] = (out[tensor.dtype] || 0) + 1;
  return out;
}, {});
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const segFile = segmentation.files["segmentation.safetensors"];
if (freeze.community1.segmentation.raw_checkpoint_sha256 !== segmentation.checkpoint_sha256 || freeze.community1.segmentation.converted_sha256 !== segFile.sha256 || freeze.community1.segmentation.tensor_count !== Object.keys(segFile.tensors).length || !same(freeze.community1.segmentation.dtypes, tensorCounts(segFile)) || freeze.community1.segmentation.lowered_filters_sha256 !== segmentation.files["filters.safetensors"].sha256) fail("Community segmentation manifest drift");
const embFile = embedding.files["embedding.safetensors"];
if (freeze.community1.embedding.raw_checkpoint_sha256 !== embedding.checkpoint_sha256 || freeze.community1.embedding.converted_sha256 !== embFile.sha256 || freeze.community1.embedding.tensor_count !== Object.keys(embFile.tensors).length || !same(freeze.community1.embedding.dtypes, tensorCounts(embFile)) || !same(freeze.community1.embedding.explicit_pinned_defaults, {base_channels: embedding.config.BaseChannels, mel_bins: embedding.config.MelBins, embed_dim: embedding.config.EmbedDim}) || freeze.community1.embedding.prefix !== embedding.prefix) fail("Community embedding manifest drift");
const plda = diarization.plda;
if (freeze.community1.plda.xvector_transform_sha256 !== plda.xvecSHA || freeze.community1.plda.plda_sha256 !== plda.pldaSHA || !same(Object.values(freeze.community1.plda.dimensions), plda.config) || freeze.community1.plda.condition_inf !== plda.conditionInf || freeze.community1.plda.inverse_residual !== plda.inverseResidual) fail("Community PLDA drift");
const policy = diarization.policy;
const expectedPolicy = {window_samples: policy.WindowSamples, step_samples: policy.StepSamples, minimum_embedding_samples: policy.MinimumEmbeddingSamples, exclude_overlap: policy.ExcludeOverlap, min_speakers: policy.MinSpeakers, max_speakers: policy.MaxSpeakers, num_speakers: policy.NumSpeakers, ahc_threshold: policy.AHCThreshold, fa: policy.Fa, fb: policy.Fb, min_duration_off: policy.MinDurationOff, constrained: policy.Constrained, default_tie_policy: "reject-ambiguous", diagnostic_tie_policy: "lowest-index"};
if (!same(freeze.community1.pipeline_policy, expectedPolicy)) fail("Community pipeline policy drift");
if (freeze.quality_gates.community_reference_delta.der_percentage_points_max !== reference.proposed_targets_not_qualified.max_der_regression_percentage_points || freeze.quality_gates.whisper_reference_delta.wer_percentage_points_max !== reference.proposed_targets_not_qualified.max_wer_regression_percentage_points || reference.proposed_targets_not_qualified.quality_budgets_require_ratification !== true) fail("provisional quality budget drift");
for (const path of freeze.source_documents) await readFile(resolve(root, path));
const canonical = JSON.stringify(freeze);
console.log(JSON.stringify({schema: freeze.schema, whisper_models: freeze.whisper.models.length, segmentation_tensors: freeze.community1.segmentation.tensor_count, embedding_tensors: freeze.community1.embedding.tensor_count, source_documents: freeze.source_documents.length, freeze_sha256: createHash("sha256").update(canonical).digest("hex"), status: "ok"}));
