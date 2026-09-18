// Package gliner2 parses the published GLiNER2.5 boundary checkpoint metadata.
//
// This first slice is deliberately limited to config.json contract validation
// and raw-field preservation for introspection. It does not implement encoder
// loading, tensor loading, or runtime inference.
package gliner2

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const (
	UpstreamModelType          = "extractor"
	SupportedArchitecture      = "boundary"
	SupportedArchitectureClass = "BoundaryExtractor"
	ConfigVersion              = 3
	ArchitectureVersion        = 1
)

var requiredTopLevelFields = []string{
	"architecture",
	"architecture_version",
	"architectures",
	"attn_implementation",
	"boundary_head",
	"config_version",
	"max_len",
	"model_name",
	"model_type",
	"token_pooling",
	"transformers_version",
}

var requiredBoundaryFields = []string{
	"abstention_loss_weight",
	"abstention_threshold",
	"adaptive_threshold",
	"bidirectional_proposals",
	"boundary_attention_heads",
	"boundary_attention_layers",
	"boundary_attention_window",
	"boundary_dim",
	"boundary_ffn_multiplier",
	"boundary_focal_clip",
	"boundary_focal_gamma_negative",
	"boundary_focal_gamma_positive",
	"boundary_marginal_loss",
	"boundary_negative_weight",
	"boundary_refinement_layers",
	"boundary_top_k_alpha",
	"boundary_top_k_bucket",
	"boundary_top_k_max",
	"candidate_attention_heads",
	"candidate_attention_layers",
	"candidate_budget",
	"candidate_pool",
	"classification_loss_weight",
	"classification_temperature",
	"consistency_loss_weight",
	"consistency_warmup_steps",
	"content_dim",
	"content_soft_max_pool",
	"count_loss_weight",
	"directional_relation_states",
	"dropout",
	"enable_abstention",
	"enable_count_head",
	"enable_records",
	"enable_relations",
	"enable_rotary_endpoints",
	"enable_span_content",
	"end_block_size",
	"end_top_k",
	"endpoint_difference_features",
	"ends_per_start",
	"export_mode",
	"hard_negative_keep_all_when_absent",
	"hard_negatives_per_positive",
	"loss_reduction",
	"max_gold_per_query",
	"max_negative_queries_per_batch",
	"min_pool_per_query",
	"minimum_hard_negatives",
	"multihead_pair_compat_heads",
	"negative_query_ratio",
	"overlap_policy",
	"pair_dim",
	"pair_temperature",
	"pool_boundary_top_k",
	"pool_size",
	"proposal_loss_weight",
	"query_attention_layers",
	"query_conditioned_inside_weight",
	"record_anchor_proposal_threshold",
	"record_anchor_threshold",
	"record_dim",
	"record_field_threshold",
	"record_instance_queries",
	"record_loss_weight",
	"record_temperature",
	"relation_argument_proposal_threshold",
	"relation_biaffine_content",
	"relation_heads_per_type",
	"relation_loss_weight",
	"relation_pair_cap",
	"relation_tails_per_type",
	"relation_temperature",
	"rerank_listwise_weight",
	"reranker_endpoint_compat",
	"rotary_base",
	"soft_iou_anneal_steps",
	"soft_iou_aux_weight",
	"start_top_k",
	"starts_per_end",
	"training_candidate_budget",
	"use_inside_evidence",
	"vectorized_pair_elements",
}

type Config struct {
	Architecture            string                     `json:"architecture"`
	ArchitectureVersion     int                        `json:"architecture_version"`
	Architectures           []string                   `json:"architectures"`
	AttentionImplementation string                     `json:"attn_implementation"`
	BoundaryHead            BoundaryHeadConfig         `json:"boundary_head"`
	ConfigVersion           int                        `json:"config_version"`
	EncoderConfig           *EncoderConfig             `json:"encoder_config,omitempty"`
	MaxLen                  *int                       `json:"max_len,omitempty"`
	ModelName               string                     `json:"model_name"`
	ModelType               string                     `json:"model_type"`
	TokenPooling            string                     `json:"token_pooling"`
	TransformersVersion     string                     `json:"transformers_version"`
	Raw                     map[string]json.RawMessage `json:"-"`
}

type BoundaryHeadConfig struct {
	AbstentionLossWeight              float64                    `json:"abstention_loss_weight"`
	AbstentionThreshold               float64                    `json:"abstention_threshold"`
	AdaptiveThreshold                 bool                       `json:"adaptive_threshold"`
	BidirectionalProposals            bool                       `json:"bidirectional_proposals"`
	BoundaryAttentionHeads            int                        `json:"boundary_attention_heads"`
	BoundaryAttentionLayers           int                        `json:"boundary_attention_layers"`
	BoundaryAttentionWindow           int                        `json:"boundary_attention_window"`
	BoundaryDim                       int                        `json:"boundary_dim"`
	BoundaryFFNMultiplier             float64                    `json:"boundary_ffn_multiplier"`
	BoundaryFocalClip                 float64                    `json:"boundary_focal_clip"`
	BoundaryFocalGammaNegative        float64                    `json:"boundary_focal_gamma_negative"`
	BoundaryFocalGammaPositive        float64                    `json:"boundary_focal_gamma_positive"`
	BoundaryMarginalLoss              string                     `json:"boundary_marginal_loss"`
	BoundaryNegativeWeight            float64                    `json:"boundary_negative_weight"`
	BoundaryRefinementLayers          int                        `json:"boundary_refinement_layers"`
	BoundaryTopKAlpha                 float64                    `json:"boundary_top_k_alpha"`
	BoundaryTopKBucket                int                        `json:"boundary_top_k_bucket"`
	BoundaryTopKMax                   int                        `json:"boundary_top_k_max"`
	CandidateAttentionHeads           int                        `json:"candidate_attention_heads"`
	CandidateAttentionLayers          int                        `json:"candidate_attention_layers"`
	CandidateBudget                   int                        `json:"candidate_budget"`
	CandidatePool                     string                     `json:"candidate_pool"`
	ClassificationLossWeight          float64                    `json:"classification_loss_weight"`
	ClassificationTemperature         float64                    `json:"classification_temperature"`
	ConsistencyLossWeight             float64                    `json:"consistency_loss_weight"`
	ConsistencyWarmupSteps            int                        `json:"consistency_warmup_steps"`
	ContentDim                        int                        `json:"content_dim"`
	ContentSoftMaxPool                bool                       `json:"content_soft_max_pool"`
	CountLossWeight                   float64                    `json:"count_loss_weight"`
	DirectionalRelationStates         bool                       `json:"directional_relation_states"`
	Dropout                           float64                    `json:"dropout"`
	EnableAbstention                  bool                       `json:"enable_abstention"`
	EnableCountHead                   bool                       `json:"enable_count_head"`
	EnableRecords                     bool                       `json:"enable_records"`
	EnableRelations                   bool                       `json:"enable_relations"`
	EnableRotaryEndpoints             bool                       `json:"enable_rotary_endpoints"`
	EnableSpanContent                 bool                       `json:"enable_span_content"`
	EndBlockSize                      int                        `json:"end_block_size"`
	EndTopK                           int                        `json:"end_top_k"`
	EndpointDifferenceFeatures        bool                       `json:"endpoint_difference_features"`
	EndsPerStart                      int                        `json:"ends_per_start"`
	ExportMode                        string                     `json:"export_mode"`
	HardNegativeKeepAllWhenAbsent     bool                       `json:"hard_negative_keep_all_when_absent"`
	HardNegativesPerPositive          int                        `json:"hard_negatives_per_positive"`
	LossReduction                     string                     `json:"loss_reduction"`
	MaxGoldPerQuery                   int                        `json:"max_gold_per_query"`
	MaxNegativeQueriesPerBatch        int                        `json:"max_negative_queries_per_batch"`
	MinPoolPerQuery                   int                        `json:"min_pool_per_query"`
	MinimumHardNegatives              int                        `json:"minimum_hard_negatives"`
	MultiheadPairCompatHeads          int                        `json:"multihead_pair_compat_heads"`
	NegativeQueryRatio                float64                    `json:"negative_query_ratio"`
	OverlapPolicy                     string                     `json:"overlap_policy"`
	PairDim                           int                        `json:"pair_dim"`
	PairTemperature                   float64                    `json:"pair_temperature"`
	PoolBoundaryTopK                  int                        `json:"pool_boundary_top_k"`
	PoolSize                          int                        `json:"pool_size"`
	ProposalLossWeight                float64                    `json:"proposal_loss_weight"`
	QueryAttentionLayers              int                        `json:"query_attention_layers"`
	QueryConditionedInsideWeight      bool                       `json:"query_conditioned_inside_weight"`
	RecordAnchorProposalThreshold     float64                    `json:"record_anchor_proposal_threshold"`
	RecordAnchorThreshold             float64                    `json:"record_anchor_threshold"`
	RecordDim                         int                        `json:"record_dim"`
	RecordFieldThreshold              float64                    `json:"record_field_threshold"`
	RecordInstanceQueries             int                        `json:"record_instance_queries"`
	RecordLossWeight                  float64                    `json:"record_loss_weight"`
	RecordTemperature                 float64                    `json:"record_temperature"`
	RelationArgumentProposalThreshold float64                    `json:"relation_argument_proposal_threshold"`
	RelationBiaffineContent           bool                       `json:"relation_biaffine_content"`
	RelationHeadsPerType              int                        `json:"relation_heads_per_type"`
	RelationLossWeight                float64                    `json:"relation_loss_weight"`
	RelationPairCap                   int                        `json:"relation_pair_cap"`
	RelationTailsPerType              int                        `json:"relation_tails_per_type"`
	RelationTemperature               float64                    `json:"relation_temperature"`
	RerankListwiseWeight              float64                    `json:"rerank_listwise_weight"`
	RerankerEndpointCompat            bool                       `json:"reranker_endpoint_compat"`
	RotaryBase                        float64                    `json:"rotary_base"`
	SoftIOUAnnealSteps                int                        `json:"soft_iou_anneal_steps"`
	SoftIOUAuxWeight                  float64                    `json:"soft_iou_aux_weight"`
	StartTopK                         int                        `json:"start_top_k"`
	StartsPerEnd                      int                        `json:"starts_per_end"`
	TrainingCandidateBudget           int                        `json:"training_candidate_budget"`
	UseInsideEvidence                 bool                       `json:"use_inside_evidence"`
	VectorizedPairElements            int                        `json:"vectorized_pair_elements"`
	Raw                               map[string]json.RawMessage `json:"-"`
}

type EncoderConfig struct {
	Architectures         []string                   `json:"architectures,omitempty"`
	AttentionDropout      float64                    `json:"attention_probs_dropout_prob,omitempty"`
	HeadDim               int                        `json:"head_dim,omitempty"`
	HiddenAct             string                     `json:"hidden_act,omitempty"`
	HiddenDropout         float64                    `json:"hidden_dropout_prob,omitempty"`
	HiddenSize            int                        `json:"hidden_size"`
	IntermediateSize      int                        `json:"intermediate_size"`
	LayerNormEps          float64                    `json:"layer_norm_eps,omitempty"`
	MaxPositionEmbeddings int                        `json:"max_position_embeddings"`
	MaxRelativePositions  int                        `json:"max_relative_positions,omitempty"`
	ModelType             string                     `json:"model_type"`
	NumAttentionHeads     int                        `json:"num_attention_heads"`
	NumHiddenLayers       int                        `json:"num_hidden_layers"`
	NumKeyValueHeads      int                        `json:"num_key_value_heads,omitempty"`
	PadTokenID            int                        `json:"pad_token_id,omitempty"`
	PositionBuckets       int                        `json:"position_buckets,omitempty"`
	RelativeAttention     bool                       `json:"relative_attention,omitempty"`
	TypeVocabSize         int                        `json:"type_vocab_size,omitempty"`
	VocabSize             int                        `json:"vocab_size"`
	Raw                   map[string]json.RawMessage `json:"-"`
}

type rawConfig struct {
	Architecture            string          `json:"architecture"`
	ArchitectureVersion     int             `json:"architecture_version"`
	Architectures           []string        `json:"architectures"`
	AttentionImplementation string          `json:"attn_implementation"`
	BoundaryHead            json.RawMessage `json:"boundary_head"`
	ConfigVersion           int             `json:"config_version"`
	EncoderConfig           json.RawMessage `json:"encoder_config,omitempty"`
	MaxLen                  *int            `json:"max_len"`
	ModelName               string          `json:"model_name"`
	ModelType               string          `json:"model_type"`
	TokenPooling            string          `json:"token_pooling"`
	TransformersVersion     string          `json:"transformers_version"`
}

func LoadConfig(r io.Reader) (Config, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Config{}, fmt.Errorf("read GLiNER2 config: %w", err)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return Config{}, fmt.Errorf("read GLiNER2 config: empty input")
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return Config{}, fmt.Errorf("decode GLiNER2 config: %w", err)
	}

	var decoded rawConfig
	if err := json.Unmarshal(data, &decoded); err != nil {
		return Config{}, fmt.Errorf("decode GLiNER2 config: %w", err)
	}

	cfg := Config{
		Architecture:            normalizeLower(decoded.Architecture),
		ArchitectureVersion:     decoded.ArchitectureVersion,
		Architectures:           append([]string(nil), decoded.Architectures...),
		AttentionImplementation: normalizeLower(decoded.AttentionImplementation),
		ConfigVersion:           decoded.ConfigVersion,
		MaxLen:                  decoded.MaxLen,
		ModelName:               strings.TrimSpace(decoded.ModelName),
		ModelType:               normalizeLower(decoded.ModelType),
		TokenPooling:            normalizeLower(decoded.TokenPooling),
		TransformersVersion:     strings.TrimSpace(decoded.TransformersVersion),
		Raw:                     cloneRawMap(raw),
	}

	if err := validateArchitecture(cfg.Architecture); err != nil {
		return Config{}, err
	}
	if err := requireFields(raw, "config", requiredTopLevelFields); err != nil {
		return Config{}, err
	}

	head, err := parseBoundaryHead(decoded.BoundaryHead)
	if err != nil {
		return Config{}, err
	}
	cfg.BoundaryHead = head

	if len(bytes.TrimSpace(decoded.EncoderConfig)) > 0 && !isJSONNull(decoded.EncoderConfig) {
		enc, err := parseEncoderConfig(decoded.EncoderConfig)
		if err != nil {
			return Config{}, err
		}
		cfg.EncoderConfig = &enc
	}

	if err := cfg.Validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) Validate() error {
	if c.ModelType != UpstreamModelType {
		return fmt.Errorf("unsupported GLiNER2 model_type %q", c.ModelType)
	}
	if err := validateArchitecture(c.Architecture); err != nil {
		return err
	}
	if c.ArchitectureVersion != ArchitectureVersion {
		return fmt.Errorf("unsupported GLiNER2 architecture_version %d", c.ArchitectureVersion)
	}
	if c.ConfigVersion != ConfigVersion {
		return fmt.Errorf("unsupported GLiNER2 config_version %d", c.ConfigVersion)
	}
	if c.ModelName == "" {
		return fmt.Errorf("missing GLiNER2 model_name")
	}
	if c.TransformersVersion == "" {
		return fmt.Errorf("missing GLiNER2 transformers_version")
	}
	if !containsOneOf(c.TokenPooling, "first", "mean", "max") {
		return fmt.Errorf("unsupported GLiNER2 token_pooling %q", c.TokenPooling)
	}
	if !containsOneOf(c.AttentionImplementation, "sdpa", "flash_attention_2", "eager") {
		return fmt.Errorf("unsupported GLiNER2 attn_implementation %q", c.AttentionImplementation)
	}
	if c.MaxLen == nil || *c.MaxLen <= 0 {
		return fmt.Errorf("invalid GLiNER2 max_len %v", valueOrNil(c.MaxLen))
	}
	if len(c.Architectures) != 1 {
		return fmt.Errorf("unsupported GLiNER2 architectures %v", c.Architectures)
	}
	if c.Architectures[0] != SupportedArchitectureClass && c.Architectures[0] != "BoundaryExtractorModel" {
		return fmt.Errorf("unsupported GLiNER2 architectures %v", c.Architectures)
	}
	if err := c.BoundaryHead.Validate(); err != nil {
		return err
	}
	if c.EncoderConfig != nil {
		if err := c.EncoderConfig.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (h BoundaryHeadConfig) Validate() error {
	if !containsOneOf(h.ExportMode, "auto", "streaming", "vectorized") {
		return fmt.Errorf("boundary_head.export_mode must be 'auto', 'streaming', or 'vectorized', got %q", h.ExportMode)
	}
	if h.VectorizedPairElements <= 0 {
		return fmt.Errorf("boundary_head.vectorized_pair_elements must be > 0")
	}
	if !(0 < h.BoundaryNegativeWeight && h.BoundaryNegativeWeight <= 1) {
		return fmt.Errorf("boundary_head.boundary_negative_weight must be in (0, 1], got %v", h.BoundaryNegativeWeight)
	}
	if !containsOneOf(h.BoundaryMarginalLoss, "bce", "asymmetric_focal") {
		return fmt.Errorf("boundary_head.boundary_marginal_loss must be 'bce' or 'asymmetric_focal', got %q", h.BoundaryMarginalLoss)
	}
	if !containsOneOf(h.LossReduction, "global", "per_query", "sum") {
		return fmt.Errorf("boundary_head.loss_reduction must be 'global', 'per_query', or 'sum', got %q", h.LossReduction)
	}
	if h.BoundaryFocalGammaPositive < 0 || h.BoundaryFocalGammaNegative < 0 {
		return fmt.Errorf("boundary_head focal gamma values must be >= 0")
	}
	if h.BoundaryFocalClip < 0 || h.BoundaryFocalClip >= 1 {
		return fmt.Errorf("boundary_head.boundary_focal_clip must be in [0, 1)")
	}
	if h.HardNegativesPerPositive < 0 || h.MinimumHardNegatives < 0 {
		return fmt.Errorf("boundary_head hard-negative counts must be >= 0")
	}
	if h.ContentDim <= 0 {
		return fmt.Errorf("boundary_head.content_dim must be > 0")
	}
	if h.RotaryBase <= 0 {
		return fmt.Errorf("boundary_head.rotary_base must be > 0")
	}
	if h.EnableRotaryEndpoints && (h.BoundaryDim%2 != 0 || h.PairDim%2 != 0) {
		return fmt.Errorf("boundary_head.enable_rotary_endpoints requires even boundary_dim and pair_dim, got %d and %d", h.BoundaryDim, h.PairDim)
	}
	if h.BoundaryAttentionLayers < 0 {
		return fmt.Errorf("boundary_head.boundary_attention_layers must be >= 0")
	}
	if h.BoundaryAttentionHeads <= 0 {
		return fmt.Errorf("boundary_head.boundary_attention_heads must be > 0")
	}
	if h.BoundaryAttentionLayers > 0 && h.BoundaryDim%h.BoundaryAttentionHeads != 0 {
		return fmt.Errorf("boundary_head.boundary_dim must be divisible by boundary_attention_heads when attention is enabled")
	}
	if h.BoundaryAttentionWindow < 0 {
		return fmt.Errorf("boundary_head.boundary_attention_window must be >= 0")
	}
	if h.MultiheadPairCompatHeads <= 0 {
		return fmt.Errorf("boundary_head.multihead_pair_compat_heads must be > 0")
	}
	if h.PairDim%h.MultiheadPairCompatHeads != 0 {
		return fmt.Errorf("boundary_head.pair_dim must be divisible by multihead_pair_compat_heads, got %d and %d", h.PairDim, h.MultiheadPairCompatHeads)
	}
	if h.BoundaryTopKAlpha < 0 {
		return fmt.Errorf("boundary_head.boundary_top_k_alpha must be >= 0")
	}
	if h.BoundaryTopKMax < max(h.StartTopK, h.EndTopK) {
		return fmt.Errorf("boundary_head.boundary_top_k_max must be >= start_top_k and end_top_k")
	}
	if h.BoundaryTopKBucket <= 0 {
		return fmt.Errorf("boundary_head.boundary_top_k_bucket must be > 0")
	}
	if !containsOneOf(h.CandidatePool, "per_query", "shared") {
		return fmt.Errorf("boundary_head.candidate_pool must be 'per_query' or 'shared', got %q", h.CandidatePool)
	}
	if h.PoolBoundaryTopK <= 0 {
		return fmt.Errorf("boundary_head.pool_boundary_top_k must be > 0")
	}
	if h.PoolSize <= 0 {
		return fmt.Errorf("boundary_head.pool_size must be > 0")
	}
	if h.MinPoolPerQuery < 0 {
		return fmt.Errorf("boundary_head.min_pool_per_query must be >= 0")
	}
	if h.MinPoolPerQuery > h.PoolSize {
		return fmt.Errorf("boundary_head.min_pool_per_query must not exceed pool_size")
	}
	if h.CandidateAttentionLayers < 0 {
		return fmt.Errorf("boundary_head.candidate_attention_layers must be >= 0")
	}
	if h.CandidateAttentionHeads <= 0 {
		return fmt.Errorf("boundary_head.candidate_attention_heads must be > 0")
	}
	if (h.CandidateAttentionLayers > 0 || h.QueryAttentionLayers > 0) && h.PairDim%h.CandidateAttentionHeads != 0 {
		return fmt.Errorf("boundary_head.pair_dim must be divisible by candidate_attention_heads when candidate or query attention is enabled")
	}
	if h.QueryAttentionLayers < 0 {
		return fmt.Errorf("boundary_head.query_attention_layers must be >= 0")
	}
	if h.AbstentionThreshold < 0 || h.AbstentionThreshold > 1 {
		return fmt.Errorf("boundary_head.abstention_threshold must be in [0, 1]")
	}
	for key, value := range map[string]float64{
		"proposal_loss_weight":       h.ProposalLossWeight,
		"consistency_loss_weight":    h.ConsistencyLossWeight,
		"rerank_listwise_weight":     h.RerankListwiseWeight,
		"soft_iou_aux_weight":        h.SoftIOUAuxWeight,
		"abstention_loss_weight":     h.AbstentionLossWeight,
		"count_loss_weight":          h.CountLossWeight,
		"classification_loss_weight": h.ClassificationLossWeight,
		"record_loss_weight":         h.RecordLossWeight,
		"relation_loss_weight":       h.RelationLossWeight,
	} {
		if value < 0 {
			return fmt.Errorf("boundary_head.%s must be >= 0", key)
		}
	}
	if h.ConsistencyWarmupSteps < 0 {
		return fmt.Errorf("boundary_head.consistency_warmup_steps must be >= 0")
	}
	if h.SoftIOUAnnealSteps < 0 {
		return fmt.Errorf("boundary_head.soft_iou_anneal_steps must be >= 0")
	}
	if !containsOneOf(h.OverlapPolicy, "flat", "nested", "longest") {
		return fmt.Errorf("boundary_head.overlap_policy must be 'flat', 'nested', or 'longest'")
	}
	for key, value := range map[string]float64{
		"pair_temperature":           h.PairTemperature,
		"relation_temperature":       h.RelationTemperature,
		"record_temperature":         h.RecordTemperature,
		"classification_temperature": h.ClassificationTemperature,
	} {
		if value <= 0 {
			return fmt.Errorf("boundary_head.%s must be > 0", key)
		}
	}
	if h.NegativeQueryRatio < 0 {
		return fmt.Errorf("boundary_head.negative_query_ratio must be >= 0")
	}
	if h.MaxNegativeQueriesPerBatch <= 0 {
		return fmt.Errorf("boundary_head.max_negative_queries_per_batch must be > 0")
	}
	if h.ExportMode == "vectorized" && h.BoundaryTopKAlpha > 0 {
		return fmt.Errorf("boundary_head.export_mode='vectorized' is incompatible with an adaptive boundary budget; use 'auto'/'streaming' or set boundary_top_k_alpha=0")
	}
	if h.RecordDim <= 0 {
		return fmt.Errorf("boundary_head.record_dim must be > 0, got %d", h.RecordDim)
	}
	if h.RecordInstanceQueries <= 0 {
		return fmt.Errorf("boundary_head.record_instance_queries must be > 0, got %d", h.RecordInstanceQueries)
	}
	for key, value := range map[string]float64{
		"record_anchor_proposal_threshold":     h.RecordAnchorProposalThreshold,
		"record_anchor_threshold":              h.RecordAnchorThreshold,
		"record_field_threshold":               h.RecordFieldThreshold,
		"relation_argument_proposal_threshold": h.RelationArgumentProposalThreshold,
	} {
		if value < 0 || value > 1 {
			return fmt.Errorf("boundary_head.%s must be in [0, 1]", key)
		}
	}
	if h.RecordAnchorProposalThreshold > h.RecordAnchorThreshold {
		return fmt.Errorf("boundary_head.record_anchor_proposal_threshold (%v) must be <= record_anchor_threshold (%v)", h.RecordAnchorProposalThreshold, h.RecordAnchorThreshold)
	}
	for key, value := range map[string]int{
		"relation_heads_per_type": h.RelationHeadsPerType,
		"relation_tails_per_type": h.RelationTailsPerType,
		"relation_pair_cap":       h.RelationPairCap,
		"boundary_dim":            h.BoundaryDim,
		"pair_dim":                h.PairDim,
		"start_top_k":             h.StartTopK,
		"end_top_k":               h.EndTopK,
		"ends_per_start":          h.EndsPerStart,
		"starts_per_end":          h.StartsPerEnd,
		"candidate_budget":        h.CandidateBudget,
		"max_gold_per_query":      h.MaxGoldPerQuery,
		"end_block_size":          h.EndBlockSize,
	} {
		if value <= 0 {
			return fmt.Errorf("boundary_head.%s must be > 0, got %d", key, value)
		}
	}
	if h.BoundaryRefinementLayers < 0 {
		return fmt.Errorf("boundary_head.boundary_refinement_layers must be >= 0, got %d", h.BoundaryRefinementLayers)
	}
	if h.BoundaryFFNMultiplier <= 0 {
		return fmt.Errorf("boundary_head.boundary_ffn_multiplier must be > 0, got %v", h.BoundaryFFNMultiplier)
	}
	if h.TrainingCandidateBudget < h.CandidateBudget {
		return fmt.Errorf("boundary_head.training_candidate_budget (%d) must be >= candidate_budget (%d)", h.TrainingCandidateBudget, h.CandidateBudget)
	}
	if h.TrainingCandidateBudget < h.MaxGoldPerQuery {
		return fmt.Errorf("boundary_head.training_candidate_budget (%d) must be >= max_gold_per_query (%d)", h.TrainingCandidateBudget, h.MaxGoldPerQuery)
	}
	if h.Dropout < 0 || h.Dropout >= 1 {
		return fmt.Errorf("boundary_head.dropout must be in [0, 1), got %v", h.Dropout)
	}
	return nil
}

func (e EncoderConfig) Validate() error {
	if strings.TrimSpace(e.ModelType) == "" {
		return fmt.Errorf("embedded encoder_config.model_type is required")
	}
	if e.HiddenSize <= 0 || e.IntermediateSize <= 0 || e.NumHiddenLayers <= 0 || e.NumAttentionHeads <= 0 || e.MaxPositionEmbeddings <= 0 || e.VocabSize <= 0 {
		return fmt.Errorf("invalid encoder_config dimensions: hidden=%d intermediate=%d layers=%d heads=%d max_positions=%d vocab=%d", e.HiddenSize, e.IntermediateSize, e.NumHiddenLayers, e.NumAttentionHeads, e.MaxPositionEmbeddings, e.VocabSize)
	}
	headDim := e.HeadDim
	if headDim == 0 {
		if e.HiddenSize%e.NumAttentionHeads != 0 {
			return fmt.Errorf("invalid encoder_config head dims: hidden=%d heads=%d head_dim=%d", e.HiddenSize, e.NumAttentionHeads, e.HeadDim)
		}
		headDim = e.HiddenSize / e.NumAttentionHeads
	}
	if e.HiddenSize != e.NumAttentionHeads*headDim {
		return fmt.Errorf("invalid encoder_config head dims: hidden=%d heads=%d head_dim=%d", e.HiddenSize, e.NumAttentionHeads, headDim)
	}
	if e.NumKeyValueHeads > 0 {
		if e.NumKeyValueHeads > e.NumAttentionHeads || e.NumAttentionHeads%e.NumKeyValueHeads != 0 {
			return fmt.Errorf("invalid encoder_config GQA dims: heads=%d kv_heads=%d", e.NumAttentionHeads, e.NumKeyValueHeads)
		}
	}
	if e.PadTokenID < 0 {
		return fmt.Errorf("invalid encoder_config pad_token_id %d", e.PadTokenID)
	}
	return nil
}

func parseBoundaryHead(data json.RawMessage) (BoundaryHeadConfig, error) {
	var head BoundaryHeadConfig
	raw, err := decodeObject(data, "boundary_head", &head)
	if err != nil {
		return BoundaryHeadConfig{}, err
	}
	if err := requireFields(raw, "boundary_head", requiredBoundaryFields); err != nil {
		return BoundaryHeadConfig{}, err
	}
	head.BoundaryMarginalLoss = normalizeLower(head.BoundaryMarginalLoss)
	head.CandidatePool = normalizeLower(head.CandidatePool)
	head.ExportMode = normalizeLower(head.ExportMode)
	head.LossReduction = normalizeLower(head.LossReduction)
	head.OverlapPolicy = normalizeLower(head.OverlapPolicy)
	head.Raw = raw
	return head, nil
}

func parseEncoderConfig(data json.RawMessage) (EncoderConfig, error) {
	var enc EncoderConfig
	raw, err := decodeObject(data, "encoder_config", &enc)
	if err != nil {
		return EncoderConfig{}, err
	}
	enc.ModelType = strings.TrimSpace(enc.ModelType)
	enc.HiddenAct = strings.TrimSpace(enc.HiddenAct)
	enc.Raw = raw
	if err := enc.Validate(); err != nil {
		return EncoderConfig{}, err
	}
	return enc, nil
}

func decodeObject(data json.RawMessage, scope string, v any) (map[string]json.RawMessage, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || isJSONNull(trimmed) {
		return nil, fmt.Errorf("%s must be a JSON object", scope)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &raw); err != nil {
		return nil, fmt.Errorf("decode %s: %w", scope, err)
	}
	if raw == nil {
		return nil, fmt.Errorf("%s must be a JSON object", scope)
	}
	if err := json.Unmarshal(trimmed, v); err != nil {
		return nil, fmt.Errorf("decode %s: %w", scope, err)
	}
	return cloneRawMap(raw), nil
}

func requireFields(raw map[string]json.RawMessage, scope string, fields []string) error {
	for _, field := range fields {
		value, ok := raw[field]
		if !ok || len(bytes.TrimSpace(value)) == 0 || isJSONNull(value) {
			return fmt.Errorf("missing published %s.%s", scope, field)
		}
	}
	return nil
}

func validateArchitecture(architecture string) error {
	if architecture != SupportedArchitecture {
		return fmt.Errorf("unsupported GLiNER2 architecture %q; only %q checkpoints are supported", architecture, SupportedArchitecture)
	}
	return nil
}

func normalizeLower(s string) string {
	return strings.ToLower(strings.TrimSpace(s))
}

func containsOneOf(value string, allowed ...string) bool {
	for _, item := range allowed {
		if value == item {
			return true
		}
	}
	return false
}

func isJSONNull(data []byte) bool {
	return bytes.Equal(bytes.TrimSpace(data), []byte("null"))
}

func cloneRawMap(in map[string]json.RawMessage) map[string]json.RawMessage {
	out := make(map[string]json.RawMessage, len(in))
	for k, v := range in {
		out[k] = append(json.RawMessage(nil), v...)
	}
	return out
}

func valueOrNil(v *int) any {
	if v == nil {
		return nil
	}
	return *v
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
