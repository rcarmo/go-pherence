package qwen

import (
	"fmt"
	"strings"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/tensor"
)

func expectQwen35DenseOrQuantShape(t *tensor.Tensor, q *Qwen35NVFP4Weight, m *mlx.QuantWeight, want []int, name string) error {
	if m != nil {
		if len(want) != 2 {
			return fmt.Errorf("%s MLX expected 2D shape, want %v", name, want)
		}
		if !((m.OutDim == want[0] && m.InDim == want[1]) || (m.OutDim == want[1] && m.InDim == want[0])) {
			return fmt.Errorf("%s MLX shape out/in=%d/%d want %v", name, m.OutDim, m.InDim, want)
		}
		return nil
	}
	return expectQwen35DenseOrNVFP4Shape(t, q, want, name)
}

func expectQwen35DenseOrNVFP4Shape(t *tensor.Tensor, q *Qwen35NVFP4Weight, want []int, name string) error {
	if q != nil {
		if len(want) != 2 {
			return fmt.Errorf("%s NVFP4 expected 2D shape, want %v", name, want)
		}
		if q.W == nil {
			return fmt.Errorf("%s has nil NVFP4 weight", name)
		}
		if !((q.W.OutDim == want[1] && q.W.InDim == want[0]) || (q.W.OutDim == want[0] && q.W.InDim == want[1])) {
			return fmt.Errorf("%s NVFP4 shape out/in=%d/%d want %v", name, q.W.OutDim, q.W.InDim, want)
		}
		return nil
	}
	return expectShape(t, want, name)
}

func qwen35FullMLXForName(l *Qwen35FullAttentionLayer, name string) *mlx.QuantWeight {
	switch {
	case strings.Contains(name, ".self_attn.q_proj.weight"):
		return l.QWm
	case strings.Contains(name, ".self_attn.k_proj.weight"):
		return l.KWm
	case strings.Contains(name, ".self_attn.v_proj.weight"):
		return l.VWm
	case strings.Contains(name, ".self_attn.o_proj.weight"):
		return l.OWm
	case strings.Contains(name, ".mlp.gate_proj.weight"):
		return l.GateWm
	case strings.Contains(name, ".mlp.up_proj.weight"):
		return l.UpWm
	case strings.Contains(name, ".mlp.down_proj.weight"):
		return l.DownWm
	default:
		return nil
	}
}

func qwen35FullQuantForName(l *Qwen35FullAttentionLayer, name string) *Qwen35NVFP4Weight {
	switch {
	case strings.Contains(name, ".self_attn.q_proj.weight"):
		return l.QWQ
	case strings.Contains(name, ".self_attn.k_proj.weight"):
		return l.KWQ
	case strings.Contains(name, ".self_attn.v_proj.weight"):
		return l.VWQ
	case strings.Contains(name, ".self_attn.o_proj.weight"):
		return l.OWQ
	case strings.Contains(name, ".mlp.gate_proj.weight"):
		return l.GateWQ
	case strings.Contains(name, ".mlp.up_proj.weight"):
		return l.UpWQ
	case strings.Contains(name, ".mlp.down_proj.weight"):
		return l.DownWQ
	default:
		return nil
	}
}

func qwen35MLXForName(l *Qwen35LinearAttentionLayer, name string) *mlx.QuantWeight {
	switch {
	case strings.Contains(name, ".linear_attn.in_proj_qkv.weight") || strings.Contains(name, ".linear_attn.in_proj_qkvz.weight"):
		return l.QKVWm
	case strings.Contains(name, ".linear_attn.in_proj_z.weight") || strings.Contains(name, ".linear_attn.in_proj_gate.weight"):
		return l.GateWm
	case strings.Contains(name, ".linear_attn.in_proj_b.weight") || strings.Contains(name, ".linear_attn.in_proj_ba.weight"):
		return l.BetaWm
	case strings.Contains(name, ".linear_attn.in_proj_a.weight"):
		return l.AlphaWm
	case strings.Contains(name, ".linear_attn.out_proj.weight"):
		return l.OutWm
	case strings.Contains(name, ".mlp.gate_proj.weight"):
		return l.MLPGateWm
	case strings.Contains(name, ".mlp.up_proj.weight"):
		return l.MLPUpWm
	case strings.Contains(name, ".mlp.down_proj.weight"):
		return l.MLPDownWm
	default:
		return nil
	}
}

func qwen35QuantForName(l *Qwen35LinearAttentionLayer, name string) *Qwen35NVFP4Weight {
	switch {
	case strings.Contains(name, ".linear_attn.in_proj_qkv.weight") || strings.Contains(name, ".linear_attn.in_proj_qkvz.weight"):
		return l.QKVWQ
	case strings.Contains(name, ".linear_attn.in_proj_z.weight") || strings.Contains(name, ".linear_attn.in_proj_gate.weight"):
		return l.GateWQ
	case strings.Contains(name, ".linear_attn.in_proj_b.weight") || strings.Contains(name, ".linear_attn.in_proj_ba.weight"):
		return l.BetaWQ
	case strings.Contains(name, ".linear_attn.in_proj_a.weight"):
		return l.AlphaWQ
	case strings.Contains(name, ".linear_attn.out_proj.weight"):
		return l.OutWQ
	case strings.Contains(name, ".mlp.gate_proj.weight"):
		return l.MLPGateWQ
	case strings.Contains(name, ".mlp.up_proj.weight"):
		return l.MLPUpWQ
	case strings.Contains(name, ".mlp.down_proj.weight"):
		return l.MLPDownWQ
	default:
		return nil
	}
}
