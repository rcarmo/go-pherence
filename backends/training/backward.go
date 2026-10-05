package training

import (
	"fmt"
	"math"
)

// LayerNormShape describes a cached pre-affine normalized base [M,N], inverse
// standard deviation [M], affine weight [N], upstream dY[M,N]. Modes overwrite
// dX (0), dWeight (1), dBias (2). An epsilon/variance is not recomputed here.
func LayerNormShape(rows, width int, mode uint32) error {
	if rows < 1 || rows > 1024 || width < 1 || width > 16384 || mode > 2 {
		return fmt.Errorf("invalid training layernorm shape/mode")
	}
	return nil
}

// DualNormShape consumes X/XT/Gvalue/Gtangent [M,N], weight[N], and forward
// stats[M,5]=(mean,dmean,inv,dinv,dvariance). kind0=population affine LayerNorm;
// kind1=uncentred x*alpha*inv with centred sample variance (N-1 denominator).
// mode0/1 overwrite input primal/tangent adjoints,2=dWeight,3=dBias(kind0 only).
func DualNormShape(rows, width int, kind, mode uint32) error {
	if rows < 1 || rows > 1024 || width < 1 || width > 16384 || kind > 1 || mode > 3 || (kind == 1 && (width < 2 || mode == 3)) {
		return fmt.Errorf("invalid dual normalization shape/kind/mode")
	}
	return nil
}

// DualModShape: kind0=shift/scale AdaLN (B planar[2,N]), kind1=product gate.
// Modes0/1 write primal/tangent A adjoints; modes2/3 write B adjoints. Shared
// conditioning across rows requires explicit accumulation outside this kernel.
func DualModShape(n int, kind, mode uint32) error {
	if n < 1 || n > 1<<26 || kind > 1 || mode > 3 {
		return fmt.Errorf("invalid dual modulation size/kind/mode")
	}
	return nil
}

// DualNormForwardShape mode0 writes stats[M,5],mode1/2 writes value/tangent.
func DualNormForwardShape(m, n int, kind, mode uint32, epsilon float32) error {
	if err := DualNormShape(m, n, kind, 0); err != nil {
		return err
	}
	if mode > 2 || epsilon <= 0 || math.IsNaN(float64(epsilon)) || math.IsInf(float64(epsilon), 0) {
		return fmt.Errorf("invalid dual norm forward mode/epsilon")
	}
	return nil
}

// AttentionForwardShape uses the same exact causal/context envelope. Modes:
// 0=scaled scores[H,S,S], 1=stable softmax[H,S,S], 2=weighted values[S,H,D].
func AttentionForwardShape(rows, heads, dim, context int, mode uint32, scale float32) error {
	if mode > 2 {
		return fmt.Errorf("invalid training attention forward mode")
	}
	return AttentionShape(rows, heads, dim, context, 0, scale)
}

// AttentionShape bounds a causal self-attention backward pass. Q/K/V/dO are
// [S,H,D]; P and auxiliary dP/dScore are dense [H,S,S]. dScore already includes
// the supplied score scale; dQ/dK must not apply it again. Masked cells are
// explicitly zeroed in dP/dScore. Context0 means full causal prefix; otherwise
// valid key/query pairs satisfy key<=query && query-key<context.
// Modes:0=dP,1=scaled dScore,2=dQ,3=dK,4=dV. Inverse RoPE is a separate step.
func AttentionShape(rows, heads, dim, context int, mode uint32, scale float32) error {
	if rows < 1 || rows > 1024 || heads < 1 || heads > 32 || dim < 1 || dim > 256 || context < 0 || context > rows || mode > 4 || scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return fmt.Errorf("invalid training causal attention shape/mode/scale")
	}
	return nil
}
