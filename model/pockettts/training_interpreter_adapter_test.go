package pockettts

// Additive TEST-ONLY adapter; scalar loops copied from the unchanged local
// reference interpreter. No grouped resident bridge or production export.
import (
	"context"
	"fmt"
	"math"

	train "github.com/rcarmo/go-pherence/backends/training"
)

type pocketAdapterLedger struct {
	FrozenConstants, BorrowCopies, SlotPayload, ReturnedPayload int64
	ReboundSlotPayload, MetadataCharge                          int64
	Dispatched                                                  int
}
type pocketAdapterOptions struct {
	reuse, shareZero bool
	before           func(int)
	after            func(int)
}

func pocketAdapterTuple(c train.Command) bool {
	if c.Kind != 0 && c.Op != "mod-forward" && c.Op != "mod-backward" && c.Op != "norm-forward" && c.Op != "norm-backward" {
		return false
	}
	switch c.Op {
	case "eos-admission", "reshape":
		return c.Mode == 0 && c.Kind == 0
	case "lsd-loss", "eos", "time-embedding", "attention-forward", "vector":
		return c.Mode <= 2 && c.Kind == 0
	case "embedding":
		return c.Mode <= 3 && c.Kind == 0
	case "relu", "gelu", "rope", "linear-forward", "silu-forward", "silu-backward":
		return c.Mode <= 1 && c.Kind == 0
	case "linear-backward", "row":
		return c.Mode <= 3 && c.Kind == 0
	case "attention-backward":
		return c.Mode <= 4 && c.Kind == 0
	case "mod-forward":
		return c.Mode <= 1 && c.Kind <= 1
	case "mod-backward":
		return c.Mode <= 3 && c.Kind <= 1
	case "norm-forward":
		return c.Mode <= 2 && c.Kind <= 1
	case "norm-backward":
		return c.Mode <= 3 && c.Kind <= 1 && !(c.Kind == 1 && c.Mode == 3)
	}
	return false
}
func pocketAdapterFreeze(source train.Tape) train.Tape {
	t := train.Tape{Tensors: make([]train.TensorSpec, len(source.Tensors)), Commands: make([]train.Command, len(source.Commands)), Outputs: append([]int(nil), source.Outputs...)}
	for i, s := range source.Tensors {
		s.Shape = append([]int(nil), s.Shape...)
		if s.Values != nil {
			s.Values = append([]float32{}, s.Values...)
		}
		t.Tensors[i] = s
	}
	for i, c := range source.Commands {
		c.Inputs = append([]int(nil), c.Inputs...)
		t.Commands[i] = c
	}
	return t
}
func runPocketAdapter(ctx context.Context, source train.Tape, borrowed map[int][]float32, protected map[int]bool, opt pocketAdapterOptions) (returned [][]float32, ledger pocketAdapterLedger, err error) {
	if ctx == nil {
		return nil, ledger, fmt.Errorf("adapter nil context")
	}
	if err = ctx.Err(); err != nil {
		return nil, ledger, err
	}
	// Bound all copied/nested inventories before validation walks or allocation.
	if len(source.Tensors) == 0 || len(source.Tensors) > 4096 || len(source.Commands) > 4096 || len(source.Outputs) == 0 || len(source.Outputs) > 4096 || len(borrowed) > 4096 || len(protected) > 4096 {
		return nil, ledger, fmt.Errorf("adapter metadata cardinality")
	}
	// Conservative planner/validation scratch is charged even though temporary.
	ledger.MetadataCharge = int64(len(source.Tensors))*64 + int64(len(source.Commands))*32 + int64(len(source.Tensors)+len(source.Commands))*256 + int64(len(source.Outputs))*8 + int64(len(borrowed)+len(protected))*128
	for _, s := range source.Tensors {
		if len(s.Shape) < 1 || len(s.Shape) > 3 {
			return nil, ledger, fmt.Errorf("adapter shape bound")
		}
		ledger.MetadataCharge += int64(len(s.Shape)) * 8
		ledger.FrozenConstants += int64(len(s.Values)) * 4
	}
	for _, c := range source.Commands {
		if len(c.Inputs) > 16 || len(c.Op) > 64 || !pocketAdapterTuple(c) {
			return nil, ledger, fmt.Errorf("adapter unsupported tuple %s/%d/%d", c.Op, c.Mode, c.Kind)
		}
		ledger.MetadataCharge += int64(len(c.Inputs))*8 + int64(len(c.Op))
	}
	if ledger.MetadataCharge > 1<<20 {
		return nil, ledger, fmt.Errorf("adapter metadata byte ceiling")
	}
	if err = source.Validate(1 << 20); err != nil {
		return nil, ledger, err
	}
	seenOutputs := map[int]bool{}
	var resultBytes int64
	for _, id := range source.Outputs {
		if seenOutputs[id] {
			return nil, ledger, fmt.Errorf("adapter duplicate output")
		}
		seenOutputs[id] = true
		n, _ := source.Tensors[id].Elements()
		resultBytes += 4 * int64(n)
		if resultBytes > 1<<20 {
			return nil, ledger, fmt.Errorf("adapter output payload ceiling")
		}
	}
	if err = train.ValidateExternalTapeBindings(source, borrowed); err != nil {
		return nil, ledger, err
	}
	for id, v := range borrowed {
		n, e := source.Tensors[id].Elements()
		if e != nil || len(v) != n || !finiteF32(v) {
			return nil, ledger, fmt.Errorf("adapter bound extent/nonfinite")
		}
		ledger.BorrowCopies += int64(n) * 4
	}
	for id, yes := range protected {
		if yes && (id < 0 || id >= len(source.Tensors)) {
			return nil, ledger, fmt.Errorf("adapter protected ID")
		}
	}
	// Defensive host copies preserve caller data even if a test callback mutates it.
	tape := pocketAdapterFreeze(source)
	bindings := map[int][]float32{}
	for id, v := range borrowed {
		bindings[id] = append([]float32(nil), v...)
	}
	// Validate exactly the frozen values that will execute, while preserving
	// the earlier full original-source admission (including dead sources).
	// This temporary header copy is an admission view, not execution storage.
	effective := tape
	effective.Tensors = append([]train.TensorSpec(nil), tape.Tensors...)
	ledger.MetadataCharge += int64(len(tape.Tensors)) * 128
	// Charge the second validation's tensor-inventory scratch too.
	ledger.MetadataCharge += int64(len(tape.Tensors)) * 16
	if ledger.MetadataCharge > 1<<20 {
		return nil, ledger, fmt.Errorf("adapter effective metadata ceiling")
	}
	for id, v := range bindings {
		effective.Tensors[id].Values = v
		effective.Tensors[id].External = false
	}
	if err = effective.Validate(1 << 20); err != nil {
		return nil, ledger, fmt.Errorf("adapter effective bound admission: %w", err)
	}
	zeroProtected := map[int]bool{}
	for id, yes := range protected {
		zeroProtected[id] = yes
	}
	for id := range bindings {
		zeroProtected[id] = true
	}
	values := make([][]float32, len(tape.Tensors))
	var slots [][]float32
	var storage train.TapeStoragePlan
	reuse := opt.reuse
	if reuse {
		if opt.shareZero {
			storage, err = train.PlanZeroTapeStorage(tape, 1<<20, zeroProtected)
		} else {
			storage, err = train.PlanTapeStorage(tape, 1<<20)
		}
		if err != nil {
			return nil, ledger, err
		}
		for _, n := range storage.Elements {
			slots = append(slots, make([]float32, n))
			ledger.SlotPayload += 4 * int64(n)
		}
		for i, id := range storage.Slots {
			values[i] = slots[id]
		}
	}
	for i, s := range tape.Tensors {
		if s.Values != nil {
			if reuse {
				copy(values[i], s.Values)
			} else {
				values[i] = append([]float32(nil), s.Values...)
				ledger.SlotPayload += 4 * int64(len(s.Values))
			}
		}
	}
	rebound := map[int]bool{}
	for id, v := range bindings {
		if reuse {
			slot := storage.Slots[id]
			if !rebound[slot] {
				ledger.ReboundSlotPayload += 4 * int64(storage.Elements[slot])
				rebound[slot] = true
			}
			for alias, s := range storage.Slots {
				if s == slot {
					values[alias] = v
				}
			}
		} else {
			values[id] = v
		}
	}
	for index, c := range tape.Commands {
		if err = ctx.Err(); err != nil {
			return nil, ledger, err
		}
		if opt.before != nil {
			opt.before(index)
		}
		if err = ctx.Err(); err != nil {
			return nil, ledger, err
		}
		in := func(i int) []float32 { return values[c.Inputs[i]] }
		shape := tape.Tensors[c.Output].Shape
		n, _ := tape.Tensors[c.Output].Elements()
		var out []float32
		if reuse {
			out = values[c.Output]
			if c.Op != "reshape" {
				clear(out)
			}
		} else {
			out = make([]float32, n)
			values[c.Output] = out
			ledger.SlotPayload += 4 * int64(n)
		}
		switch c.Op {
		case "eos-admission":
			flag, err := train.EOSAdmissionFlag(in(0), in(1), in(2)[0], c.Rows, c.Width)
			if err != nil {
				return nil, ledger, err
			}
			out[0] = flag
		case "lsd-loss":
			channels := len(in(0))
			if c.Mode == 2 {
				cutoff := in(2)[0]
				for i, p := range in(0) {
					y := in(1)[i]
					d := p - y
					if !isFinite(p) || !isFinite(y) || !isFinite(d) || float32(math.Abs(float64(d))) > cutoff {
						out[0] = 1
					}
				}
				break
			}
			scale := float32(math.Exp(float64(in(2)[0]))) / float32(channels)
			if c.Mode == 1 {
				for i, p := range in(0) {
					out[i] = (2 * (p - in(1)[i])) * scale
				}
			} else {
				var square, correction float32
				for i, p := range in(0) {
					d := p - in(1)[i]
					term := d*d - correction
					next := square + term
					correction = (next - square) - term
					square = next
				}
				normalized := square * scale
				out[0], out[1], out[2] = square, normalized-in(2)[0], normalized-1
			}
		case "eos":
			if c.Mode == 2 {
				var sum, correction float32
				for i, x := range in(0) {
					if in(1)[i] >= 0 {
						term := x - correction
						next := sum + term
						correction = (next - sum) - term
						sum = next
					}
				}
				out[0] = sum / float32(c.Rows)
			} else {
				for i, x := range in(0) {
					label := in(1)[i]
					if label < 0 {
						continue
					}
					e := float32(math.Exp(-math.Abs(float64(x))))
					if c.Mode == 0 {
						var base float32
						if label == 1 && x < 0 {
							base = -x
						} else if label == 0 && x > 0 {
							base = x
						}
						out[i] = base + float32(math.Log1p(float64(e)))
					} else {
						var residual float32
						if label == 1 {
							if x >= 0 {
								residual = -e / (1 + e)
							} else {
								residual = -1 / (1 + e)
							}
						} else {
							if x >= 0 {
								residual = 1 / (1 + e)
							} else {
								residual = e / (1 + e)
							}
						}
						out[i] = (residual / float32(c.Rows)) * c.Scale
					}
				}
			}
		case "time-embedding":
			half := len(in(0))
			time, dtime := in(1)[0], in(1)[1]
			if c.Mode < 2 {
				for i := range out {
					f := in(0)[i%half]
					angle := time * f
					s, co := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
					if c.Mode == 0 {
						if i < half {
							out[i] = co
						} else {
							out[i] = s
						}
					} else {
						if i < half {
							out[i] = (-s * f) * dtime
						} else {
							out[i] = (co * f) * dtime
						}
					}
				}
			} else {
				for j, f := range in(0) {
					angle := time * f
					s, co := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
					fc, fs, sc, ss := -s*f, co*f, -co*f*f, -s*f*f
					out[0] += in(2)[j]*fc + in(2)[half+j]*fs
					out[0] += in(3)[j]*sc*dtime + in(3)[half+j]*ss*dtime
				}
			}
		case "embedding":
			for i := range out {
				if c.Mode == 0 {
					out[i] = in(0)[int(in(1)[i/c.Width])*c.Width+i%c.Width]
				} else if c.Mode == 3 {
					slot := i / c.Width
					if in(1)[slot] >= 0 {
						for at := int(in(1)[c.Rows+slot]); at < int(in(1)[c.Rows+slot+1]); at++ {
							row := int(in(1)[2*c.Rows+1+at])
							out[i] += in(0)[row*c.Width+i%c.Width]
						}
					}
				} else if c.Mode == 2 {
					token := i / c.Width
					for at := int(in(1)[token]); at < int(in(1)[token+1]); at++ {
						row := int(in(1)[c.Parts+1+at])
						out[i] += in(0)[row*c.Width+i%c.Width]
					}
				} else {
					for row := 0; row < c.Rows; row++ {
						if in(1)[row] == float32(i/c.Width) {
							out[i] += in(0)[row*c.Width+i%c.Width]
						}
					}
				}
			}
		case "relu":
			for i, x := range in(0) {
				if c.Mode == 0 {
					out[i] = x
					if x < 0 {
						out[i] = 0
					}
				} else if x > 0 {
					out[i] = in(1)[i]
				}
			}
		case "gelu":
			for i, x := range in(0) {
				u := float32(.7978845608028654) * (x + ((float32(.044715)*x)*x)*x)
				e := float32(math.Exp(float64(-2 * float32(math.Abs(float64(u))))))
				halfOnePlus := float32(1) / (1 + e)
				if u < 0 {
					halfOnePlus = e / (1 + e)
				}
				sech2 := (4 * e) / ((1 + e) * (1 + e))
				if c.Mode == 0 {
					out[i] = x * halfOnePlus
				} else {
					du := float32(.7978845608028654) * (1 + (float32(.134145)*x)*x)
					derivative := halfOnePlus + ((.5*x)*sech2)*du
					out[i] = in(1)[i] * derivative
				}
			}
		case "row":
			for i := range out {
				switch c.Mode {
				case 0:
					out[i] = in(0)[(i/c.Width*c.Parts+c.Slot)*c.Width+i%c.Width]
				case 1:
					col, part, row := i%c.Width, (i/c.Width)%c.Parts, i/(c.Parts*c.Width)
					if part == c.Slot {
						out[i] = in(0)[row*c.Width+col]
					}
				case 2:
					out[i] = in(0)[i] * in(1)[i%c.Width]
				case 3:
					for row := 0; row < c.Rows; row++ {
						out[i] += in(0)[row*c.Width+i] * in(1)[row*c.Width+i]
					}
				}
			}
		case "attention-backward":
			xs := tape.Tensors[c.Inputs[0]].Shape
			rows, heads, dim := xs[0], xs[1], xs[2]
			allowed := func(query, key int) bool { return key <= query && (c.Offset == 0 || query-key < c.Offset) }
			if c.Mode < 2 {
				for h := 0; h < heads; h++ {
					for query := 0; query < rows; query++ {
						base := (h*rows + query) * rows
						var weighted float32
						if c.Mode == 1 {
							for key := 0; key < rows; key++ {
								if allowed(query, key) {
									weighted += in(3)[base+key] * in(5)[base+key]
								}
							}
						}
						for key := 0; key < rows; key++ {
							if !allowed(query, key) {
								continue
							}
							if c.Mode == 0 {
								for d := 0; d < dim; d++ {
									out[base+key] += in(4)[(query*heads+h)*dim+d] * in(2)[(key*heads+h)*dim+d]
								}
							} else {
								out[base+key] = (in(5)[base+key] - weighted) * in(3)[base+key] * c.Scale
							}
						}
					}
				}
			} else {
				for row := 0; row < rows; row++ {
					for h := 0; h < heads; h++ {
						for d := 0; d < dim; d++ {
							idx := (row*heads+h)*dim + d
							for j := 0; j < rows; j++ {
								if c.Mode == 2 {
									if allowed(row, j) {
										out[idx] += in(5)[(h*rows+row)*rows+j] * in(1)[(j*heads+h)*dim+d]
									}
								} else if allowed(j, row) {
									if c.Mode == 3 {
										out[idx] += in(5)[(h*rows+j)*rows+row] * in(0)[(j*heads+h)*dim+d]
									} else {
										out[idx] += in(3)[(h*rows+j)*rows+row] * in(4)[(j*heads+h)*dim+d]
									}
								}
							}
						}
					}
				}
			}
		case "attention-forward":
			xs := tape.Tensors[c.Inputs[0]].Shape
			rows, heads, dim := xs[0], xs[1], xs[2]
			allowed := func(query, key int) bool { return key <= query && (c.Offset == 0 || query-key < c.Offset) }
			if c.Mode == 0 {
				for h := 0; h < heads; h++ {
					for query := 0; query < rows; query++ {
						for key := 0; key < rows; key++ {
							if !allowed(query, key) {
								continue
							}
							var sum float32
							for d := 0; d < dim; d++ {
								sum += in(0)[(query*heads+h)*dim+d] * in(1)[(key*heads+h)*dim+d]
							}
							out[(h*rows+query)*rows+key] = sum * c.Scale
						}
					}
				}
			}
			if c.Mode == 1 {
				for h := 0; h < heads; h++ {
					for query := 0; query < rows; query++ {
						base := (h*rows + query) * rows
						maximum := float32(-math.MaxFloat32)
						for key := 0; key < rows; key++ {
							if allowed(query, key) {
								maximum = max(maximum, in(3)[base+key])
							}
						}
						var sum float32
						for key := 0; key < rows; key++ {
							if allowed(query, key) {
								out[base+key] = float32(math.Exp(float64(in(3)[base+key] - maximum)))
								sum += out[base+key]
							}
						}
						for key := 0; key < rows; key++ {
							out[base+key] /= sum
						}
					}
				}
			}
			if c.Mode == 2 {
				for query := 0; query < rows; query++ {
					for h := 0; h < heads; h++ {
						for d := 0; d < dim; d++ {
							for key := 0; key < rows; key++ {
								if allowed(query, key) {
									out[(query*heads+h)*dim+d] += in(3)[(h*rows+query)*rows+key] * in(2)[(key*heads+h)*dim+d]
								}
							}
						}
					}
				}
			}
		case "rope":
			xs := tape.Tensors[c.Inputs[0]].Shape
			for i := 0; i < len(out); i += 2 {
				row, col := i/(xs[1]*xs[2]), i%xs[2]
				cosine, sine := in(1)[row*xs[2]+col], in(1)[row*xs[2]+col+1]
				if c.Mode == 1 {
					sine = -sine
				}
				out[i] = in(0)[i]*cosine - in(0)[i+1]*sine
				out[i+1] = in(0)[i+1]*cosine + in(0)[i]*sine
			}
		case "reshape":
			if !reuse {
				values[c.Output] = append([]float32(nil), in(0)...)
				ledger.SlotPayload += 4 * int64(len(in(0))) // nonreuse reshape copy, including discarded fresh out
			}
		case "vector":
			for i := range out {
				switch c.Mode {
				case 0:
					out[i] = in(0)[i] + in(1)[i]*c.Scale
				case 1:
					out[i] = in(0)[c.Offset+i]
				case 2:
					if i >= c.Offset && i-c.Offset < len(in(0)) {
						out[i] = in(0)[i-c.Offset]
					}
				}
			}
		case "linear-forward":
			xs, ws := tape.Tensors[c.Inputs[0]].Shape, tape.Tensors[c.Inputs[2]].Shape
			m, k, n := xs[0], xs[1], ws[0]
			for r := 0; r < m; r++ {
				for col := 0; col < n; col++ {
					sum := float32(0)
					x := in(0)
					if c.Mode == 1 {
						x = in(1)
					}
					for q := 0; q < k; q++ {
						sum += x[r*k+q] * in(2)[col*k+q]
					}
					if c.Mode == 0 {
						sum += in(3)[col]
					}
					out[r*n+col] = sum
				}
			}
		case "linear-backward":
			xs, ws := tape.Tensors[c.Inputs[0]].Shape, tape.Tensors[c.Inputs[2]].Shape
			m, k, n := xs[0], xs[1], ws[0]
			for i := range out {
				if c.Mode < 2 {
					row, col := i/k, i%k
					seed := in(3)
					if c.Mode == 1 {
						seed = in(4)
					}
					for q := 0; q < n; q++ {
						out[i] += seed[row*n+q] * in(2)[q*k+col]
					}
				} else if c.Mode == 2 {
					row, col := i/k, i%k
					for q := 0; q < m; q++ {
						out[i] += in(3)[q*n+row] * in(0)[q*k+col]
					}
					for q := 0; q < m; q++ {
						out[i] += in(4)[q*n+row] * in(1)[q*k+col]
					}
				} else {
					for q := 0; q < m; q++ {
						out[i] += in(3)[q*n+i]
					}
				}
			}
		case "silu-forward", "silu-backward":
			for i, x := range in(0) {
				s := sigmoidF32(x)
				first := s * (1 + x*(1-s))
				if c.Op == "silu-forward" {
					if c.Mode == 0 {
						out[i] = x * s
					} else {
						out[i] = (in(1)[i] * s) * (1 + x*(1-s))
					}
				} else {
					second := s * (1 - s) * (2 + x*(1-2*s))
					if c.Mode == 0 {
						out[i] = in(2)[i]*first + (in(3)[i]*second)*in(1)[i]
					} else {
						out[i] = in(3)[i] * first
					}
				}
			}
		case "mod-forward", "mod-backward":
			count := len(in(0))
			for i := range out {
				col := i % count
				a, at := in(0)[col], in(1)[col]
				if c.Op == "mod-forward" {
					b, bt := in(2)[col], in(3)[col]
					if c.Kind == 0 {
						b = 1 + in(2)[count+col]
						bt = in(3)[count+col]
					}
					if c.Mode == 0 {
						out[i] = a * b
						if c.Kind == 0 {
							out[i] += in(2)[col]
						}
					} else {
						out[i] = at*b + a*bt
						if c.Kind == 0 {
							out[i] += in(3)[col]
						}
					}
				} else {
					gv, gt := in(4)[col], in(5)[col]
					if c.Kind == 0 {
						if c.Mode < 2 {
							scale := 1 + in(2)[count+col]
							if c.Mode == 0 {
								out[i] = gv*scale + gt*in(3)[count+col]
							} else {
								out[i] = gt * scale
							}
						} else if i < count {
							if c.Mode == 2 {
								out[i] = gv
							} else {
								out[i] = gt
							}
						} else if c.Mode == 2 {
							out[i] = gv*a + gt*at
						} else {
							out[i] = gt * a
						}
					} else {
						switch c.Mode {
						case 0:
							out[i] = gv*in(2)[i] + gt*in(3)[i]
						case 1:
							out[i] = gt * in(2)[i]
						case 2:
							out[i] = gv*a + gt*at
						case 3:
							out[i] = gt * a
						}
					}
				}
			}
		case "norm-forward":
			xs := tape.Tensors[c.Inputs[0]].Shape
			m, width := xs[0], xs[1]
			for row := 0; row < m; row++ {
				base := row * width
				if c.Mode == 0 {
					mean, dm := float32(0), float32(0)
					for col := 0; col < width; col++ {
						mean += in(0)[base+col]
						dm += in(1)[base+col]
					}
					mean /= float32(width)
					dm /= float32(width)
					v, dv := float32(0), float32(0)
					for col := 0; col < width; col++ {
						center, dc := in(0)[base+col]-mean, in(1)[base+col]-dm
						v += center * center
						dv += 2 * center * dc
					}
					denom := float32(width)
					if c.Kind == 1 {
						denom--
					}
					v /= denom
					dv /= denom
					inv := float32(1 / math.Sqrt(float64(v+c.Epsilon)))
					dinv := -.5 * inv * inv * inv * dv
					copy(out[row*5:], []float32{mean, dm, inv, dinv, dv})
				} else {
					stats := in(4)[row*5 : row*5+5]
					for col := 0; col < width; col++ {
						v, dv := in(0)[base+col], in(1)[base+col]
						if c.Kind == 0 {
							v -= stats[0]
							dv -= stats[1]
						}
						if c.Mode == 1 {
							out[base+col] = (v * in(2)[col]) * stats[2]
							if c.Kind == 0 {
								out[base+col] = (v*stats[2])*in(2)[col] + in(3)[col]
							}
						} else {
							out[base+col] = in(2)[col] * (dv*stats[2] + v*stats[3])
						}
					}
				}
			}
		case "norm-backward":
			xs := tape.Tensors[c.Inputs[0]].Shape
			m, width := xs[0], xs[1]
			if c.Mode >= 2 {
				for col := range out {
					for row := 0; row < m; row++ {
						i := row*width + col
						stats := in(5)[row*5 : row*5+5]
						v, dv := in(0)[i], in(1)[i]
						if c.Kind == 0 {
							v -= stats[0]
							dv -= stats[1]
						}
						if c.Mode == 3 {
							out[col] += in(3)[i]
						} else {
							out[col] += (in(3)[i]*v)*stats[2] + in(4)[i]*(dv*stats[2]+v*stats[3])
						}
					}
				}
			} else {
				for row := 0; row < m; row++ {
					base := row * width
					stats := in(5)[row*5 : row*5+5]
					mean, dm, inv, dinv, dvar := stats[0], stats[1], stats[2], stats[3], stats[4]
					gi, gdi := float32(0), float32(0)
					for col := 0; col < width; col++ {
						v, dv := in(0)[base+col], in(1)[base+col]
						if c.Kind == 0 {
							v -= mean
							dv -= dm
						}
						g, h := in(3)[base+col]*in(2)[col], in(4)[base+col]*in(2)[col]
						gi += g*v + h*dv
						gdi += h * v
					}
					gvar := gi*(-.5)*inv*inv*inv + gdi*.75*inv*inv*inv*inv*inv*dvar
					gdvar := gdi * (-.5) * inv * inv * inv
					denom := float32(width)
					if c.Kind == 1 {
						denom--
					}
					total := float32(0)
					for col := 0; col < width; col++ {
						center, dc := in(0)[base+col]-mean, in(1)[base+col]-dm
						g, h := in(3)[base+col]*in(2)[col], in(4)[base+col]*in(2)[col]
						v := h * inv
						if c.Mode == 0 {
							v = g*inv + h*dinv
							v += gvar*2*center/denom + gdvar*2*dc/denom
						} else {
							v += gdvar * 2 * center / denom
						}
						out[base+col] = v
						total -= v
					}
					if c.Kind == 0 {
						for col := 0; col < width; col++ {
							out[base+col] += total / float32(width)
						}
					}
				}
			}
		default:
			return nil, ledger, fmt.Errorf("adapter unsupported %s %v", c.Op, shape)
		}
		ledger.Dispatched++
		if opt.after != nil {
			opt.after(index)
		}
		if err := ctx.Err(); err != nil {
			return nil, ledger, err
		}
	}

	returned = make([][]float32, len(tape.Outputs))
	for i, id := range tape.Outputs {
		if !finiteF32(values[id]) {
			return nil, ledger, fmt.Errorf("adapter returned nonfinite")
		}
		returned[i] = append([]float32(nil), values[id]...)
		ledger.ReturnedPayload += 4 * int64(len(returned[i]))
	}
	return returned, ledger, nil
}
