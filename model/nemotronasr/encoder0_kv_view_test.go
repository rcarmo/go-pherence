package nemotronasr

import (
	"math"
	"reflect"
	"testing"
)

func TestEncoderKVInternalViewMatchesPackedAndRollback(t *testing.T) {
	var packed, view Encoder0KVCache
	for step, frames := range []int{4, 4, 4, 40, 5, 4, 4, 3} {
		keys, values := make([]float32, frames*encoderWidth), make([]float32, frames*encoderWidth)
		for i := range keys {
			keys[i] = float32(i + step*1000)
			values[i] = -keys[i]
		}
		wantK, wantV, err := packed.Update(keys, values, frames)
		if err != nil {
			t.Fatal(err)
		}
		gotK, gotV, err := view.updateVisibleView(keys, values, frames)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(gotK, wantK) || !reflect.DeepEqual(gotV, wantV) {
			t.Fatal("visible buffers differ", step)
		}
		pk, pv, pr, ps := packed.Snapshot()
		vk, vv, vr, vs := view.Snapshot()
		if len(view.keys) > (asrKVWindow-1+frames)*encoderWidth || len(view.values) > (asrKVWindow-1+frames)*encoderWidth {
			t.Fatal("visible history exceeded retained plus current rows", step)
		}
		if !reflect.DeepEqual(pk, vk) || !reflect.DeepEqual(pv, vv) || pr != vr || ps != vs {
			t.Fatal("retained state differs", step)
		}
		// A prepared transaction may advance, but its failure must not mutate the
		// old immutable view's backing storage or metadata.
		beforeK, beforeV, beforeR, beforeS := view.Snapshot()
		prepared := view
		next := make([]float32, 4*encoderWidth)
		if _, _, err := prepared.updateVisibleView(next, next, 4); err != nil {
			t.Fatal(err)
		}
		afterK, afterV, afterR, afterS := view.Snapshot()
		if !reflect.DeepEqual(beforeK, afterK) || !reflect.DeepEqual(beforeV, afterV) || beforeR != afterR || beforeS != afterS {
			t.Fatal("prepared update mutated old view")
		}
		// Public output remains independently mutable even after internal-view use.
		pk, pv, err = packed.Update(keys, values, frames)
		if err != nil {
			t.Fatal(err)
		}
		vk, vv, err = view.Update(keys, values, frames)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(pk, vk) || !reflect.DeepEqual(pv, vv) {
			t.Fatal("public update after view differs")
		}
		heldK, _, _, _ := view.Snapshot()
		vk[0] = float32(math.Inf(1))
		checkK, _, _, _ := view.Snapshot()
		if !reflect.DeepEqual(heldK, checkK) {
			t.Fatal("public output aliases state")
		}
	}
}
func TestEncoderKVViewValidationPreservesState(t *testing.T) {
	var c Encoder0KVCache
	keys := make([]float32, 4*encoderWidth)
	for i := range keys {
		keys[i] = float32(i)
	}
	if _, _, err := c.updateVisibleView(keys, keys, 4); err != nil {
		t.Fatal(err)
	}
	beforeK, beforeV, r, s := c.Snapshot()
	bad := append([]float32(nil), keys...)
	bad[0] = float32(math.NaN())
	if _, _, err := c.updateVisibleView(bad, keys, 4); err == nil {
		t.Fatal("NaN accepted")
	}
	k, v, rr, ss := c.Snapshot()
	if !reflect.DeepEqual(k, beforeK) || !reflect.DeepEqual(v, beforeV) || r != rr || s != ss {
		t.Fatal("bad update mutated state")
	}
	c.skip = c.stride + 1
	if _, _, err := c.updateVisibleView(keys, keys, 4); err == nil {
		t.Fatal("invalid view accepted")
	}
}
