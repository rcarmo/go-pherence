# IME2 vmadot Test Vectors

Verified on SpacemiT K3 (X100 cores), VLEN=256, SEW=8.

## vmadot (signed × signed, funct3=3)

### Assembly encoding
```
vmadot v28, v0, v1  →  0xe2103e2b
```
GCC flag: `-march=rv64gcv_xsmtvdotii`

### Operation
```
C[4×4] += A[4×8] × B[4×8]^T
```
Where A is in v0 (32 bytes), B in v1 (32 bytes), C in v28:v29 (32 bytes int32).

### Test vector 1: Identity check
```
A = [1,0,0,0,0,0,0,0,  0,1,0,0,0,0,0,0,  0,0,1,0,0,0,0,0,  0,0,0,1,0,0,0,0]
B = [1,1,1,1,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0]
Expected C[0][0] = 1, C[1][0] = 0, C[2][0] = 0, C[3][0] = 0
```

### Test vector 2: Random (verified bit-exact)
```
A[i] = (i*7 + 3) % 127 - 63  (for i=0..31)
B[i] = (i*13 + 5) % 127 - 63 (for i=0..31)

Expected C (4×4 int32):
  C[0][0]=7372  C[0][1]=-447  C[0][2]=-4710 C[0][3]=-2242
  C[1][0]=1772  C[1][1]=-2127 C[1][2]=-2470 C[1][3]=-3194
  C[2][0]=-4209 C[2][1]=13338 C[2][2]=2183  C[2][3]=-336
  C[3][0]=3272  C[3][1]=-1677 C[3][2]=-3070 C[3][3]=-2939
```

## Key findings

1. **X100 cores are CPU 0-7** (not 8-15 as documentation might suggest)
2. **INT8 only reliable** — INT16 produces arithmetic errors (confirmed by Remlab)
3. **vd must be even** (EMUL=2 for int32 accumulator spanning 2 vector registers)
4. **vl=32 required** for 4×8 matrix layout (VLEN=256, SEW=8)
5. **B is implicitly transposed** in the vmadot operation
