// Independent CPU F32 GELU error-function fixture. Run in a disposable Cargo
// project with libm = "=0.2.16" and this file as the binary entrypoint:
// cargo run --release --offline > erff_libm_0_2_16.u32le
// Each record is little-endian [input_bits, libm::erff(input)_bits].
use std::io::{self, Write};

fn emit(w: &mut impl Write, bits: u32) -> io::Result<()> {
    w.write_all(&bits.to_le_bytes())?;
    w.write_all(&libm::erff(f32::from_bits(bits)).to_bits().to_le_bytes())
}
fn main() -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    for bits in [0u32, 0x80000000, 0x3f000000, 0x3f800000, 0x40000000,
                 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000] {
        emit(&mut out, bits)?;
    }
    // Check both sides of the underflow, polynomial, erfc, and saturation
    // branches, including negative zero and nonfinite values.
    for bits in [0x00000001u32, 0x007fffff, 0x00800000, 0x31800000,
                 0x3f580000, 0x3fa00000, 0x4036db6d, 0x40c00000,
                 0x41e00000] {
        for delta in -4i32..=4 {
            let b = bits.wrapping_add_signed(delta);
            emit(&mut out, b)?;
            emit(&mut out, b | 0x80000000)?;
        }
    }
    // Dense finite coverage of erfc2's expf branch and arbitrary bit patterns.
    let mut rng = 0x516f_52a5u32;
    for _ in 0..8192 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = (rng as f32) * (12.0f32 / u32::MAX as f32) - 6.0;
        emit(&mut out, v.to_bits())?;
    }
    for _ in 0..1024 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        emit(&mut out, rng)?;
    }
    out.flush()
}
