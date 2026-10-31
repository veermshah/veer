//! Number formats, quantization simulators, and the sound error-bound algebra
//! used by Veer's precision type layer.
//!
//! Every tensor value in Veer carries a static [`Bound`]: an upper bound `m` on
//! the magnitude of the *exact* values (‖t‖∞ ≤ m) and an upper bound `e` on the
//! absolute error of the computed values (‖t̂ − t‖∞ ≤ e). The functions in
//! [`bound`] propagate these bounds through quantization and arithmetic; the
//! simulators in [`Format::quantize`] and [`tensor`] are what the bounds are
//! tested against (see `tests/soundness.rs`).

pub mod tensor;

/// Unit roundoff of IEEE f32 (round to nearest).
pub const U32: f64 = 5.960464477539063e-8; // 2^-24

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    F32,
    Bf16,
    F16,
    Fp8E4m3,
    Fp8E5m2,
    Int8,
    /// OCP Microscaling FP4 (E2M1 elements, E8M0 power-of-two scale, block 32),
    /// scale chosen as in the OCP MX v1.0 spec: 2^(floor(log2(amax)) - 2).
    Mxfp4,
    /// MXFP4 with overflow-aware scaling: 2^ceil(log2(amax / 6)), which never clamps.
    Mxfp4Oas,
    /// NVIDIA NVFP4 (E2M1 elements, FP8 E4M3 block scale, block 16, FP32 tensor scale).
    Nvfp4,
}

/// Parameters of an IEEE-like minifloat.
#[derive(Clone, Copy, Debug)]
pub struct Minifloat {
    pub mant_bits: i32,
    pub emin: i32,
    pub max: f64,
}

pub const E2M1: Minifloat = Minifloat { mant_bits: 1, emin: 0, max: 6.0 };
pub const E4M3: Minifloat = Minifloat { mant_bits: 3, emin: -6, max: 448.0 };
pub const E5M2: Minifloat = Minifloat { mant_bits: 2, emin: -14, max: 57344.0 };
pub const F16: Minifloat = Minifloat { mant_bits: 10, emin: -14, max: 65504.0 };
pub const BF16: Minifloat = Minifloat { mant_bits: 7, emin: -126, max: 3.3895313892515355e38 };

impl Minifloat {
    /// Unit roundoff for normal numbers.
    pub fn u(&self) -> f64 {
        2f64.powi(-(self.mant_bits + 1))
    }
    /// Half the subnormal spacing: absolute error floor near zero.
    pub fn floor(&self) -> f64 {
        2f64.powi(self.emin - self.mant_bits - 1)
    }
    /// Round to nearest (ties to even), saturating at ±max.
    pub fn round(&self, x: f64) -> f64 {
        if x == 0.0 || !x.is_finite() {
            return if x.is_nan() { 0.0 } else if x.is_infinite() { self.max.copysign(x) } else { 0.0 };
        }
        let a = x.abs();
        let e = exponent(a).max(self.emin);
        let quantum = 2f64.powi(e - self.mant_bits);
        let q = round_half_even(a / quantum) * quantum;
        q.min(self.max).copysign(x)
    }
}

/// floor(log2(a)) for finite positive a, exact (via the bit pattern).
fn exponent(a: f64) -> i32 {
    let bits = a.to_bits();
    let raw = ((bits >> 52) & 0x7ff) as i32;
    if raw == 0 {
        // subnormal f64: normalize
        exponent(a * 2f64.powi(54)) - 54
    } else {
        raw - 1023
    }
}

fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 {
        let t = v.trunc();
        if (t as i64) % 2 == 0 {
            t
        } else {
            t + v.signum()
        }
    } else {
        r
    }
}

/// How a format's quantization error is bounded.
#[derive(Clone, Copy, Debug)]
pub enum ErrModel {
    /// |q(x) − x| ≤ u·|x| + floor, and |x| must not exceed `max`.
    PerElement { u: f64, floor: f64, max: f64 },
    /// |q(x) − x| ≤ c · max|x| over the tensor (block maxima are ≤ the tensor max).
    Scaled { c: f64 },
}

pub const ALL_FORMATS: &[Format] =
    &[Format::F32, Format::Bf16, Format::F16, Format::Fp8E4m3, Format::Fp8E5m2, Format::Int8, Format::Mxfp4, Format::Mxfp4Oas, Format::Nvfp4];

impl Format {
    pub fn parse(s: &str) -> Option<Format> {
        Some(match s {
            "f32" | "fp32" => Format::F32,
            "bf16" => Format::Bf16,
            "f16" | "fp16" => Format::F16,
            "fp8_e4m3" | "fp8" => Format::Fp8E4m3,
            "fp8_e5m2" => Format::Fp8E5m2,
            "int8" => Format::Int8,
            "mxfp4" => Format::Mxfp4,
            "mxfp4_oas" => Format::Mxfp4Oas,
            "nvfp4" => Format::Nvfp4,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Format::F32 => "f32",
            Format::Bf16 => "bf16",
            Format::F16 => "f16",
            Format::Fp8E4m3 => "fp8_e4m3",
            Format::Fp8E5m2 => "fp8_e5m2",
            Format::Int8 => "int8",
            Format::Mxfp4 => "mxfp4",
            Format::Mxfp4Oas => "mxfp4_oas",
            Format::Nvfp4 => "nvfp4",
        }
    }

    /// Storage cost per element in bits, including amortized block scales.
    pub fn bits_per_elem(self) -> f64 {
        match self {
            Format::F32 => 32.0,
            Format::Bf16 | Format::F16 => 16.0,
            Format::Fp8E4m3 | Format::Fp8E5m2 | Format::Int8 => 8.0,
            Format::Mxfp4 | Format::Mxfp4Oas => 4.0 + 8.0 / 32.0,
            Format::Nvfp4 => 4.0 + 8.0 / 16.0,
        }
    }

    pub fn block_size(self) -> Option<usize> {
        match self {
            Format::Mxfp4 | Format::Mxfp4Oas => Some(32),
            Format::Nvfp4 => Some(16),
            _ => None,
        }
    }

    pub fn err_model(self) -> ErrModel {
        match self {
            Format::F32 => ErrModel::PerElement { u: 0.0, floor: 0.0, max: f32::MAX as f64 },
            Format::Bf16 => ErrModel::PerElement { u: BF16.u(), floor: BF16.floor(), max: BF16.max },
            Format::F16 => ErrModel::PerElement { u: F16.u(), floor: F16.floor(), max: F16.max },
            Format::Fp8E4m3 => ErrModel::PerElement { u: E4M3.u(), floor: E4M3.floor(), max: E4M3.max },
            Format::Fp8E5m2 => ErrModel::PerElement { u: E5M2.u(), floor: E5M2.floor(), max: E5M2.max },
            // scale = amax/127, error ≤ scale/2; small slack for f32 storage of q·scale.
            Format::Int8 => ErrModel::Scaled { c: 1.0 / 254.0 + 2.0 * U32 },
            // amax/scale ∈ [4, 8): values in (6, 8) clamp to 6 → error up to 2·scale ≤ amax/2.
            Format::Mxfp4 => ErrModel::Scaled { c: 0.5 + 2.0 * U32 },
            // amax/scale ∈ (3, 6]: no clamping; half-spacing 1·scale < amax/3.
            Format::Mxfp4Oas => ErrModel::Scaled { c: 1.0 / 3.0 + 2.0 * U32 },
            // e4m3 scale rounding (≤ 1/16 rel) → error ≤ (17/96)·bmax, plus a tiny
            // subnormal-scale floor relative to the tensor max.
            Format::Nvfp4 => ErrModel::Scaled { c: 17.0 / 96.0 + 3e-6 + 2.0 * U32 },
        }
    }

    /// Worst-case relative element error, as a human-facing summary.
    pub fn rel_err_summary(self) -> f64 {
        match self.err_model() {
            ErrModel::PerElement { u, .. } => u,
            ErrModel::Scaled { c } => c,
        }
    }

    /// Simulate quantize→dequantize of a tensor, returning the values that the
    /// hardware would compute with (as f32).
    pub fn quantize(self, xs: &[f32]) -> Vec<f32> {
        match self {
            Format::F32 => xs.to_vec(),
            Format::Bf16 => xs.iter().map(|&x| BF16.round(x as f64) as f32).collect(),
            Format::F16 => xs.iter().map(|&x| F16.round(x as f64) as f32).collect(),
            Format::Fp8E4m3 => xs.iter().map(|&x| E4M3.round(x as f64) as f32).collect(),
            Format::Fp8E5m2 => xs.iter().map(|&x| E5M2.round(x as f64) as f32).collect(),
            Format::Int8 => {
                let amax = xs.iter().fold(0f64, |m, &x| m.max((x as f64).abs()));
                if amax == 0.0 {
                    return vec![0.0; xs.len()];
                }
                let scale = amax / 127.0;
                xs.iter().map(|&x| ((x as f64 / scale).round().clamp(-127.0, 127.0) * scale) as f32).collect()
            }
            Format::Mxfp4 | Format::Mxfp4Oas => {
                let mut out = Vec::with_capacity(xs.len());
                for block in xs.chunks(32) {
                    let amax = block.iter().fold(0f64, |m, &x| m.max((x as f64).abs()));
                    if amax == 0.0 {
                        out.extend(std::iter::repeat_n(0.0, block.len()));
                        continue;
                    }
                    let e = if self == Format::Mxfp4 {
                        exponent(amax) - 2 // emax of E2M1 is 2
                    } else {
                        (amax / 6.0).log2().ceil() as i32
                    };
                    let e = e.clamp(-127, 127);
                    let scale = 2f64.powi(e);
                    out.extend(block.iter().map(|&x| (E2M1.round(x as f64 / scale) * scale) as f32));
                }
                out
            }
            Format::Nvfp4 => {
                let tmax = xs.iter().fold(0f64, |m, &x| m.max((x as f64).abs()));
                if tmax == 0.0 {
                    return vec![0.0; xs.len()];
                }
                let g = tmax / (448.0 * 6.0);
                let mut out = Vec::with_capacity(xs.len());
                for block in xs.chunks(16) {
                    let bmax = block.iter().fold(0f64, |m, &x| m.max((x as f64).abs()));
                    let s = E4M3.round(bmax / 6.0 / g) * g;
                    if s == 0.0 {
                        out.extend(std::iter::repeat_n(0.0, block.len()));
                        continue;
                    }
                    out.extend(block.iter().map(|&x| (E2M1.round(x as f64 / s) * s) as f32));
                }
                out
            }
        }
    }
}

/// Static magnitude/error bound carried by every tensor type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bound {
    /// ‖exact‖∞ ≤ m
    pub m: f64,
    /// ‖computed − exact‖∞ ≤ e
    pub e: f64,
}

impl Bound {
    pub fn exact(m: f64) -> Bound {
        Bound { m, e: 0.0 }
    }
    /// e / m, the bound on error relative to the largest possible value.
    pub fn rel(&self) -> f64 {
        if self.m == 0.0 {
            if self.e == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            self.e / self.m
        }
    }
}

/// Sound, first-order-free (fully rigorous) bound propagation rules.
pub mod bound {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    pub struct Overflow {
        pub magnitude: f64,
        pub max: f64,
    }

    pub fn gamma(k: f64) -> f64 {
        let ku = k * U32;
        if ku >= 1.0 { f64::INFINITY } else { ku / (1.0 - ku) }
    }

    pub fn quant(b: Bound, f: Format) -> Result<Bound, Overflow> {
        let reach = b.m + b.e;
        match f.err_model() {
            ErrModel::PerElement { u, floor, max } => {
                if reach > max {
                    return Err(Overflow { magnitude: reach, max });
                }
                Ok(Bound { m: b.m, e: b.e + u * reach + floor })
            }
            ErrModel::Scaled { c } => Ok(Bound { m: b.m, e: b.e + c * reach }),
        }
    }

    /// `[.., k] @ [k, ..]` accumulated in f32.
    pub fn matmul(k: usize, a: Bound, b: Bound) -> Bound {
        let k = k as f64;
        let m = k * a.m * b.m;
        let e = k * (a.e * b.m + a.m * b.e + a.e * b.e) + gamma(k) * k * (a.m + a.e) * (b.m + b.e);
        Bound { m, e }
    }

    pub fn add(a: Bound, b: Bound) -> Bound {
        let m = a.m + b.m;
        let e = a.e + b.e;
        Bound { m, e: e + U32 * (m + e) }
    }

    pub fn scale(a: Bound, c: f64) -> Bound {
        let c = c.abs();
        Bound { m: c * a.m, e: c * a.e + 2.0 * U32 * c * (a.m + a.e) * 1.0001 }
    }

    pub fn relu(a: Bound) -> Bound {
        a
    }

    /// SiLU is 1.0998-Lipschitz and |silu(x)| ≤ |x|.
    pub fn silu(a: Bound) -> Bound {
        Bound { m: a.m, e: 1.1 * a.e + 8.0 * U32 * (a.m + a.e) }
    }
}
