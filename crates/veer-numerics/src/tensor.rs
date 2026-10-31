//! A tiny CPU tensor that carries both the *computed* values (f32, after
//! simulated quantization) and a *shadow* of the exact values (f64), so the
//! real error of any computation can be measured and compared to its static
//! [`crate::Bound`].

use crate::Format;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub fmt: Format,
    pub data: Vec<f32>,
    pub exact: Vec<f64>,
}

/// SplitMix64: small deterministic PRNG so programs are reproducible.
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl Tensor {
    pub fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = Self::numel(&shape);
        Tensor { shape, fmt: Format::F32, data: vec![0.0; n], exact: vec![0.0; n] }
    }

    pub fn rand_uniform(shape: Vec<usize>, lo: f64, hi: f64, seed: u64) -> Tensor {
        let n = Self::numel(&shape);
        let mut rng = Rng::new(seed);
        let data: Vec<f32> = (0..n).map(|_| (lo + (hi - lo) * rng.next_f64()) as f32).map(|x| x.clamp(lo.min(hi) as f32, lo.max(hi) as f32)).collect();
        let exact = data.iter().map(|&x| x as f64).collect();
        Tensor { shape, fmt: Format::F32, data, exact }
    }

    pub fn from_values(shape: Vec<usize>, values: &[f32]) -> Tensor {
        Tensor { shape, fmt: Format::F32, data: values.to_vec(), exact: values.iter().map(|&x| x as f64).collect() }
    }

    pub fn quant(&self, fmt: Format) -> Tensor {
        Tensor { shape: self.shape.clone(), fmt, data: fmt.quantize(&self.data), exact: self.exact.clone() }
    }

    /// 2-D matmul `[m, k] @ [k, n]`, f32 accumulation (no FMA), result in f32.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, k) = (self.shape[0], self.shape[1]);
        let n = other.shape[1];
        assert_eq!(k, other.shape[0], "matmul inner dimension");
        let mut data = vec![0f32; m * n];
        let mut exact = vec![0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                let mut ex = 0f64;
                for t in 0..k {
                    let p = self.data[i * k + t] * other.data[t * n + j];
                    acc += p;
                    ex += self.exact[i * k + t] * other.exact[t * n + j];
                }
                data[i * n + j] = acc;
                exact[i * n + j] = ex;
            }
        }
        Tensor { shape: vec![m, n], fmt: Format::F32, data, exact }
    }

    /// Elementwise add; `other` may be a row vector broadcast over rows.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let n_other = other.data.len();
        let data = self.data.iter().enumerate().map(|(i, &a)| a + other.data[i % n_other]).collect();
        let exact = self.exact.iter().enumerate().map(|(i, &a)| a + other.exact[i % n_other]).collect();
        Tensor { shape: self.shape.clone(), fmt: Format::F32, data, exact }
    }

    pub fn scale(&self, c: f64) -> Tensor {
        let c32 = c as f32;
        Tensor {
            shape: self.shape.clone(),
            fmt: Format::F32,
            data: self.data.iter().map(|&x| c32 * x).collect(),
            exact: self.exact.iter().map(|&x| c * x).collect(),
        }
    }

    pub fn relu(&self) -> Tensor {
        self.map(|x| x.max(0.0))
    }

    pub fn silu(&self) -> Tensor {
        self.map(|x| x / (1.0 + (-x).exp()))
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            fmt: Format::F32,
            data: self.data.iter().map(|&x| f(x as f64) as f32).collect(),
            exact: self.exact.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Measured ‖computed − exact‖∞.
    pub fn measured_err(&self) -> f64 {
        self.data.iter().zip(&self.exact).fold(0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()))
    }

    /// ‖exact‖∞.
    pub fn max_abs(&self) -> f64 {
        self.exact.iter().fold(0f64, |m, &x| m.max(x.abs()))
    }
}
