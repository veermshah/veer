//! The static error bounds must never be smaller than the measured error.
use proptest::prelude::*;
use veer_numerics::{bound, tensor::Tensor, Bound, Format, ALL_FORMATS};

fn fmt_strategy() -> impl Strategy<Value = Format> {
    (0..ALL_FORMATS.len()).prop_map(|i| ALL_FORMATS[i])
}

fn check(t: &Tensor, b: Bound, what: &str) {
    let measured = t.measured_err();
    assert!(t.max_abs() <= b.m * (1.0 + 1e-12) + 1e-300, "{what}: magnitude {} > bound {}", t.max_abs(), b.m);
    assert!(measured <= b.e * (1.0 + 1e-9) + 1e-300, "{what}: measured err {measured:e} > bound {:e}", b.e);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn quantize_is_sound(fmt in fmt_strategy(), n in 1usize..200, mag in 1e-6f64..400.0, seed in any::<u64>()) {
        let t = Tensor::rand_uniform(vec![n], -mag, mag, seed);
        let b = bound::quant(Bound::exact(mag), fmt).unwrap();
        check(&t.quant(fmt), b, fmt.name());
    }

    #[test]
    fn quantized_mlp_is_sound(
        fx in fmt_strategy(), fw in fmt_strategy(),
        m in 1usize..4, k in 1usize..96, n in 1usize..24,
        mx in 0.01f64..8.0, mw in 0.01f64..2.0, seed in any::<u64>(),
    ) {
        let x = Tensor::rand_uniform(vec![m, k], -mx, mx, seed);
        let w = Tensor::rand_uniform(vec![k, n], -mw, mw, seed ^ 0xABCD);
        let bias = Tensor::rand_uniform(vec![1, n], -1.0, 1.0, seed ^ 7);
        let bx = bound::quant(Bound::exact(mx), fx).unwrap();
        let bw = bound::quant(Bound::exact(mw), fw).unwrap();
        let xq = x.quant(fx);
        let wq = w.quant(fw);
        let y = xq.matmul(&wq);
        let by = bound::matmul(k, bx, bw);
        check(&y, by, "matmul");
        let z = y.add(&bias).silu().scale(0.5);
        let bz = bound::scale(bound::silu(bound::add(by, Bound::exact(1.0))), 0.5);
        check(&z, bz, "add/silu/scale");
        let r = z.relu();
        check(&r, bound::relu(bz), "relu");
    }
}

#[test]
fn mxfp4_clamp_is_the_worst_case() {
    // amax just below 8·2^k makes OCP MXFP4 clamp: exactly the issue the
    // overflow-aware variant fixes. The bound must still hold, and OAS must be better.
    let vals: Vec<f32> = (0..32).map(|i| if i == 0 { 7.9 } else { 7.0 }).collect();
    let t = Tensor::from_values(vec![32], &vals);
    let e_ocp = t.quant(Format::Mxfp4).measured_err();
    let e_oas = t.quant(Format::Mxfp4Oas).measured_err();
    assert!(e_ocp > e_oas, "ocp {e_ocp} oas {e_oas}");
    assert!(e_ocp <= bound::quant(Bound::exact(7.9), Format::Mxfp4).unwrap().e);
}

#[test]
fn fp8_overflow_is_detected() {
    assert!(bound::quant(Bound::exact(1000.0), Format::Fp8E4m3).is_err());
    assert!(bound::quant(Bound::exact(1000.0), Format::Nvfp4).is_ok());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]
    /// Wildly heterogeneous magnitudes stress block scales (tiny blocks next to huge ones).
    #[test]
    fn quantize_heterogeneous_blocks(fmt in fmt_strategy(), vals in prop::collection::vec((-1.0f64..1.0, 0i32..14), 1..120)) {
        let xs: Vec<f32> = vals.iter().map(|(v, e)| (v * 10f64.powi(-e) * 300.0) as f32).collect();
        let t = Tensor::from_values(vec![xs.len()], &xs);
        let b = bound::quant(Bound::exact(t.max_abs()), fmt).unwrap();
        check(&t.quant(fmt), b, fmt.name());
    }
}
