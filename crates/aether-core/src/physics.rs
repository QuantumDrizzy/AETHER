//! Closed-form reference physics for surfacing in the GUI/CLI.
//!
//! Currently: graphene nearest-neighbor tight-binding bands, E = ±t·|f(k)|.
//! This mirrors the Python reference in `research/electronic_structure` and is
//! kept tiny and unit-tested (bandwidth 6t, Dirac point on the path).

use std::f64::consts::PI;

/// Graphene bands along the high-symmetry path Γ → K → M → Γ.
///
/// Returns `(distance, e_lower, e_upper)` in (1/Å path length, eV, eV), with
/// `n_per_seg` samples per segment. Closed form: `E±(k) = ± t·|Σ_δ e^{i k·δ}|`.
pub fn graphene_band_path(t: f64, a_cc: f64, n_per_seg: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let s3 = 3.0_f64.sqrt();
    // nearest-neighbor vectors A -> B (Å)
    let deltas = [
        [a_cc, 0.0],
        [-a_cc / 2.0, a_cc * s3 / 2.0],
        [-a_cc / 2.0, -a_cc * s3 / 2.0],
    ];
    let f_abs = |k: [f64; 2]| -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for d in deltas {
            let ph = k[0] * d[0] + k[1] * d[1];
            re += ph.cos();
            im += ph.sin();
        }
        (re * re + im * im).sqrt()
    };

    let g = 2.0 * PI / (3.0 * a_cc);
    let corners = [[0.0, 0.0], [g, g / s3], [g, 0.0], [0.0, 0.0]]; // Γ, K, M, Γ

    let (mut dist, mut lo, mut hi) = (Vec::new(), Vec::new(), Vec::new());
    let mut d0 = 0.0;
    for seg in 0..3 {
        let (a, b) = (corners[seg], corners[seg + 1]);
        let seglen = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        for i in 0..n_per_seg {
            let frac = i as f64 / n_per_seg as f64;
            let k = [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac];
            let e = t * f_abs(k);
            dist.push(d0 + seglen * frac);
            lo.push(-e);
            hi.push(e);
        }
        d0 += seglen;
    }
    let e = t * f_abs(corners[3]);
    dist.push(d0);
    lo.push(-e);
    hi.push(e);
    (dist, lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_is_6t_and_path_hits_a_dirac_point() {
        let t = 2.7;
        let (_d, lo, hi) = graphene_band_path(t, 1.42, 200);
        // Γ endpoints give |E| = 3t -> full bandwidth 6t.
        let max_hi = hi.iter().cloned().fold(f64::MIN, f64::max);
        assert!((max_hi - 3.0 * t).abs() < 1e-6, "bandwidth != 6t");
        // The path passes through K, where the gap closes.
        let min_gap = hi.iter().zip(&lo).map(|(h, l)| h - l).fold(f64::MAX, f64::min);
        assert!(min_gap < 1e-6, "no Dirac point on the path: gap={min_gap}");
    }
}
