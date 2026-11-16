# Superfast 1-Norm Estimation

MATLAB implementation and numerical benchmark suite for the paper **"Can we accelerate LAPACK's 1-norm estimator?"** by Soo Go and Victor Y. Pan ([arXiv:2505.20528](https://arxiv.org/abs/2505.20528)).

## Algorithms
- **`sparsified_1norm_est.m`**: $k$-sparsified variant of LAPACK's 1-norm power method supporting `'random'` and `'even'` (equispaced) sampling modes.
- **`sparsified_1norm_est_hybrid.m`**: Hybrid variant combining $k$ random and $k$ equispaced probes per iteration.
- **`maxvol_1.m`**: Rank-1 `maxvol` cross-approximation for fast entrywise maximum search.
- **`sparsified_1norm_est_w_ca.m` / `sparsified_1norm_est_w_ca_hybrid.m`**: Sparsified 1-norm estimators warm-started by `maxvol_1`.

## Benchmarks
1. Place [Anymatrix](https://github.com/mmft/anymatrix) on your MATLAB path (required for `regtools/shaw` and `regtools/gravity`).
2. Run `run_benchmarks.m` to reproduce the accuracy and timing experiments (set `QUICK = true` for a fast smoke test).
3. Run `make_tables.m` on the saved `.mat` results to generate the LaTeX tables from the paper.

## Citation
```bibtex
@misc{go2025accelerate,
  title={Can we accelerate {LAPACK}'s 1-norm estimator?},
  author={Soo Go and Victor Y. Pan},
  year={2025},
  eprint={2505.20528},
  archivePrefix={arXiv},
  primaryClass={math.NA}
}
